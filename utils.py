'''
desc: 工具函数
'''

import os
import random
import time
import hashlib
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale

from configs import DRAW_CONFIGS, MODEL_CONFIGS, CROPPED_CONFIGS

def gen_md5_info(prefix):
    '''
    基于前缀+时间戳+随机数生成md5信息
    '''
    content = '{}_{}_{}'.format(prefix, int(time.time()), random.randint(0, 1000000))
    md5hash = hashlib.md5(content.encode('utf-8'))
    return md5hash.hexdigest()

def draw_boxes_and_labels(image_path, boxes, temps):
    """
    在红外图像上绘制矩形框和对应温度数值。
    
    :param image_path: 原始图像路径
    :param boxes: 矩形框的坐标列表，每个矩形框是一个元组 (x1, y1, x2, y2)
    :param temps: 每个矩形框对应的温度数值列表
    """

    image = cv2.imread(image_path)
    
    # # 确保图像是BGR格式
    # if len(image.shape) == 3 and image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 绘制矩形框和标签
    for (x1, y1, x2, y2, name), temp in zip(boxes, temps):
        # 绘制矩形框
        if name == 'xianzhang':
            cv2.rectangle(image, (x1, y1), (x2, y2), DRAW_CONFIGS['draw_color']['xianzhang'], 2)
        elif name == 'taoguan':
            cv2.rectangle(image, (x1, y1), (x2, y2), DRAW_CONFIGS['draw_color']['taoguan'], 2)
    
        # 获取标签名称
        text_content = str(temp)
    
        # 绘制文本
        cv2.putText(image, text_content, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, DRAW_CONFIGS['draw_scale'], (255, 255, 255), 2, cv2.LINE_AA)
    
    # 保存图像
    if not os.path.exists(DRAW_CONFIGS['save_pic_path']):
        os.makedirs(DRAW_CONFIGS['save_pic_path'])
    save_fname = DRAW_CONFIGS['save_pic_path'] + '/' + 'draw_temp_{}.jpg'.format(gen_md5_info('DRAW_TEMP'))
    cv2.imwrite(save_fname, image)

def check_ir_image(image_path: str):
    '''
    判断输入图像是否为红外图像
    
    :Params image: 原始图像
    '''
    
    image = cv2.imread(image_path)
    
    imagecv = cv2.resize(image, (320, 320))
    # 将图像从BGR转换到HSV颜色空间
    hsv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2HSV)
    # 设置红色的阈值范围
    # 注意：这些值可能需要根据您的图像进行调整
    lower_red = np.array([0, 80, 70])
    upper_red = np.array([10, 255, 255])
    # 根据阈值创建一个掩模
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # # 对原始图像和掩模进行位运算，提取红色区域
    red_only = cv2.bitwise_and(imagecv, imagecv, mask=mask)
    red_sum = np.sum(red_only)

    if red_sum >= 578230:
        return True
    
    return False

def crop_img(image, box, save=False):
    '''
    裁剪图像
    
    :params image: 原始图像
    :params box: 待裁剪的区域, 形如[x1, y1, x2, y2]
    
    return
        cropped_image: 裁剪后的图像
    '''
    if len(box) != 4:
        raise ValueError("box must be a list of 4 elements. eg: [x1,y1,x2,y2]")
    
    x1, y1, x2, y2 = box
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

    if save:
        if not os.path.exists(CROPPED_CONFIGS['cropped_save_path']):
            os.makedirs(CROPPED_CONFIGS['cropped_save_path'])
        save_fname = CROPPED_CONFIGS['cropped_save_path'] + '/' + 'cropped_img_{}.jpg'.format(gen_md5_info('CROPPED_IMG'))
        cv2.imwrite(save_fname, cropped_image)
    
    return cropped_image

def shrink_contour(contour: np.ndarray, shrink_ratio=0.9):
    '''
    基于比例对原始图像轮廓进行缩放
    
    :params contour: 原始图像轮廓 形如 [[[x1, y1]], [[x2, y2]], ...]
    :params shrink_ratio: 缩放比例
    
    return
    '''
    poly_coord = [c[0] for c in contour]
    polygon = Polygon(poly_coord)
    centroid = polygon.centroid
    scaled_polygon = scale(polygon, xfact=shrink_ratio, yfact=shrink_ratio, origin=centroid)
    
    reshaped_contours_coord = []
    for point in scaled_polygon.exterior.coords:
        reshaped_contours_coord.append(point)
    shrink_contour = np.array(reshaped_contours_coord).astype(np.int32).reshape(-1, 1, 2)
    return shrink_contour

def transform_coordinate(x_prime, y_prime, crop_area, original_size):
    """
    将裁剪图像上的坐标点转换为原始图像上的坐标点。
    
    参数:
    x_prime: 裁剪图像上的x坐标。
    y_prime: 裁剪图像上的y坐标。
    crop_area: 裁剪区域，由(x1, y1, x2, y2)定义。
    original_size: 原始图像的尺寸，由(width, height)定义。注意, 是w*h, cv2.imread读取的图像尺寸是h*w
    
    返回:
    转换后的坐标点(x, y)。
    """
    x1, y1, x2, y2 = crop_area
    original_width, original_height = original_size
    
    # 转换坐标点
    x = x_prime + x1
    y = y_prime + y1
    
    # 检查并调整坐标点是否超出原始图像边界
    x = min(max(x, 0), original_width - 1)
    y = min(max(y, 0), original_height - 1)
    
    return x, y

def draw_contour_and_labels(image_path: str, contour_list: list, temps: list):
    """
    在红外图像上绘制矩形框和对应温度数值。
    
    :param image_path: 原始图像路径
    :param contour_list: 目标的轮廓坐标集, list
    :param temps: 每个目标轮廓对应的温度数值列表
    """

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # # 确保图像是BGR格式
    # if len(image.shape) == 3 and image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 绘制分割轮廓和区域最高温度值
    for (contour, name), temp in zip(contour_list, temps):
        if name == 'xianzhang':
            _ = cv2.drawContours(image, [contour], 0, DRAW_CONFIGS['draw_color']['xianzhang'], 2)
        elif name == 'taoguan':
            _ = cv2.drawContours(image, [contour], 0, DRAW_CONFIGS['draw_color']['taoguan'], 2)
        
        x, y, w, h = cv2.boundingRect(contour)
        text_position = (x + w // 2 - 5, y + h // 2) #文本输出位置
        text = str(temp)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = DRAW_CONFIGS['draw_scale']
        font_color = (255, 255, 255)  # 白色文本
        #line_type = 2
        thickness = 1 #文本粗细
        line_type = cv2.LINE_AA
        
    
        # 绘制文本
        cv2.putText(image, text, text_position, font, font_scale, font_color, thickness, line_type, bottomLeftOrigin=False)
    
    # 保存图像
    if not os.path.exists(DRAW_CONFIGS['save_pic_path']):
        os.makedirs(DRAW_CONFIGS['save_pic_path'])
    save_fname = DRAW_CONFIGS['save_pic_path'] + '/' + 'draw_temp_{}.jpg'.format(gen_md5_info('DRAW_TEMP'))
    cv2.imwrite(save_fname, image)