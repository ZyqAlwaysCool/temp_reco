'''
desc: 红外图像预测模型
'''

import os
import cv2
from ultralytics import YOLO
import math
from utils import *
from configs import *
from thermal_parser.thermal_parser.thermal import *

class YoloIRPredictor:
    def __init__(self, taoguan_det_model, taoguan_seg_model, xianzhang_seg_model):
        self.taoguan_det_model = YOLO(taoguan_det_model)
        self.taoguan_seg_model = YOLO(taoguan_seg_model)
        self.xianzhang_seg_model = YOLO(xianzhang_seg_model)
    
    def __get_yolo_pred_info(self, yolo_pred_res):
        '''
        获取yolo预测结果信息
        
        :params yolo_pred_res: yolo预测原始结果

        return
            boxes: xyxy形式预测框, 包含左上和右下坐标
            classes: 预测的类别
            names: 预测类别对应的名称
            confidences: 预测置信度得分
        '''
        boxes = yolo_pred_res[0].boxes.xyxy.tolist()
        classes = yolo_pred_res[0].boxes.cls.tolist()
        names = yolo_pred_res[0].names
        confidences = yolo_pred_res[0].boxes.conf.tolist()
        
        return boxes, classes, names, confidences
    
    def __isolate_target(self, yolo_pred_res, save=False):
        '''
        将图像目标区域提取,其他区域置为黑色
        
        :params yolo_pred_res: yolo预测结果
        :params save: 是否保存图像结果

        return
            mask3ch: mask二值图像,[0, 255]
        '''

        for r in yolo_pred_res:
            img = np.copy(r.orig_img)
            #img_name = Path(r.path).stem

            # Iterate each object contour (multiple detections)
            for ci, c in enumerate(r):
                #  Get detection class name
                label = c.names[c.boxes.cls.tolist().pop()]

                # Create binary mask
                b_mask = np.zeros(img.shape[:2], np.uint8)

                #  Extract contour result
                contour = c.masks.xy.pop()
                #  Changing the type
                contour = contour.astype(np.int32)
                #  Reshaping
                contour = contour.reshape(-1, 1, 2)


                # Draw contour onto mask
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                
                # Create 3-channel mask
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR) #二值mask图像

                # Isolate object with binary mask
                isolated = cv2.bitwise_and(mask3ch, img)
                
                if save:
                    if not os.path.exists(DEFAULT_CONFIG['save_path']):
                        os.makedirs(DEFAULT_CONFIG['save_path'])
                    save_fname = DEFAULT_CONFIG['save_path'] + '/' + "isolated_{}_{}.JPG".format(label, gen_md5_info('ISOLATED'))
                    cv2.imwrite(save_fname, isolated)
                
                return mask3ch
    
    def __seg_taoguan(self, image: np.ndarray, conf=0.5):
        '''
        分割套管目标, 如果检测到套管目标, 则将目标物体隔离出来, 周围像素置为黑色区域, 用于减少周围区域对后续温度识别的干扰
        
        :params image: 图像, 格式为np.ndarray
        :params conf: 置信度, 默认为0.5
        '''
        result = self.taoguan_seg_model.predict(image, conf=conf)

        if len(result[0].boxes.xyxy.tolist()) > 0:
            return self.__isolate_target(result, DEFAULT_CONFIG['is_show'])
        else:
            raise Exception('no taoguan target found.')
    
    def __seg_xianzhang(self, image: np.ndarray, conf=0.5):
        '''
        分割线掌目标, 如果检测到线掌目标, 则将目标物体隔离出来, 周围像素置为黑色区域, 用于减少周围区域对后续温度识别的干扰
        
        :params image: 图像, 格式为np.ndarray
        :params conf: 置信度, 默认为0.5
        '''
        result = self.xianzhang_seg_model.predict(image, conf=conf)

        if len(result[0].boxes.xyxy.tolist()) > 0:
            return self.__isolate_target(result, DEFAULT_CONFIG['is_show'])
        else:
            raise Exception('no xianzhang target found.')
    
    def predict(self, image_path: str, conf=0.5, visual=False):
        thermal = Thermal(dtype=np.float32)
        temperature_mat = thermal.parse_dirp2(filepath_image=image_path) #基于原始图像获取温度矩阵, 代表每个像素点的温度值
        
        image = cv2.imread(image_path) 
        # 获取套管目标识别结果
        taoguan_det_result = self.taoguan_det_model.predict(image, save=visual, conf=conf)
        boxes, classes, names, confidences = self.__get_yolo_pred_info(taoguan_det_result)

        # 基于套管目标识别结果, 分割出具体的目标轮廓
        cropped_img_mask = None
        boxes_list = []
        temp_list = []
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            confidence = conf
            detected_class = cls
            name = names[int(cls)]

            if not DEFAULT_CONFIG['enable_taoguan'] and name == 'taoguan':
                continue
            if not DEFAULT_CONFIG['enable_xianzhang'] and name == 'xianzhang':
                continue

            x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2) #向上取整
            #print("x1: {}, y1: {}, x2: {}, y2: {}, confidence: {}, detected_class: {}, name: {}".format(x1, y1, x2, y2, confidence, detected_class, name))


            #基于预测box裁剪出识别的目标区域, 并获取目标分割后的mask图像
            cropped_img = crop_img(image, box, CROPPED_CONFIGS['is_show'])
            
            if name == 'taoguan':
                cropped_img_mask = self.__seg_taoguan(cropped_img)
            elif name == 'xianzhang':
                cropped_img_mask = self.__seg_xianzhang(cropped_img)
            else:
                raise Exception('unknown class: {}'.format(name))
            
            #分割后的mask二值图像中包含0和255, 255表示目标外区域, 属于干扰项, 在计算温度时需要剔除
            indices = np.where(cropped_img_mask == 255)
            no_cal_coord = list(set(zip(indices[0], indices[1]))) #np坐标的格式(行, 列),映射到图像是(y, x)
            cropped_temp_img = temperature_mat[y1:y2, x1:x2]

            for i in no_cal_coord:
                try:
                    cropped_temp_img[i[0]][i[1]] = 0 #将温度矩阵中对应的坐标置0,不参与最高温度计算
                except:
                    #边界坐标点处理
                    print('error coordinates=({}) image.shape=({})'.format(i, cropped_temp_img.shape))
                    cropped_temp_img[i[0]-1][i[1]-1] = 0
            
            max_temp = np.max(cropped_temp_img)

            boxes_list.append([x1, y1, x2, y2, name])
            temp_list.append(np.round(max_temp, 2))
        
        return boxes_list, temp_list
            
            
    
    
        
    