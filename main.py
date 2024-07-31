import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from predictor import YoloIRPredictor
from utils import *
from configs import *

def ir_temp_reco(image_path: str):
    '''
    大疆拍摄红外图像温度识别
    '''
    if not check_ir_image(image_path):
        raise Exception('input image is not a valid infrared image')
    
    yolo_ir_predictor = YoloIRPredictor(MODEL_CONFIGS['models']['taoguan_det'],
                                        MODEL_CONFIGS['models']['taoguan_seg'], 
                                        MODEL_CONFIGS['models']['xianzhang_seg'])
    boxes_list, temp_list = yolo_ir_predictor.predict(image_path)
    assert(len(boxes_list) == len(temp_list))
    
    cnt = len(boxes_list)
    
    if DRAW_CONFIGS['is_show']:
        draw_boxes_and_labels(image_path, boxes_list, temp_list)
    else:
        for i in range(cnt):
            print('boxes_info=({}) temp=({})'.format(boxes_list[i], temp_list[i]))
            

if __name__ == '__main__':
    img_path = './img'
    for img in os.listdir(img_path):
        real_img_path = img_path + '/' + img
        ir_temp_reco(real_img_path)