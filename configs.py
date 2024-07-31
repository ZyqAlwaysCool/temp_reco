'''
desc: dj红外图像温度识别相关配置
'''

# 模型配置
MODEL_CONFIGS = {
    'models': {
        'taoguan_det': './model_weights/v1/taoguan_det.pt',
        'taoguan_seg': './model_weights/v1/taoguan_seg.pt',
        'xianzhang_seg': './model_weights/v1/xianzhang_seg.pt',
    },
}

# 绘图配置
DRAW_CONFIGS = {
    'draw_color': {
        'xianzhang': (0, 255, 0),
        'taoguan': (0, 0, 255),
    },
    'draw_scale': 0.5,
    'save_pic_path': './draw_res',
    'is_show': True,
}

# 图像裁剪配置
CROPPED_CONFIGS = {
    'cropped_save_path': './cropped_res',
    'is_show': False,
}

DEFAULT_CONFIG = {
    'save_path': './tmp_res',
    'is_show': False,
    'enable_taoguan': True, #检测套管目标
    'enable_xianzhang': True, #检测线掌目标
}

