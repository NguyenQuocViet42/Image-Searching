import os 
import yaml
import torch 

def get_tools_elements(phase, type_name, config_path):
    if phase == 'detect':
        return get_detect_model(type_name, config_path)
    elif phase == 'tracking':
        return get_tracking_type(type_name, config_path)

def get_detect_model(model_name, model_config):
    config = open(model_config, 'r')
    config_params = yaml.safe_load(config)
    detect_model = None 

    if model_name == 'yolov5-face':
        from tools.face_detect.yolov5.model import YOLOv5Face

        model_path = config_params['model_path']
        detect_thresh = config_params['detect_thresh']
        scale_up_roi = config_params['scale_roi']
        match_thresh = config_params['match_thresh']
        fp16 = config_params['fp16']
        img_size = config_params['img_size']
        infer_shape = (img_size, img_size)
        detect_model = YOLOv5Face(model_path, detect_thresh, match_thresh, scale_roi=scale_up_roi, fp16=fp16, infer_shape=infer_shape)

    elif model_name == 'yolov5-person':
        from tools.person_detect.yolov5.model import YOLOv5Person

        model_path = config_params['model_path']
        detect_thresh = config_params['detect_thresh']
        match_thresh = config_params['match_thresh']
        fp16 = config_params['fp16']
        img_size = config_params['img_size']
        infer_shape = (img_size, img_size)
        detect_model = YOLOv5Person(model_path, detect_thresh, match_thresh, fp16=fp16, infer_shape=infer_shape)

    return detect_model

def get_tracking_type(tracking_name, track_config):
    config = open(track_config, 'r')
    config_params = yaml.safe_load(config)
    tracker = None

    if tracking_name == 'bytetrack':
        from tools.tracking.bytetrack.byte_tracker import BYTETracker

        detect_thresh = config_params['detect_thresh']
        track_high_thresh = config_params['track_high_thresh']
        track_buffer = config_params['track_buffer']
        match_thresh = config_params['match_thresh']
        frame_rate = config_params['frame_rate']
        tracker = BYTETracker(detect_thresh, match_thresh, track_buffer, frame_rate, track_high_thresh)

    elif tracking_name == 'sparsetrack':
        from tools.tracking.sparsetrack.sparse_tracker import SparseTracker

        frame_rate = config_params['frame_rate']
        detect_thresh = config_params['detect_thresh']
        track_buffer = config_params['track_buffer']
        depth_levels = config_params['depth_levels']
        high_thresh = config_params['high_thresh']
        match_thresh = config_params['match_thresh']
        down_scale = config_params['down_scale']
        depth_levels_low = config_params['depth_levels_low']
        confirm_thresh = config_params['confirm_thresh']
        tracker = SparseTracker(frame_rate, detect_thresh, track_buffer, depth_levels, high_thresh, 
                                match_thresh, down_scale, depth_levels_low, confirm_thresh)
    
    return tracker 
