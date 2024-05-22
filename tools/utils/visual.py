import cv2 
import numpy as np 

def visual_one_image_detect(boxes, confs, class_id, image):
    draw = image.copy()
    for i, box in enumerate(boxes):
        if len(box) == 0:
            continue
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)
        thickness = 2
        draw = cv2.rectangle(draw, start_point, end_point, color, thickness) 
    return draw

def visual_one_image_tracking(boxes, object_ids, confs, class_id, image):
    draw = image.copy()
    for i, box in enumerate(boxes):
        if len(box) == 0:
            continue
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        box_color = (0, 255, 0)
        text_color = (255, 0, 0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontScale = 1
        draw = cv2.rectangle(draw, start_point, end_point, box_color, thickness)

        draw = cv2.putText(draw, str(int(object_ids[i])), start_point, font,  
                   fontScale, text_color, thickness, cv2.LINE_AA) 
    return draw
