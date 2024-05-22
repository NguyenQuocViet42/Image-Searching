import cv2
import numpy as np
import onnxruntime 
import math 

from tools.face_detect.yolov5.utils import xywh2xyxy, nms, non_max_suppression_face

class YOLOv5Face:
    def __init__(self, model_path, conf_thresh=0.8, iou_thresh=0.5, infer_shape=(640, 640), fp16=False, scale_roi=20):
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.infer_shape = infer_shape
        self.fp16 = fp16
        self.img_height = []
        self.img_width = []
        self.scale_ratio = []
        self.dw = []
        self.dh = []
        self.scale_roi = scale_roi
        self.max_angle = 35
        self.initialize_model(model_path)
        
    def __call__(self, images):
        return self.detect_objects(images)
    
    def detect_objects(self, images):
        input_tensor = np.array([])
        
        for image in images:
            input_image = self.prepare_input(image)
            if input_tensor.shape[0] == 0:
                input_tensor = input_image
            else:
                input_tensor = np.concatenate((input_tensor, input_image), axis=0)

        outputs = self.inference(input_tensor)
        
        self.boxes, self.scores, self.class_ids, self.landmarks = self.process_output(outputs)
        
        return self.boxes, self.scores, self.class_ids, self.landmarks
    
    def initialize_model(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def prepare_input(self, image):
        img_height, img_width = image.shape[:2]
        self.img_height.append(img_height)
        self.img_width.append(img_width)

        # input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = self.reshape_input(image)

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        if self.fp16:
            input_tensor = input_img[np.newaxis, :, :, :].astype(np.float16)
        else:
            input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def reshape_input(self, image):
        new_shape = self.infer_shape
        image_shape = image.shape[:2]

        # Scale ratio (new / old)
        r = min(new_shape[0] / image_shape[0], new_shape[1] / image_shape[1])
        self.scale_ratio.append(r) 

        # Compute padding
        new_unpad = int(round(image_shape[1] * r)), int(round(image_shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if image_shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        self.dh.append(dh)
        self.dw.append(dw)
        return image
    
    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs
    
    def process_output(self, output):
        predictions = np.squeeze(output[0])
        
        if len(predictions.shape) < 3:
            if self.fp16:
                predictions = predictions[np.newaxis, :, :].astype(np.float16)
            else:
                predictions = predictions[np.newaxis, :, :].astype(np.float32)
        
        predictions = non_max_suppression_face(predictions, self.conf_threshold, self.iou_threshold)

        out_boxes = []
        out_scores = []
        out_class = []
        out_landmarks = []
        for i, pred in enumerate(predictions):
            boxes = pred[:, :4]
            confs = pred[:, 4].cpu().detach().numpy()
            landmarks = pred[:, 5:15].cpu().detach().numpy() #Chua dung
            class_ids = pred[:, 15].cpu().detach().numpy()
            temp_box = []
            temp_landmarks = []
            if len(confs) == 0:
                out_boxes.append([])
                out_scores.append([])
                out_class.append([])
                out_landmarks.append([])
                continue

            for j, box in enumerate(boxes):
                right_mouth = [landmarks[i][6], landmarks[i][7]]
                left_mouth = [landmarks[i][8], landmarks[i][9]]
                right_eye = [landmarks[i][0], landmarks[i][1]]
                left_eye = [landmarks[i][2], landmarks[i][3]]
                nose = [landmarks[i][4], landmarks[i][5]]
                last_right_point_crop = [box[2] - box[0] - 1, right_eye[1]]
                out_landmarks.append([left_eye, right_eye, left_mouth, right_mouth, nose])
                angle = self.calculate_angle_between_points(right_eye, left_eye, last_right_point_crop)
                
                if (angle > self.max_angle or angle == -1):
                    continue
                
                if not self.check_suitable_face(right_mouth, left_mouth, nose, right_eye, left_eye):
                    continue

                if self.dh[i] > 0.0:
                    bx1 = box[0] / self.scale_ratio[i]
                    bx2 = box[2] / self.scale_ratio[i]
                    by1 = (box[1] - self.dh[i]) / self.scale_ratio[i]
                    by2 = (box[3] - self.dh[i]) / self.scale_ratio[i]
                elif self.dw[i] > 0.0:
                    bx1 = (box[0] - self.dw[i]) / self.scale_ratio[i]
                    bx2 = (box[2] - self.dw[i]) / self.scale_ratio[i]
                    by1 = box[1] / self.scale_ratio[i]
                    by2 = box[3] / self.scale_ratio[i]
                else:
                    bx1, bx2, by1, by2 = box[0], box[2], box[1], box[3]
                
                if self.scale_roi is not None:
                    half_x = (bx2 - bx1) / self.scale_roi
                    half_y = (by2 - by1) / self.scale_roi
                    bx1 -= half_x
                    by1 -= half_y
                    bx2 += half_x
                    by2 += half_y
                
                bx1 = max(min(bx1, self.img_width[i]), 0)
                bx2 = max(min(bx2, self.img_width[i]), 0)
                by1 = max(min(by1, self.img_height[i]), 0)
                by2 = max(min(by2, self.img_height[i]), 0)

                temp_box.append([bx1, by1, bx2, by2])

            out_boxes.append(np.array(temp_box))
            out_landmarks.append(np.array(temp_landmarks))
            out_scores.append(confs)
            out_class.append(class_ids)
            
        return out_boxes, out_scores, out_class, out_landmarks
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes
    
    def calculate_angle_between_points(self, p1, p2, p3):
        m1 = self.calculate_slop_between_points(p2, p1)
        m2 = self.calculate_slop_between_points(p3, p1)

        if ((1 + m1*m2) == 0 or m1 == -1 or m2 == -1):
            return -1
        
        tan_angle = (m2 - m1) / (1 + m1*m2)
        degree_angle = math.atan(tan_angle/math.pi) * 180
        return abs(degree_angle)

    def calculate_slop_between_points(self, p1, p2):
        if (p2[0] - p1[0]) == 0:
            return -1
        else:
            return (p2[1] - p1[1]) / (p2[0] - p1[0])
    
    def cross_product(self, p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def check_suitable_face(self, right_mouth, left_mouth, nose, right_eye, left_eye):
        cross_right = self.cross_product(right_mouth, right_eye, nose)
        cross_left = self.cross_product(left_mouth, left_eye, nose)

        if (cross_right * cross_left) > 0:
            return False
        
        topY = min(right_eye[1], left_eye[1])
        bottomY = max(right_mouth[1], left_mouth[1])

        if (nose[1] <= topY or nose[1] >= bottomY):
            return False 
        
        return True