import cv2
import numpy as np
import onnxruntime 

from tools.person_detect.yolov5.utils import xywh2xyxy, nms

class YOLOv5Person:
    def __init__(self, model_path, conf_thresh=0.8, iou_thresh=0.5, infer_shape=(640, 640), fp16=False):
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.infer_shape = infer_shape
        self.fp16 = fp16
        self.img_height = []
        self.img_width = []
        self.scale_ratio = []
        self.dw = []
        self.dh = []
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
        
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        
        return self.boxes, self.scores, self.class_ids
    
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

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        predictions = np.squeeze(output[0]).T
        if len(predictions.shape) < 3:
            if self.fp16:
                predictions = predictions[:, :, np.newaxis].astype(np.float16)
            else:
                predictions = predictions[:, :, np.newaxis].astype(np.float32)
        
        scores = np.max(predictions[:, 4:, :], axis=1)

        out_boxes = []
        out_scores = []
        out_class = []
        for i in range(scores.shape[1]):
            h_scores = scores[:, i] > self.conf_threshold
            
            h_predictions = predictions[h_scores, :, i]
            s = scores[:, i][scores[:, i] > self.conf_threshold]
            if len(s) == 0:
                out_boxes.append([])
                out_scores.append([])
                out_class.append([])
                continue

            class_ids = np.argmax(h_predictions[:, 4:], axis=1)
            boxes = self.extract_boxes(h_predictions)
            indices = nms(boxes, s, self.iou_threshold)
            temp_box = boxes[indices].copy()

            for j in range(len(boxes[indices])):
                if self.dh[i] > 0.0:
                    temp_box[j][0] = boxes[indices][j][0] / self.scale_ratio[i]
                    temp_box[j][2] = boxes[indices][j][2] / self.scale_ratio[i]
                    temp_box[j][1] = (boxes[indices][j][1] - self.dh[i]) /self.scale_ratio[i]
                    temp_box[j][3] = (boxes[indices][j][3] - self.dh[i]) /self.scale_ratio[i]
                elif self.dw[i] > 0.0:
                    temp_box[j][0] = (boxes[indices][j][0] - self.dw[i] )/ self.scale_ratio[i]
                    temp_box[j][2] = (boxes[indices][j][2] - self.dw[i]) / self.scale_ratio[i]
                    temp_box[j][1] = boxes[indices][j][1] / self.scale_ratio[i]
                    temp_box[j][3] = boxes[indices][j][3] / self.scale_ratio[i]

                temp_box[j][0] = max(min(temp_box[j][0], self.img_width[i]), 0)
                temp_box[j][2] = max(min(temp_box[j][2], self.img_width[i]), 0)
                temp_box[j][1] = max(min(temp_box[j][1], self.img_height[i]), 0)
                temp_box[j][3] = max(min(temp_box[j][3], self.img_height[i]), 0)

            out_boxes.append(temp_box)
            out_scores.append(s[indices])
            out_class.append(class_ids[indices])

        # scores = np.max(predictions[:, 4:], axis=1)
        # h_scores = scores > self.conf_threshold
        # print(predictions.shape)
        # print(h_scores.shape)
        # predictions = predictions[scores > self.conf_threshold, :]
        # scores = scores[scores > self.conf_threshold]
        
        # if len(scores) == 0:
        #     return [], [], []
        
        # class_ids = np.argmax(predictions[:, 4:], axis=1)

        # boxes = self.extract_boxes(predictions)

        # indices = nms(boxes, scores, self.iou_threshold)
        # out_boxes = boxes[indices].copy()

        # for i in range(len(boxes[indices])):
        #     if self.dh > 0.0:
        #         s = self.img_height / self.img_width
        #         out_boxes[i][0] = boxes[indices][i][0] / self.scale_ratio
        #         out_boxes[i][2] = boxes[indices][i][2] / self.scale_ratio
        #         out_boxes[i][1] = (boxes[indices][i][1] - self.dh) /self.scale_ratio
        #         out_boxes[i][3] = (boxes[indices][i][3] - self.dh) /self.scale_ratio

        #     elif self.dw > 0.0:
        #         out_boxes[i][0] = (boxes[indices][i][0] - self.dw )/ self.scale_ratio
        #         out_boxes[i][2] = (boxes[indices][i][2] - self.dw) / self.scale_ratio
        #         out_boxes[i][1] = boxes[indices][i][1] / self.scale_ratio
        #         out_boxes[i][3] = boxes[indices][i][3] / self.scale_ratio

        # return out_boxes, scores[indices], class_ids[indices]
        
        return out_boxes, out_scores, out_class
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes