import torch
import cv2 
import numpy as np
import onnxruntime 
import torchvision
import math
import time

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
    
def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=(), device='cuda:0'):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 16), device=torch.device(device))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        x = torch.from_numpy(x)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=torch.device(device))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = torch.from_numpy(xywh2xyxy(x[:, :4]))
        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 15:].max(1, keepdims=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=torch.device(device))).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

def draw_box(image, box):
    image = cv2.imread(image).astype(np.float32)

    real_h, real_w = image.shape[:2]
    x, y = int(box[0]), int(box[1])
    h, w = int(box[3]) - int(box[1]), int(box[2]) - int(box[0])
    a = x + w // 2
    b = y + h // 2

    box[0] = max(0, a - w // 3)
    box[1] = max(0, b - h // 3)
    w = w // 1.5
    h = h // 1.5

    end_h = min(box[1] + h, real_h)
    end_w = min(box[0] + w, real_w)
    color = (0, 255, 0)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(end_w), int(end_h)), color, 2)
        
    return image

def detect_warm_up():
    print("Starting Face Detector")
    test_frame = np.random.randn(640, 640, 3).astype(np.float32)

    for i in range(10):
        list_boxs = draw_box(test_frame)
    print("Face Detector Ready")

def rescale_boxes(boxes, original_width, original_height):
    scale_width = original_width / 640
    scale_height = original_height / 640
    rescaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min_rescaled = x_min * scale_width
        y_min_rescaled = y_min * scale_height
        x_max_rescaled = x_max * scale_width
        y_max_rescaled = y_max * scale_height
        rescaled_boxes.append([x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled])
    
    return rescaled_boxes

def crop_square(image, list_box):
    face_list = []
    for box in list_box:
        real_h, real_w = image.shape[:2]
        x, y = int(box[0]), int(box[1])
        h, w = int(box[3]) - int(box[1]), int(box[2]) - int(box[0])
        a = x + w // 2
        b = y + h // 2

        x = max(0, a - w // 3)
        y = max(0, b - h // 3)
        w = w // 1.5
        h = h // 1.5

        end_h = min(y + h, real_h)
        end_w = min(x + w, real_w)
        face = image[int(y):int(end_h), int(x):int(end_w), :]
        
        face_list.append(face)
        
    return face_list

def detect_face(frame, detect_model):
    out_face = detect_model([frame])
    return out_face[0][0], [out_face[3][0]]

def get_faces(image, detect_model):
    original_width, original_height = image.shape[1], image.shape[0]
    image_resized = cv2.resize(image,(640, 640))
    boxs, landmarks = detect_face(image_resized, detect_model)

    if boxs is None or landmarks is None or len(boxs) == 0:
        return [], []

    rescaled_boxes = rescale_boxes(boxs, original_width, original_height)
    faces = crop_square(image, rescaled_boxes)

    return faces, rescaled_boxes