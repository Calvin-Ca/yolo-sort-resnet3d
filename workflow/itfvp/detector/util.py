import cv2
import numpy as np
import torch
import torchvision

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class YoloPreprocess(torch.nn.Module):
    def __init__(self, target_shape=640, stride=32, auto=True, scaleup=True, pad_color=114):
        super(YoloPreprocess, self).__init__()
        self.stride = stride
        self.new_shape = (target_shape, target_shape)
        self.auto = auto
        self.scaleup = scaleup
        self.resize_sq = transforms.Resize([target_shape,])
        self.resize_rect = transforms.Resize([target_shape-1,], max_size=target_shape)
        self.pad_color = pad_color

    def forward(self, imgs):
        """
        Args:
            imgs (list): list of images, each image is a numpy array, shape (H, W, C), RGB
        Returns:
            imgs (list): list of images, each image is a numpy array, shape (C, H, W), RGB
            ratios (list): list of ratios
            pads (list): list of pads
        """
        #=====================letterbox=====================
        assert len(set(img.shape for img in imgs)) == 1, 'Batched images must be of the same shape'
        shape = imgs.shape[-2:]  # current shape [height, width]
        new_shape = self.new_shape
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if shape[0] == shape[1]:
            imgs = self.resize_sq(imgs)
        else:
            imgs = self.resize_rect(imgs)
        new_unpad = imgs.shape[-2:][::-1] # (w, h)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        imgs = torch.nn.functional.pad(imgs, [left, right, top, bottom], value=self.pad_color)
        #=====================letterbox=====================
        imgs = imgs.float()
        imgs /= 255.0
        return imgs, ratio, (dw, dh)




def xywh2xyxy(bbox):
    """
    bbox: (n, 4) tensor, (x, y) is the center of the box, (w, h) is the width and height of the box
    """
    x, y, w, h = bbox.unbind(1)
    w, h = w / 2, h / 2
    return torch.stack((x - w, y - h, x + w, y + h), dim=1)

def box_iou(box1, box2):
    """
    box1: (n, 4) tensor, (x1, y1, x2, y2)
    box2: (m, 4) tensor, (x1, y1, x2, y2)
    return: (n, m) tensor, iou
    """
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (n, m, 2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (n, m, 2)
    wh = (rb - lt).clamp(min=0)  # (n, m, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (n, m)
    area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))[:, None]  # (n, 1)
    area2 = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]))[None, :]  # (1, m)
    iou = inter / (area1 + area2 - inter + 1e-6)
    return iou




def NMS(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=(), max_det=300, nm=0):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    only for single label
    Args:
        prediction: (n,6) tensor per image [xyxy, conf, cls]
        conf_thres: confidence threshold, between 0.0 and 1.0
        iou_thres: IoU threshold, between 0.0 and 1.0
    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output



def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2



def scale_boxes(img1_shape, boxes, img0_shape):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
