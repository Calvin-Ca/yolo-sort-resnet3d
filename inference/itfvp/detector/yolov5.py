import torch
import numpy as np
from pathlib import Path
from .util import letterbox, NMS, scale_boxes, YoloPreprocess
from torchvision.transforms import Compose, ToTensor, Normalize
import logging
import time
from typing import List
import onnxruntime


class AverageTimeCostManager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.time_cost = {}
        self.count = {}

    def update(self, time_cost):
        for k, v in time_cost.items():
            if k not in self.time_cost:
                self.time_cost[k] = 0
                self.count[k] = 0
            self.time_cost[k] += v
            self.count[k] += 1
    
    def get(self):
        res = {}
        for k, v in self.time_cost.items():
            res[k] = v / self.count[k]
        return res

class YoloDetector(object):
    def __init__(self, weights_path, device='cpu', calc_time_cost=False, batch_size=1, imgsz=640):
        self.device = device
        self.cuda = self.device != 'cpu'
        self.half = self.device != 'cpu'
        if weights_path.suffix == ".onnx":
            self._init_onnx(weights_path)
        else:
            raise NotImplementedError
        self._transform = Compose([
            ToTensor(),
        ])
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.calc_time_cost = calc_time_cost
        self.time_cost_manager = AverageTimeCostManager()
        #self.preprocessor = YoloPreprocess(target_shape=imgsz, stride=self.stride, auto=False).to(self.device)
        logging.debug(f"YoloDetector initialized on {self.device}")
        logging.debug(f"Model: {weights_path}")
        self._warmup()


    def _preprocess1(self, imgs: List[np.array], imgsz=640):
        inputs = torch.from_numpy(np.stack(imgs)).to(self.device).transpose(1, 3)
        inputs, _, _ = self.preprocessor(inputs)
        return list(inputs.cpu().numpy())

    def _preprocess(self, imgs, imgsz=640):
        """
        Args:
            imgs (list): list of images, each image is a numpy array, shape (H, W, C), RGB
            imgsz (int): image size

        Returns:
            inputs (list): list of images, each image is a numpy array, shape (C, H, W), RGB
        """
        inputs = []
        for img in imgs:
            img = letterbox(img, new_shape=imgsz, auto=False, stride=self.stride)[0]
            #img = self._transform(img)
            #img = img.numpy()
            img = img.transpose(2, 0, 1) # HWC to CHW
            img = np.ascontiguousarray(img)
            img = img.astype(np.float32)
            img /= 255.0
            inputs.append(img)
        return inputs

    def _init_onnx(self, weights_path):
        
        self.onnx = True
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.cuda else ["CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(weights_path, providers=providers)
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.stride = int(metadata['stride'])
        self.output_names = [x.name for x in self.session.get_outputs()]



    def _from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def _forward(self, im):
        if self.onnx:  # ONNX Runtime
            if isinstance(im, torch.Tensor):
                im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        if isinstance(y, list):
            y = self._from_numpy(y[0]) if len(y) == 1 else [self._from_numpy(a) for a in y]
        else:
            y = self._from_numpy(y)  
        return y
    
    def _warmup(self):
        inputs = self._preprocess([np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)]*self.batch_size, imgsz=self.imgsz)
        self._forward(inputs)

    def detect(self, imgs, conf_thres=0.25, iou_thres=0.45):
        """
        imgs (list): list of images, each image is a numpy array, shape (H, W, C), RGB
        return (list): list of detections, each detection is a numpy array, shape (N, 6), (x1, y1, x2, y2, score, class)
        """
        ts1 = time.time()
        time_cost = {}
        inputs = self._preprocess(imgs, self.imgsz)
        ts2 = time.time()
        ts1, time_cost['preprocess'] = ts2, ts2 - ts1
        pred = self._forward(inputs)
        ts2 = time.time()
        ts1, time_cost['inference'] = ts2, ts2 - ts1
        pred = NMS(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        ts2 = time.time()
        ts1, time_cost['NMS'] = ts2, ts2 - ts1
        # Process predictions
        dets = []
        for i, det in enumerate(pred):  # per image
            det = det.cpu().detach().numpy()
            im0_shape = imgs[i].shape
            im_shape = inputs[i].shape
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im_shape[1:], det[:, :4], im0_shape).round()
            dets.append(det)
        ts2 = time.time()
        ts1, time_cost['postprocess'] = ts2, ts2 - ts1
        if self.calc_time_cost:
            logging.debug((
                f"Preprocess: {time_cost['preprocess']:.3f},"
                f"Inference: {time_cost['inference']:.3f},"
                f"NMS: {time_cost['NMS']:.3f},"
                f"Postprocess: {time_cost['postprocess']:.3f}"
                ))
            self.time_cost_manager.update(time_cost)
        return dets