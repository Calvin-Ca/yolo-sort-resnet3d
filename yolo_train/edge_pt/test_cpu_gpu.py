import cv2
import numpy as np
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
import torch
from itfvp.detector import YoloV5Detector
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tmp.log"),
        logging.StreamHandler()
    ]
)

class FileInputStreamer(object):
    '''
    File Stream Input
    read from video file
    '''
    def __init__(self, path, keep_rate=False):
        self.path = path
        self._cap = cv2.VideoCapture(self.path)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_read = 0
        self.keep_rate = keep_rate

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_read> 1:
            raise StopIteration
        current_ts = time.time()
        if self.keep_rate:
            if self.frame_read == 0:
                self.start_ts = current_ts
            ts_gap = self.frame_read/self.fps - (current_ts - self.start_ts)
            if ts_gap > 0:
                time.sleep(ts_gap)
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_read += 1
            return frame
        else:
            raise StopIteration

def check_video(
    model_path,
    device,
    imgsz,
    video_source,
    batch_size=1,
    ):
    if device == -1:
        device = 'cpu'
    else:
        device = torch.device(f"cuda:{device}")
    input_streamer = FileInputStreamer(video_source)
    frame_all = [f for f in input_streamer]

    detector = YoloV5Detector(model_path, device=device, calc_time_cost=True, batch_size=batch_size, imgsz=imgsz)
    """
    frames = []
    for frame0 in input_streamer:
        if len(frames) < batch_size:
            frames.append(frame0)
            continue
        det = detector.detect(frames, conf_thres=0.5)
        frames = []
    if len(frames) > 0:
        pass
        #det = detector.detect(frames, conf_thres=0.5)
    """
    #for frames in [frame_all[i:i+batch_size] for i in range(0, len(frame_all), batch_size)]:
    #    if len(frames) < batch_size:
    #        break
    for _ in range(10000):
        frames = [frame_all[0]]
        det = detector.detect(frames, conf_thres=0.5)

    time_cost = detector.time_cost_manager.get()
    return time_cost

import glob
import argparse

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_dir", type=str, help="model dir", required=True)
    arg_parser.add_argument("--video_path", type=str, help="video path", required=True)
    arg_parser.add_argument("--host", type=str, help="host", required=True)
    arg_parser.add_argument("--report_path", type=str, help="markdown report path", default="report.md")
    arg_parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    arg_parser.add_argument("--device", type=int, help="device", default=-1)

    args = arg_parser.parse_args()
    model_dir = args.model_dir
    video_path = args.video_path
    markdown_path = args.report_path
    batch_size = args.batch_size
    device = args.device
    host = args.host

    report = open(markdown_path, "a")
    # scan dir
    case_list = []
    for model_path in glob.glob(model_dir + "/*.onnx"):
        if "640.yolov5s" not in model_path:
        #if "640.yolov8s" not in model_path:
            continue
        for batch_size in [1]:
            case_list.append((model_path, batch_size))
    for case_index, (model_path, batch_size) in enumerate(case_list):
        model_name = model_path.split("/")[-1] # worker.1280.yolov5n6.onnx
        logging.info(f"Testing {model_name} with batch_size {batch_size}, case {case_index}")
        imgsz = int(model_name.split(".")[1]) # 1280
        model_name = model_name.split(".")[2] # yolov5n6

        time_cost = check_video(model_path, device, imgsz, video_path, batch_size=batch_size)
        if case_index == 0:
            report.write("|Device|Model|Imgsz|BatchSize|Total(ms)|")
            for name in time_cost.keys():
                report.write(f"{name}(ms)|")
            report.write("\n")
        total_time_cost = sum(time_cost.values())
        report.write(f"|{host}|{model_name}|{imgsz}|{batch_size}|{total_time_cost*1000/batch_size:.3f}|")
        for name, cost in time_cost.items():
            report.write(f"{cost*1000/batch_size:.3f}|")
        report.write("\n")
        report.flush()
    report.close()


if __name__ == "__main__":
    main()
