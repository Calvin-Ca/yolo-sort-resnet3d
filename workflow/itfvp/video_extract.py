import sys
from pathlib import Path
import torch
from video_io import BlankFrame, InputStreamer, OutputStreamer
from detector import YoloDetector
from util.draw import draw_boxes
from highlight import ResNet18FeatExtractor, cosin_distance
import logging
import signal
import argparse

ROOT = Path(__file__).parents[1]
ROOT not in sys.path and sys.path.append(ROOT)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/caic/code/sheng/itf-vp/logs/highlight.log"),
        logging.StreamHandler()
    ]
)

# 检测视频中的特定对象，并根据检测结果和特征匹配来决定输出原始帧还是空白帧,输出根据检测逻辑处理后的视频帧
def video_extract(
       video_source,
       target_path,
       device,
       detect_model_path,
       add_blank=False,
       draw_bbox=False,
       feat_dist_thres=0,
       max_age=30,
       min_hits=3,
       time_since_update=0,  # step since last update
       time_activate=0,
):
   """
   video_source: str, video path or webcam
   target_path: str, target path
   device: int, device id
   add_blank: bool, add blank frame if no detection
   draw_bbox: bool, draw bbox if detection
   feat_dist_thres: float, feature distance threshold
   max_age: int, max age if no detection
   min_hits: int, min hits if detection
   """
   device = torch.device(f"cuda:{device}")
   input_streamer = None
   output_streamer = None

   try:
       input_streamer = InputStreamer(video_source)

       width = input_streamer.width
       height = input_streamer.height
       fps = input_streamer.fps

       output_streamer = OutputStreamer(target_path, width, height, fps)

       def signal_handler(sig, frame):
           if input_streamer:
               input_streamer.terminate()
           if output_streamer:
               output_streamer.terminate()
           logging.info("Received SIGINT or SIGTERM. Terminating gracefully...")

       signal.signal(signal.SIGINT, signal_handler)
       signal.signal(signal.SIGTERM, signal_handler)

       time_since_update = 0  # step since last update
       time_activate = 0
#        detect_model_path = FILE.parent / "itfvp/resource/det/det_vh.onnx"
       detector = YoloDetector(detect_model_path, device=device)
       blank_frame = BlankFrame()
       last_feat = None
       feat_extractor = ResNet18FeatExtractor(device)

       def active(is_active):
           nonlocal time_since_update, time_activate
           if is_active:
               time_activate += 1
               time_since_update = 0
           else:
               time_since_update += 1
               time_activate = 0

       frame_index = -1
       for frame0 in input_streamer:
           frame_index += 1
           # det = detector.detect([frame0], conf_thres=0.5, imgsz=(640, 640))
           det = detector.detect([frame0], conf_thres=0.5)
           det = det[0]

           if len(det) > 0:
               if feat_dist_thres > 0:
                   feat = feat_extractor.feature(frame0)
                   if last_feat is not None:
                       dist = cosin_distance(feat, last_feat)
                       logging.debug(f"cosin distance: {dist}")
                       if dist > feat_dist_thres:
                           active(True)
                       else:
                           active(False)
                   else:
                       active(True)
                   last_feat = feat
               else:
                   active(True)
               if draw_bbox:
                   frame0 = draw_boxes(frame0, det[:, :4])
           else:
               active(False)
               last_feat = None

           if time_since_update > max_age:
               # no detection for a long time
               if add_blank:
                   output_streamer.write(blank_frame)
           else:
               if time_since_update > 0:
                   # no detection for a while
                   output_streamer.write(frame0)
               else:  # time_since_update == 0
                   # detection
                   if time_activate > min_hits or frame_index < min_hits:
                       output_streamer.write(frame0)
                   else:
                       if add_blank:
                           output_streamer.write(blank_frame)

       output_streamer.write(None)

   finally:
       # 确保流被正确关闭
       if input_streamer:
           try:
               input_streamer.terminate()
               logging.info("输入流已终止")
           except Exception as e:
               logging.error(f"终止输入流失败: {e}")

       if output_streamer:
           try:
               output_streamer.terminate()
               logging.info("输出流已终止")
           except Exception as e:
               logging.error(f"终止输出流失败: {e}")


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", type=str, default=None, help="webcam or video path")
    arg_parser.add_argument("--target", type=str, default=None, help="target path")
    arg_parser.add_argument("--device", type=int, default=0, help="device id")
    arg_parser.add_argument("--add_blank", action="store_true", help="add blank frame")
    arg_parser.add_argument("--draw_bbox", action="store_true", help="draw boxes")
    arg_parser.add_argument("--feat_dist_thres", type=float, default=0.0, help="feature distance threshold")
    args = arg_parser.parse_args()

    detect_model_path= ROOT / "resource/machine.onnx"

    video_extract(
        args.source,
        args.target,
        args.device,  
        detect_model_path,
        add_blank=args.add_blank,
        draw_bbox=args.draw_bbox,
        feat_dist_thres=args.feat_dist_thres
        )







