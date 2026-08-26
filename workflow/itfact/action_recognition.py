import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import List, Tuple, Union
import argparse

ROOT = Path(__file__).parents[1]
ROOT not in sys.path and sys.path.append(ROOT) # add ROOT to PATH

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

class ActionRecognition(object):
    def __init__(self, model_path, device):
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.device = device
        self.model.to(device)
    
    def detect(self, inputs: Union[List[np.ndarray], np.ndarray]):
        """
        Args:
            frames: list of np.ndarray, RGB, (0, 255)
        """
        # model input: N, C, T, H, W 
        if isinstance(inputs, list):
            inputs = np.stack(inputs)
        assert inputs.ndim == 4
        inputs = torch.from_numpy(inputs)
        if inputs.shape[-1] == 3: # (T, H, W, C)
            inputs = inputs.permute(3, 0, 1, 2)
        elif inputs.shape[1] == 3: # (T, C, H, W)
            inputs = inputs.permute(1, 0, 2, 3)
        inputs = inputs.float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            res = self.model(inputs)[0]
            act_id = torch.argmax(res, dim=-1).item()
        return {
            "act_id": act_id,
            "logits": res,
        }

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", type=str, required=True)
    arg_parser.add_argument("--device", type=int, default=0)
    args = arg_parser.parse_args()

    video_path = args.source
    device = args.device
    
    model_path  = ROOT / "resource/jit.end2end.pt"
    act_model = ActionRecognition(model_path, device=device)
    
    frames = [ frame for frame in read_video(video_path)][:20]
    det_res = act_model.detect(frames)
    print(det_res)





