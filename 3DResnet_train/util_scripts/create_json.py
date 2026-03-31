import argparse
import json
from pathlib import Path
import random

from util_scripts.utils import get_n_frames

def create_train_json(root_dir_path: Path, dst_json_path: Path):
    dst_data = {}
    dst_data["labels"] = []
    dst_data['database'] = {}
    for class_dir_path in sorted(root_dir_path.iterdir()):
        label = class_dir_path.name
        print(class_dir_path.name)
        dst_data["labels"].append(label)
        for video_file_path in sorted(class_dir_path.iterdir()):
            print(label, video_file_path.name)
            key = video_file_path.name
            item = {}
            item["subset"] = "training" if random.random() < 0.8 else "validation"
            n_frames = get_n_frames(video_file_path)
            item["annotations"] = {}
            item["annotations"]["label"] = label
            item['annotations']['segment'] = (1, n_frames + 1)
            dst_data["database"][key] = item
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)

def create_inference_json(root_dir_path: Path, dst_json_path: Path):
    dst_data = {}
    dst_data["labels"] = ["clean", "concrete", "formwork", "prepare", "rebar", "rest_talk", "scaffold", "transport", "walk"]
    dst_data['database'] = {}
    for video_file_path in sorted(root_dir_path.iterdir()):
        key = video_file_path.name
        item = {}
        item["subset"] = "predict"
        n_frames = get_n_frames(video_file_path)
        item["annotations"] = {}
        item['annotations']['segment'] = (1, n_frames + 1)
        dst_data["database"][key] = item
    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help='Path of root video directory (jpg).')
    parser.add_argument('dst_path',
                        default=None,
                        type=Path,
                        help='Path of dst json file.')
    
    parser.add_argument('--nolabel', default=False, action='store_true')

    args = parser.parse_args()

    if not args.nolabel:
        create_train_json(args.video_path, args.dst_path)
    else:
        create_inference_json(args.video_path, args.dst_path)
