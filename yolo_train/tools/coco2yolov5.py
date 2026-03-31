
#covert coco dataset to yolov5 format

import json
import os
import shutil
import argparse
import pycocotools.coco as coco

def parse_coco_json(fp, base_dir="."):
    dataset = coco.COCO(fp) # load annotations
    cat_ids = dataset.getCatIds()
    cat_remap = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    images = dataset.loadImgs(dataset.getImgIds()) # load images
    for image in images:
        #get annotations by image id
        print(image)
        annIds = dataset.getAnnIds(imgIds=image['id'], iscrowd=None)
        anns = dataset.loadAnns(annIds)
        yolo_label = ""
        width = image['width']
        height = image['height']
        for ann in anns:
            #get bbox
            bbox = ann['bbox']
            #transform bbox to yolo format
            x_min, y_min, w, h = bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height
            x_center, y_center = x_min + w/2, y_min + h/2
            bbox = [x_center, y_center, w, h]
            #get category id
            cat_id = ann['category_id']
            cat_id = cat_remap[cat_id]
            #convert to yolo format
            yolo_label += str(cat_id) + " " + " ".join([str(i) for i in bbox]) + "\n"
        #save yolo label
        image_name = image['file_name']
        image_basename = os.path.basename(image_name)
        image_basename = os.path.splitext(image_basename)[0]
        image_path = os.path.join(base_dir, image_name)
        label_path = base_dir + "/labels/" + image_basename + ".txt"
        with open(label_path, "w") as f:
            f.write(yolo_label)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--coco_json", type=str, required=True, help="coco json file path")
    arg_parser.add_argument("--base_dir", type=str, required=True, help="base dir")


    args = arg_parser.parse_args()
    coco_json = args.coco_json
    parse_coco_json(coco_json, args.base_dir)


    