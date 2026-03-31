
cd yolov5
for dir_name in "worker.1280.yolov5n6.pt" "worker.1280.yolov5s6.pt" "worker.640.yolov5n6.pt" "worker.640.yolov5s6.pt" "worker.640.yolov5m6.pt" "worker.640.yolov5l6.pt";do
python export.py --weights runs/train/$dir_name/weights/best.pt --include onnx --dynamic
mv runs/train/$dir_name/weights/best.onnx ../edge_pt/onnx/${dir_name/%pt/onnx}
done
cd ..
for dir_name in "worker.1280.yolov8n.pt" "worker.1280.yolov8s.pt" "worker.640.yolov8n.pt" "worker.640.yolov8s.pt" "worker.640.yolov8m.pt" "worker.640.yolov8l.pt";do
yolo export model=runs/detect/yolov8/train/$dir_name/weights/best.pt format=onnx dynamic opset=12
mv runs/detect/yolov8/train/$dir_name/weights/best.onnx edge_pt/onnx/${dir_name/%pt/onnx}
done

