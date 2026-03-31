export PYTHONPATH="$PWD/yolov5:$PYTHONPATH"
for i in 640,yolov5n6 640,yolov5s6 640,yolov5m6 640,yolov5l6 1280,yolov5n6 1280,yolov5s6;do
IFS=",";set -- $i;img_size=$1;weights=$2
echo $img_size, $weights
weights="$weights.pt"
python yolov5/train.py \
	--data data/worker.yaml \
	--weights checkpoints/$weights \
	--img $img_size \
	--batch-size 16 \
	--device 0 \
	--epochs 100 \
	--exist-ok \
	--name "worker.${img_size}.${weights}"
done
