export PYTHONPATH="$PWD/yolov5:$PYTHONPATH"
python yolov5/train.py \
    --data data/machine.yaml \
    --cfg yolov5m.yaml --weights 'checkpoints/yolov5m.pt' \
	--img 640 \
	--batch-size 16 \
	--device 0 \
	--epochs 100 \
	--exist-ok \
	--name "machine"
