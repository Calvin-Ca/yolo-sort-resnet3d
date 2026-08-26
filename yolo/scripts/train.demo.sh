export PYTHONPATH="$PWD/yolov5:$PYTHONPATH"
#python -m torch.distributed.run --nproc_per_node 4 --master_port 23456 train.py \
python yolov5/train.py \
	--data data/worker_demo.yaml \
    --cfg yolov5m.yaml --weights 'checkpoints/yolov5m.pt' \
	--img 1280 \
	--batch-size 8 \
	--epochs 100 \
	--exist-ok \
	--name "action_demo"

	#--device 0,1,2,3

