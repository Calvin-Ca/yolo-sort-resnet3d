img_size_list=(640)
weights_list=("yolov5s" "yolov5m")
export PYTHONPATH="$PWD/yolov5:$PYTHONPATH"
for img_size in ${img_size_list[@]};do
for weights in ${weights_list[@]};do
echo $img_size, $weights
weights="$weights.pt"
python yolov5/train.py \
	--data data/worker_vh.yaml \
	--weights checkpoints/$weights \
	--img $img_size \
	--batch-size 16 \
	--device 0 \
	--epochs 100 \
	--exist-ok \
	--name "worker_vh.${img_size}.${weights}"
done
done
