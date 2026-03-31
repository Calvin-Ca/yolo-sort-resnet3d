img_size_list=(640 1280)
weights_list=("yolov5n" "yolov5s" "yolov5m" "yolov5l" "yolov5x" "yolov5n6" "yolov5s6" "yolov5m6")
for img_size in ${img_size_list[@]};do
for weights in ${weights_list[@]};do
weights="$weights.pt"
python -m torch.distributed.run --nproc_per_node 4 --master_port 23456 train.py \
	--data data/worker.yaml \
	--weights $weights \
	--img $img_size \
	--batch-size 16 \
	--device 0,1,2,3 \
	--epochs 15 \
	--exist-ok \
	--name "worker.${img_size}.${weights}"
done
done
