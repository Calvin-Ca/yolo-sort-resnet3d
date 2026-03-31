data_path="/home/p/data/itf2023/datasets/activity/dataset_demo"
#sleep 20000
ts=`date +%Y%m%d%H`
python main.py \
    --video_path $data_path/jpg \
    --annotation_path $data_path/annotation.json \
    --result_path results/warmup_ls/$ts --dataset self --model resnet \
    --model_depth 50 --n_classes 9 --batch_size 8 --n_threads 4 --checkpoint 50 \
    --n_epochs 200 --lr_scheduler "warmup" --warmup_epochs 20 \
    --sample_duration 30 --sample_size 240 --learning_rate 0.005 \
    --n_val_samples 1 \
    --label_smoothing 0.1 --tensorboard
    #--loss_weights "5.13,4.21,6.17,10.0,1.00,1.15,10.0,2.06,1.18" \
