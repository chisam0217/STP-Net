export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name motion \
    --train_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64_new/Map_array/normalized_train_video.npz \
    --valid_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64_new/Map_array/normalized_test_seen_video.npz \
    --save_dir checkpoints/motion_predrnn \
    --gen_frm_dir results/motion_predrnn \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 17 \
    --total_length 34 \
    --num_hidden 64,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 128 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 500 \
    --snapshot_interval 500 \
    # --pretrained_model ./models/predrnn/model.ckpt-500




# export CUDA_VISIBLE_DEVICES=0
# cd ..
# python -u run.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name motion \
#     --train_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64/Map_array/normalized_train_video.npz \
#     --valid_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64/Map_array/normalized_test_seen_video.npz \
#     --save_dir checkpoints/motion_predrnn \
#     --gen_frm_dir results/motion_predrnn \
#     --model_name predrnn \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 3 \
#     --input_length 5 \
#     --total_length 14 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --scheduled_sampling 1 \
#     --sampling_stop_iter 50000 \
#     --sampling_start_value 1.0 \
#     --sampling_changing_rate 0.00002 \
#     --lr 0.0003 \
#     --batch_size 256 \
#     --max_iterations 80000 \
#     --display_interval 1 \
#     --test_interval 1 \
#     --snapshot_interval 100 
#     # --pretrained_model /home/xiao/Projects/predrnn-pytorch/models/model.ckpt-100
