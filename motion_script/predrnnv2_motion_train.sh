export CUDA_VISIBLE_DEVICES=1
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name motion \
    --train_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64_new/Map_array/normalized_train_video.npz \
    --valid_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64_new/Map_array/normalized_test_seen_video.npz \
    --save_dir checkpoints/motion_predrnn_v2 \
    --gen_frm_dir results/motion_predrnn_v2 \
    --model_name predrnn_memory_decoupling \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 17 \
    --total_length 34 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 128 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 500 \
    --snapshot_interval 500 \
    # --pretrained_model ./models/predrnnv2/model.ckpt-500



# export CUDA_VISIBLE_DEVICES=1
# cd ..
# python -u run.py \
#     --is_training 1 \
#     --device cuda \
#     --dataset_name motion \
#     --train_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64/Map_array/normalized_train_video.npz \
#     --valid_data_paths /home/xiao/Projects/ST_LSTM_MP/Data/2D_map_64/Map_array/normalized_test_seen_video.npz \
#     --save_dir checkpoints/motion_predrnn_v2 \
#     --gen_frm_dir results/motion_predrnn_v2 \
#     --model_name predrnn_memory_decoupling \
#     --reverse_input 1 \
#     --img_width 64 \
#     --img_channel 1 \
#     --input_length 5 \
#     --total_length 14 \
#     --num_hidden 128,128,128,128 \
#     --filter_size 5 \
#     --stride 1 \
#     --patch_size 4 \
#     --layer_norm 0 \
#     --decouple_beta 0.1 \
#     --reverse_scheduled_sampling 1 \
#     --r_sampling_step_1 25000 \
#     --r_sampling_step_2 50000 \
#     --r_exp_alpha 2500 \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --max_iterations 80000 \
#     --display_interval 100 \
#     --test_interval 5000 \
#     --snapshot_interval 500 \
#    # --pretrained_model ./checkpoints/mnist_predrnn_v2/mnist_model.ckpt