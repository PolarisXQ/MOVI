# Check environment variables.
[ -n "${WAN2_1_VACE_1_3B_MODEL_PATH:-}" ] || {
    echo "WAN2_1_VACE_1_3B_MODEL_PATH is not set, please source path_setup.sh first!" >&2
    exit 1
}
[ -n "${DAVIS17_DATASET_PATH:-}" ] || {
    echo "DAVIS17_DATASET_PATH is not set, please source path_setup.sh first!" >&2
    exit 1
}

model_path="${WAN2_1_VACE_1_3B_MODEL_PATH}"

# DAVIS17 dataset path.
dataset_base_path=${DAVIS17_DATASET_PATH}

# Check environment variables.
[ -n "${WAN2_1_VACE_1_3B_MODEL_PATH:-}" ] || {
    echo "WAN2_1_VACE_1_3B_MODEL_PATH is not set, please source path_setup.sh first!" >&2
    exit 1
}
[ -n "${DAVIS17_DATASET_PATH:-}" ] || {
    echo "DAVIS17_DATASET_PATH is not set, please source path_setup.sh first!" >&2
    exit 1
}


accelerate launch examples/wanvideo/model_training/train_davis17_vace.py \
    --dataset_list "DAVIS17" \
    --sampling_strategy "proportional" \
    --split_name "train" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --dataset_repeat 1 \
    --model_paths "[
        \"${model_path}/diffusion_pytorch_model.safetensors\",
        \"${model_path}/models_t5_umt5-xxl-enc-bf16.pth\",
        \"${model_path}/Wan2.1_VAE.pth\",
        \"./models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"
    ]" \
    --tokenizer_path "${model_path}/google/umt5-xxl/" \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --trainable_models "multiview_feature_bank_adapter,multiview_ipadapter" \
    --remove_prefix_in_ckpt "pipe.vace.,pipe.multiview_feature_bank_adapter.,pipe.multiview_ipadapter." \
    --output_path "./models/train/Wan2.1-VACE-1.3B_lora_davis17-multiview" \
    --lora_base_model "vace" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "vace_video,vace_video_mask,multiview_reference_image" \
    --use_gradient_checkpointing_offload \
    --min_instance_ratio 0.00 \
    --min_bbox_ratio 0.00 \
    --use_masked_vace_video \
    --trajectory_type "mask" \
    --sparse_box_interval 5 \
    --save_steps 100 \
    --mask_strategy "bbox_with_traj" \
    --bbox_scale 1.0 \
    --multiview_reference_mode "temporal_concat+feature_bank" \
    --use_multiview_consistency_check \
    --multiview_ipadapter_scale 0.2 \
    --use_depth_head \
    --lambda_latent_depth 0.001 \
    --use_perception_head \
    --lambda_latent_segmentation 0.001 \
    --lambda_temporal_coherence 0.05 \
    --temporal_coherence_method "simple" 