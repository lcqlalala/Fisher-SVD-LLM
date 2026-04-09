# --num_epochs 2 # 实验使用的是3，但最好的验证损失出现在2
# --batch_size 2
# --gradient_accumulation 8
# --learning_rate 3e-4
# --num_samples 2048
# --scheduler constant
# --warmup_ratio 0.05
# --max_grad_norm 0.3
# --label_smoothing 0.0
# --min_lr_ratio 0.3
# --block_weight_decay 0.0
# --seq_len 1024
# --use_amp


CUDA_VISIBLE_DEVICES=1 python block_finetune.py \
    --prune_model /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4phase3b005.pt \
    --output_dir /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/block_finetune_output_005_test \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 16 \
    --learning_rate 2e-4 \
    --num_samples 2048 \
    --scheduler constant \
    --label_smoothing 0.0 \
    --min_lr_ratio 0.3 \
    --block_weight_decay 0.0 \
    --seq_len 512 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules o_proj up_proj down_proj \
    --use_amp 2>&1 | tee block_finetune.log \


# data = load_dataset('/data1/lichangqun/Dobi-SVD/data_cache/wikitext2', 'default', split='train')
    # --train_layernorm \

# block_finetune_v4.py
# num_epochs 2 learning_rate 2e-4 num_samples 1024 scheduler cosine min_lr_ratio 0.1 block_weight_decay 0.0 train_layernorm seq_len 512 batch_size 4 gradient_accumulation 4 
# 100%|████████████████████████████████████████████████████████████████████| 167/167 [18:47<00:00,  6.75s/it]
# PPL after pruning: {'wikitext2': 13.363236107143964}
# Weight Memory: 12126.5166015625 MiB

# tmux lcq2: block_finetune_v4.py  与上面进行对比，关键是 0.1 vs 0.05 block_finetune_output_01 cuda:1
# PPL after pruning: {'wikitext2': 14.134608979985696}
# Weight Memory: 12171.6416015625 MiB




# tmux lcq: block_finetune_v5.py 最新版；进一步提高性能； block_finetune_output_005 cuda:3
# PPL after pruning: {'wikitext2': 12.462244674972917}
# Weight Memory: 12126.5166015625 MiB

# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/block_finetune_output_005/model_block_finetuned.pt...
# 100%|███████████████████████████████████████████████████████████| 167/167 [19:04<00:00,  6.85s/it]
# PPL after pruning: {'wikitext2': 11.376391087634467}
# Weight Memory: 12126.5166015625 MiB