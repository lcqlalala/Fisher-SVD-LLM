

# Base configuration
MODEL_ID="/data1/common/llm-models/llama-7b"
DATASET="wikitext2"

CUDA_VISIBLE_DEVICES=3 python SVDLLM.py \
    --step 14 \
    --model_path /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4_3611_phase123.pt  \


# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_2/MODEL_ID_whitening_then_update_0.8.pt...
# 100%|█████████████████████████████████████████████████████████████| 167/167 [16:01<00:00,  5.76s/it]
# PPL after pruning: {'wikitext2': 8.311875854526237}
# Weight Memory: 22008.802734375 MiB

# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_1/MODEL_ID_whitening_only_0.8.pt...
# 100%|█████████████████████████████████████████████████████████████| 167/167 [16:00<00:00,  5.75s/it]
# PPL after pruning: {'wikitext2': 7.886789331088916}
# Weight Memory: 22008.802734375 MiB


# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_2_06/MODEL_ID_whitening_then_update_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:46<00:00,  3.51s/it]
# PPL after pruning: {'wikitext2': 56.884194875713}
# Weight Memory: 11958.490234375 MiB

# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_1_06/MODEL_ID_whitening_only_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:45<00:00,  3.51s/it]
# PPL after pruning: {'wikitext2': 72.59055825253797}
# Weight Memory: 11958.490234375 MiB




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:32<00:00,  3.43s/it]
# PPL after pruning: {'wikitext2': 61.41726840786484}
# Weight Memory: 11958.490234375 MiB

