# r = (m×n×ratio) / (m+n) 0.4(819 1194) 0.6()

# Base configuration
# 1113
MODEL_ID="/data1/common/llm-models/llama-7b"
DATASET="wikitext2"


# （800, 2.0, 128）max_alloc = min(original_rank, max(min_alloc, int(uniform_rank * 2.0))) min_alloc = max(min_rank, int(uniform_rank * 0.3))
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:24<00:00,  3.38s/it]
# PPL after pruning: {'wikitext2': 53.29182121354696}
# Weight Memory: 12005.6142578125 MiB / 29895

# 最好
# （800, 2.0, 128）max_alloc = min(original_rank, max(min_alloc, int(uniform_rank * 1.5))) min_alloc = max(min_rank, int(uniform_rank * 0.3))
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:21<00:00,  3.36s/it]
# PPL after pruning: {'wikitext2': 43.09177434902402}
# Weight Memory: 12014.740234375 MiB

# （819, 2.0, 128）max_alloc = min(original_rank, max(min_alloc, int(uniform_rank * 1.1))) min_alloc = max(min_rank, int(uniform_rank * 0.3))
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:29<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 52.63984381683912}
# Weight Memory: 12022.0615234375 MiB

# 自适应
#（800, 2.0, 128） max_factor = 1.35 + 0.45 * concentration  # Range: [1.1, 2.0]
# max_alloc = min(original_rank, max(min_alloc, int(uniform_rank * max_factor))) min_alloc = max(min_rank, int(uniform_rank * f_min_optimal))
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:28<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 43.1216508693093}
# Weight Memory: 12012.1357421875 MiB

# 增加Phase 4 （最好2）
# （16, 2.0, 128） 其它和自适应保持一致
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:30<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 39.80870502232289}
# Weight Memory: 12056.3271484375 MiB



# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:29<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 44.96454757884785}
# Weight Memory: 12056.3271484375 MiB

# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:28<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 42.71228401966957}
# Weight Memory: 12056.3271484375 MiB

# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|████████████████████████████████████████████████████████████████████| 167/167 [09:30<00:00,  3.42s/it]
# PPL after pruning: {'wikitext2': 40.4181166436927}
# Weight Memory: 12056.3271484375 MiB

# 1-way 分组 token_sample_ratio=0.2
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|████████████████████████████████████████████████████████████████████| 167/167 [09:29<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 37.5644491446029}
# Weight Memory: 12056.3271484375 MiB

# 4-way 分组 token_sample_ratio=0.2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:28<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 42.16043651045356}
# Weight Memory: 12056.3271484375 MiB

# 2-way 分组 token_sample_ratio=0.2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:29<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 39.95459417758434}
# Weight Memory: 12056.3271484375 MiB

# 1-way 分组 token_sample_ratio=0.3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:29<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 36.72666245921762}
# Weight Memory: 12056.3271484375 MiB

# 1-way 分组 token_sample_ratio=0.6
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:30<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 36.201346264037944}
# Weight Memory: 12056.3271484375 MiB

# Phase 4: ALS Calibration (2 iterations, update_sigma=True, token_sample=80%)...
#   Sampling 1638 tokens per sequence (total ~209664 tokens)
#   0%|          | 0/32 [01:17<?, ?it/s]
# Traceback (most recent call last):
#   File "/data1/lichangqun/SVD-LLM/fisher_svd.py", line 1486, in compress
#     self.phase4_als_calibration(calib_loader, num_iters=als_iters,
#   File "/data1/lichangqun/SVD-LLM/fisher_svd.py", line 1899, in phase4_als_calibration
#     imp1, cnt1 = calibrate_projections(attn_mlp_first)
#   File "/data1/lichangqun/SVD-LLM/fisher_svd.py", line 1818, in calibrate_projections
#     V = torch.linalg.lstsq(X, Z_target).solution  # (in_dim, r)
# RuntimeError: CUDA error: an illegal memory access was encountered
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# 相同的代码 不用Phase 3b
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:28<00:00,  3.40s/it]
# PPL after pruning: {'wikitext2': 36.28004349706273}
# Weight Memory: 12056.3271484375 MiB

# 相同的代码 用Phase 3b block_share==0.02
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [14:57<00:00,  5.38s/it]
# PPL after pruning: {'wikitext2': 37.47676431780071}
# Weight Memory: 12082.20361328125 MiB

# use_omp_selection
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [15:08<00:00,  5.44s/it]
# PPL after pruning: {'wikitext2': 38.399304251265036}
# Weight Memory: 12083.462890625 MiB

# block_share=0.02; block_size=16
# PPL after pruning: {'wikitext2': 44.723835483594236}
# Weight Memory: 12062.22705078125 MiB

# block_share=0.1; block_size=64
# PPL after pruning: {'wikitext2': 52.27421104721992}
# Weight Memory: 12073.0126953125 MiB

# block_share=0.1; block_size=32
# PPL after pruning: {'wikitext2': 53.367283278234645}
# Weight Memory: 12086.0205078125 MiB

# block_share=0.05; block_size=16；refine_blocks=True
# PPL after pruning: {'wikitext2': 40.7216515314179}
# Weight Memory: 12126.52197265625 MiB

# block_share=0.1; block_size=16；refine_blocks=True; omp_top_k_per_iter=128; joint_optimize_iters=2
# PPL after pruning: {'wikitext2': 53.71022953343348}
# Weight Memory: 12171.62841796875 MiB

# block_share=0.02; block_size=16；refine_blocks=True; omp_top_k_per_iter=128; joint_optimize_iters=2, use_fisher_weight_als
# PPL after pruning: {'wikitext2': 38.44030438568766}
# Weight Memory: 12086.056640625 MiB

    # --use_fisher_weight_als \
    # --use_residual_blocks \
    # --use_omp_selection \
# CUDA_VISIBLE_DEVICES=1,3 python SVDLLM.py \
#     --model MODEL_ID \
#     --num_gpus 2 \
#     --step 10  \
#     --ratio 0.6 \
#     --use_als \
#     --use_fisher_weight_als \
#     --use_residual_blocks \
#     --block_share 0.05 \
#     --use_omp_selection \
#     --token_sample_ratio 0.6 \
#     --whitening_nsamples 128 \
#     --dataset wikitext2 \
#     --seed 3 \
#     --model_seq_len 2048 \
#     --save_path ./svd_llm_output_step_10_06_test 2>&1 | tee SVDLLM6.log


CUDA_VISIBLE_DEVICES=2,3 python SVDLLM.py \
    --model MODEL_ID \
    --num_gpus 2 \
    --step 10  \
    --ratio 0.6 \
    --whitening_nsamples 128 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/New 2>&1 | tee SVDLLM_new.log

# Phase1 Phase2 Phase3
# PPL after pruning: {'wikitext2': 44.66423710732979}
# Weight Memory: 12056.3271484375 MiB

#  Phase1 Phase2 Phase3 Phase4
# PPL after pruning: {'wikitext2': 36.113132141924744}
# Weight Memory: 12056.3271484375 MiB

# SVD-LLM:
#   profle_svdllm_low_resource() → whitening() → truncate

# Fisher-Aware SVD (新算法):
#   profle_svdllm_low_resource() → Phase1(SVD) → Phase2(Fisher) → Phase3(截断)
#         ↑                            ↑
#       复用SVD-LLM               复用白化变换逻辑

# Phase 1-3: SVD分解 → Fisher估计 → 全局截断
#            ↓
#     svd_components 存储在 CPU 上
#            ↓
# Phase 4: 校准
#     1. 捕获第一层的输入激活
#     2. 对每一层:
#        a. 使用 forward hook 捕获线性层输入 X
#        b. 计算原始输出: Y = X @ W^T
#        c. 从 CPU 加载 SVD 组件 (U, S, VT) → 移到 GPU
#        d. 优化: minimize ||X @ (U@V)^T - Y||²
#        e. 更新 svd_components (存回 CPU)

# Phase 1: 白化 SVD → 奇异值反映输入分布重要性
# Phase 2: Fisher 估计 → 额外考虑输出端敏感度  
# Phase 3: 全局截断 → 按 σᵢ² × Fᵢᵢ 选择保留哪些方向
# Phase 4: 子空间优化 → 在已选方向内找最优系数 ✓ (新实现)


            #         所有Layer的所有Projection
            #                   ↓
            # ┌─────────────────────────────────────┐
            # │  计算每个singular value的Score      │
            # │  Score = σ² × Fisher               │
            # └─────────────────────────────────────┘
            #                   ↓
            # ┌─────────────────────────────────────┐
            # │  贪心分配算法（Marginal Utility）    │
            # │  Priority = Score[k] / (m + n)     │
            # │  每次选择性价比最高的singular value │
            # └─────────────────────────────────────┘
            #                   ↓
            # ┌─────────────────────────────────────┐
            # │  每个Projection保留top-k个          │
            # │  k由全局竞争决定                    │
            # └─────────────────────────────────────┘

# 做出的截断决策既考虑了该分量承载的信息量（σ），又考虑了该分量对最终任务的重要性（F）

# max_rank_ratio: 效果
# 1.0: 完全uniform分配（Fisher只影响层间，不影响层内）
# 1.3: 轻微不均匀，内存稳定
# 1.5: 中等不均匀（默认）
# 2.0: 较大不均匀，可能导致内存增加

# top_k_per_iter	速度	精度	推荐场景
# 1	             慢	   最优	      小模型/最终评测
# 8-32	         中	  接近最优	   大模型/常规使用
# 64+	             快	  可能次优	   快速实验

# Computing whitening matrices...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [10:42<00:00, 20.08s/it]
# Using cross-entropy loss mode with 2 GPU(s).
# Fisher-Aware SVD Compression
#   Mode: Cross-Entropy Loss (full)
#   GPUs: 2
#   Fisher λ: 2.0 (log-space formula)
#   Min rank: 800
#   Using 2 GPUs: ['cuda:0', 'cuda:1']
# Phase 1: SVD Decomposition...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [14:33<00:00, 27.28s/it]
#   Decomposed 32 layers
# Phase 2: Sensitivity Estimation via Empirical Fisher...
#   Using end-to-end task loss (cross-entropy) for Fisher estimation...
#   Computing per-sample gradients for accurate Fisher information...
#   Distributing model across 2 GPUs...
#   Model distributed: 16-17 layers per GPU
#   Added 34 device transfer hooks
#   Gradient checkpointing enabled
#   0%|                                                                                                                                                                                                   | 0/128 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
# /data1/lichangqun/miniconda3/envs/dobisvd/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
#   warnings.warn(
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [16:23<00:00,  7.68s/it]
#   Average calibration loss: 1.7937
#   Estimated Fisher information using 128 samples (per-sample gradients)
# Phase 3: Global Truncation (target ratio: 40.00%, min_rank: 800, λ=2.0)...
#   Importance scoring: LOG-SPACE formula
#   Formula: Score = log(σ) + 2.0 × log(F)
#   Projections: 224 with Fisher, 0 fallback to magnitude
#   Fisher stats: min=1.00e-10, max=4.25e-08, mean=1.00e-10
#   Sigma stats: min=0.0000, max=71638.3125, mean=182.6895
#   Log-space ranges: log(σ) range=22.5, log(F) range=5.4
#   Effective Fisher influence: 2.0 × 5.4 = 10.7
#   Top-20% overlap (old vs new formula): 53.9%
#   Top-40% overlap (old vs new formula): 65.7%
#   Top-60% overlap (old vs new formula): 77.2%
#   Using NO normalization (raw S² × F × layer_factor)
#   Layer factors: L0=0.50, L16=1.00, L31=1.50
#   Fisher vs Magnitude ranking overlap: 14.5% (100% = identical, 0% = completely different)
#   Total original params: 6,476,005,376
#   Target params (ratio=40%): 2,590,402,150
#   After min allocation (30% of uniform): 1,998,848,000 params (30.9%)
#   Greedy allocations: 39165
#   Final params: 2,590,396,160 (40.00%)
#   Total singular values kept: 218365
#   Allocation: 23 layers <90%, 1 ~100%, 8 >110% of uniform
#   Layer 9: 5600/6858 (82% of uniform)
#   Layer 7: 10364/6858 (151% of uniform)
#   Selection pattern: 100.0% contiguous (top-k), 0.0% non-contiguous
#     (100% contiguous = Fisher not helping, just keeping top singular values)
#   Truncation examples:
#     Layer 0 self_attn.q_proj: 4096 -> 800
#     Layer 0 self_attn.k_proj: 4096 -> 800
#     Layer 0 self_attn.v_proj: 4096 -> 800
#   Rank statistics: min=800, max=2388, avg=974.8
#   Actual compression ratio: 40.00%
#   Kept 218365 singular values out of 917504




# Computing whitening matrices...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [08:16<00:00, 15.52s/it]
# Using cross-entropy loss mode with 2 GPU(s).
# Fisher-Aware SVD Compression
#   Mode: Cross-Entropy Loss (full)
#   GPUs: 2
#   Fisher λ: 2.0 (log-space formula)
#   Min rank: 800
#   Using 2 GPUs: ['cuda:0', 'cuda:1']
# Phase 1: SVD Decomposition...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [14:59<00:00, 28.10s/it]
#   Decomposed 32 layers
# Phase 2: Sensitivity Estimation via Empirical Fisher...
#   Using end-to-end task loss (cross-entropy) for Fisher estimation...
#   Computing per-sample gradients for accurate Fisher information...
#   Distributing model across 2 GPUs...
#   Model distributed: 16-17 layers per GPU
#   Added 34 device transfer hooks
#   Gradient checkpointing enabled
#   0%|                                                                                                                                                                                                   | 0/128 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
# /data1/lichangqun/miniconda3/envs/dobisvd/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
#   warnings.warn(
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [16:23<00:00,  7.68s/it]
#   Average calibration loss: 1.7937
#   Estimated Fisher information using 128 samples (per-sample gradients)
# Phase 3: Global Truncation (target ratio: 40.00%, min_rank: 800, λ=2.0)...
#   Importance scoring: LOG-SPACE formula
#   Formula: Score = log(σ) + 2.0 × log(F)
#   Projections: 224 with Fisher, 0 fallback to magnitude
#   Fisher stats: min=1.00e-10, max=4.25e-08, mean=1.00e-10
#   Sigma stats: min=0.0000, max=71638.3125, mean=182.6895
#   Log-space ranges: log(σ) range=22.5, log(F) range=5.4
#   Effective Fisher influence: 2.0 × 5.4 = 10.7
#   Top-20% overlap (old vs new formula): 53.9%
#   Top-40% overlap (old vs new formula): 65.7%
#   Top-60% overlap (old vs new formula): 77.2%
#   Using NO normalization (raw S² × F × layer_factor)
#   Layer factors: L0=0.50, L16=1.00, L31=1.50
#   Fisher vs Magnitude ranking overlap: 14.5% (100% = identical, 0% = completely different)
#   Total original params: 6,476,005,376
#   Target params (ratio=40%): 2,590,402,150
#   After min allocation (30% of uniform): 1,998,848,000 params (30.9%)
#   Greedy allocations: 39165
#   Final params: 2,590,396,160 (40.00%)
#   Total singular values kept: 218365
#   Allocation: 19 layers <90%, 0 ~100%, 13 >110% of uniform
#   Layer 14: 5600/6858 (82% of uniform)
#   Layer 12: 8573/6858 (125% of uniform)
#   Selection pattern: 100.0% contiguous (top-k), 0.0% non-contiguous
#     (100% contiguous = Fisher not helping, just keeping top singular values)
#   Truncation examples:
#     Layer 0 self_attn.q_proj: 4096 -> 800
#     Layer 0 self_attn.k_proj: 4096 -> 800
#     Layer 0 self_attn.v_proj: 4096 -> 800
#   Rank statistics: min=800, max=1791, avg=974.8
#   Actual compression ratio: 40.00%
#   Kept 218365 singular values out of 917504
# Applying compression to model...
#   Original params: 6,476,005,376
#   Compressed params: 2,590,396,160
#   Compression ratio: 40.00%
#   0%|                                                                                                                                                                                                    | 0/32 [00:00<?, ?it/s]  Layer 0 self_attn.q_proj: rank 4096 -> 800
#   Layer 0 self_attn.k_proj: rank 4096 -> 800
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:15<00:00,  2.09it/s]
#   Replaced 224 linear layers with SVD factorization



# Computing whitening matrices...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [10:26<00:00, 19.57s/it]
# Using cross-entropy loss mode with 2 GPU(s).
# Fisher-Aware SVD Compression
#   Mode: Cross-Entropy Loss (full)
#   GPUs: 2
#   Fisher λ: 2.0 (log-space formula)
#   Min rank: 819
#   Using 2 GPUs: ['cuda:0', 'cuda:1']
# Phase 1: SVD Decomposition...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [14:45<00:00, 27.67s/it]
#   Decomposed 32 layers
# Phase 2: Sensitivity Estimation via Empirical Fisher...
#   Using end-to-end task loss (cross-entropy) for Fisher estimation...
#   Computing per-sample gradients for accurate Fisher information...
#   Distributing model across 2 GPUs...
#   Model distributed: 16-17 layers per GPU
#   Added 34 device transfer hooks
#   Gradient checkpointing enabled
#   0%|                                                                                                                                                                                                   | 0/128 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
# /data1/lichangqun/miniconda3/envs/dobisvd/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
#   warnings.warn(
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [16:24<00:00,  7.69s/it]
#   Average calibration loss: 1.7937
#   Estimated Fisher information using 128 samples (per-sample gradients)
# Phase 3: Global Truncation (target ratio: 40.00%, min_rank: 819, λ=2.0)...
#   Importance scoring: LOG-SPACE formula
#   Formula: Score = log(σ) + 2.0 × log(F)
#   Projections: 224 with Fisher, 0 fallback to magnitude
#   Fisher stats: min=1.00e-10, max=4.25e-08, mean=1.00e-10
#   Sigma stats: min=0.0000, max=71638.3125, mean=182.6895
#   Log-space ranges: log(σ) range=22.5, log(F) range=5.4
#   Effective Fisher influence: 2.0 × 5.4 = 10.7
#   Top-20% overlap (old vs new formula): 53.9%
#   Top-40% overlap (old vs new formula): 65.7%
#   Top-60% overlap (old vs new formula): 77.2%
#   Using NO normalization (raw S² × F × layer_factor)
#   Layer factors: L0=0.50, L16=1.00, L31=1.50
#   Fisher vs Magnitude ranking overlap: 14.5% (100% = identical, 0% = completely different)
#   Total original params: 6,476,005,376
#   Target params (ratio=40%): 2,590,402,150
#   After min allocation (30% of uniform): 2,046,320,640 params (31.6%)
#   Greedy allocations: 36822
#   Final params: 2,590,397,952 (40.00%)
#   Total singular values kept: 220278
#   Allocation: 8 layers <90%, 24 ~100%, 0 >110% of uniform
#   Layer 24: 5733/6858 (84% of uniform)
#   Layer 3: 7539/6858 (110% of uniform)
#   Selection pattern: 100.0% contiguous (top-k), 0.0% non-contiguous
#     (100% contiguous = Fisher not helping, just keeping top singular values)
#   Truncation examples:
#     Layer 0 self_attn.q_proj: 4096 -> 900
#     Layer 0 self_attn.k_proj: 4096 -> 900
#     Layer 0 self_attn.v_proj: 4096 -> 900
#   Rank statistics: min=819, max=1313, avg=983.4
#   Actual compression ratio: 40.00%
#   Kept 220278 singular values out of 917504
# Applying compression to model...
#   Original params: 6,476,005,376
#   Compressed params: 2,590,397,952
#   Compression ratio: 40.00%
#   0%|                                                                                                                                                                                                    | 0/32 [00:00<?, ?it/s]  Layer 0 self_attn.q_proj: rank 4096 -> 900
#   Layer 0 self_attn.k_proj: rank 4096 -> 900
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:15<00:00,  2.11it/s]
#   Replaced 224 linear layers with SVD factorization





# Computing whitening matrices...
# 100%|██████████| 32/32 [08:16<00:00, 15.50s/it]
# Using cross-entropy loss mode with 2 GPU(s).
# Fisher-Aware SVD Compression
#   Mode: Cross-Entropy Loss (full)
#   GPUs: 2
#   Fisher λ: 2.0 (log-space formula)
#   Min rank: 800 (adaptive f_min and max_factor)
#   Using 2 GPUs: ['cuda:0', 'cuda:1']
# Phase 1: SVD Decomposition...
# 100%|██████████| 32/32 [14:37<00:00, 27.43s/it]
#   Decomposed 32 layers
# Phase 2: Sensitivity Estimation via Empirical Fisher...
#   Using end-to-end task loss (cross-entropy) for Fisher estimation...
#   Computing per-sample gradients for accurate Fisher information...
#   Distributing model across 2 GPUs...
#   Model distributed: 16-17 layers per GPU
#   Added 34 device transfer hooks
#   Gradient checkpointing enabled
#   0%|          | 0/128 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
# /data1/lichangqun/miniconda3/envs/dobisvd/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
#   warnings.warn(
# 100%|██████████| 128/128 [16:22<00:00,  7.67s/it]
#   Average calibration loss: 1.7937
#   Estimated Fisher information using 128 samples (per-sample gradients)
# Phase 3: Global Truncation (target ratio: 40.00%, min_rank: 800, λ=2.0)...
#   Importance scoring: LOG-SPACE formula
#   Formula: Score = log(σ) + 2.0 × log(F)
#   Projections: 224 with Fisher, 0 fallback to magnitude
#   Fisher stats: min=1.00e-10, max=4.25e-08, mean=1.00e-10
#   Sigma stats: min=0.0000, max=71638.3125, mean=182.6895
#   Log-space ranges: log(σ) range=22.5, log(F) range=5.4
#   Effective Fisher influence: 2.0 × 5.4 = 10.7
#   Top-20% overlap (old vs new formula): 53.9%
#   Top-40% overlap (old vs new formula): 65.7%
#   Top-60% overlap (old vs new formula): 77.2%
#   Using NO normalization (raw S² × F × layer_factor)
#   Layer factors: L0=0.50, L16=1.00, L31=1.50
#   Fisher vs Magnitude ranking overlap: 14.5% (100% = identical, 0% = completely different)
#   Total original params: 6,476,005,376
#   Target params (ratio=40%): 2,590,402,150
#   Score concentration: min=0.010, max=0.441, mean=0.077
#   Adaptive f_min: 0.100 (floor_share=0.62)
#   Adaptive max_factor: min=1.35, max=1.55, mean=1.38
#   After min allocation: 1,998,848,000 params (30.9%)
#   Greedy allocations: 39311
#   Final params: 2,590,396,416 (40.00%)
#   Total singular values kept: 218511
#   Allocation: 16 layers <90%, 0 ~100%, 16 >110% of uniform
#   Layer 16: 5600/6858 (82% of uniform)
#   Layer 0: 8423/6858 (123% of uniform)
#   Selection pattern: 100.0% contiguous (top-k), 0.0% non-contiguous
#     (100% contiguous = Fisher not helping, just keeping top singular values)
#   Truncation examples:
#     Layer 0 self_attn.q_proj: 4096 -> 800
#     Layer 0 self_attn.k_proj: 4096 -> 800
#     Layer 0 self_attn.v_proj: 4096 -> 1119
#   Rank statistics: min=800, max=1653, avg=975.5
#   Actual compression ratio: 40.00%
#   Kept 218511 singular values out of 917504
# Applying compression to model...
#   Original params: 6,476,005,376
#   Compressed params: 2,590,396,416
#   Compression ratio: 40.00%
# 100%|██████████| 32/32 [00:14<00:00,  2.17it/s]
#   Layer 0 self_attn.q_proj: rank 4096 -> 800
#   Layer 0 self_attn.k_proj: rank 4096 -> 800
#   Replaced 224 linear layers with SVD factorization
# (dobisvd) lichangqun@guizhou3-gpu-a800-640g:~/SVD-LLM$ bash run_evaluate.sh 
# evaluating /data1/lichangqun/SVD-LLM/svd_llm_output_step_10_06_test/MODEL_ID_fisher_svd_0.4.pt...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [09:28<00:00,  3.41s/it]
# PPL after pruning: {'wikitext2': 43.1216508693093}
# Weight Memory: 12012.1357421875 MiB
