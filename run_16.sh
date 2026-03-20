#!/bin/bash
set -x

# error OMM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 1 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1.txt


# done
# Throughput (TFLOP/s/GPU) - Max: 267.65, Min: 264.35, Avg: 265.45
# Tokens (tokens/s/GPU) - Max: 771.1, Min: 761.6, Avg: 764.76
# Time per iteration (ms) - Max: 85839.7, Min: 84338.4, Avg: 85400.85
# Avg HIP Memory Usage: 107.62 GB (56.06%)
# Avg ROCm Memory Usage: 110.08 GB (57.34%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2.txt

# done
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 3 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_3.txt


# done
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 4 2>&1 | tee log_16_bf16_pp_16_mbs_2_gbs_2048_turbo_deepep_recompute_4.txt


# done
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_recompute_2_vpp_2.txt


# done
# Throughput (TFLOP/s/GPU) - Max: 240.55, Min: 240.2, Avg: 240.44
# Tokens (tokens/s/GPU) - Max: 693.0999999999999, Min: 691.9000000000001, Avg: 692.65
# Time per iteration (ms) - Max: 94495.9, Min: 94408.2, Avg: 94461.12
# Avg HIP Memory Usage: 81.78 GB (42.60%)
# Avg ROCm Memory Usage: 85.49 GB (44.53%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_vpp_2.txt


# Require a lot of memory, focus on FP8 training 
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 1 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4.txt

# RuntimeError: Training execution failed: Rank 127, node j16u37, device 7, iteration 1: Unexpected result nan (message='found NaN in local forward loss calculation')
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_rope_fusion_sync_moe_1.txt

# RuntimeError: Training execution failed: Rank 127, node j16u37, device 7, iteration 1: Unexpected result nan (message='found NaN in local forward loss calculation')
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--apply_rope_fusion true \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_rope_fusion.txt

# RuntimeError: Training execution failed: Rank 127, node j16u37, device 7, iteration 1: Unexpected result nan (message='found NaN in local forward loss calculation')
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_sync_moe_1.txt


./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 10 \
--pipeline_model_parallel_size 16 \
--micro_batch_size 1 \
--global_batch_size 2048 \
--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
--use_turbo_deepep true \
--moe_shared_expert_overlap false \
--recompute_granularity full \
--recompute_method block \
--apply_rope_fusion true \
--turbo_sync_free_moe_stage 1 \
--recompute_num_layers 2 2>&1 | tee log_16_bf16_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_rope_fusion_sync_moe_1.txt


#--------------------------------------------------------------------------------------------------------------------------------------
# FP8
#--------------------------------------------------------------------------------------------------------------------------------------

# Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 64 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_64_turbo_deepep_1.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 280.6, Min: 273.3, Avg: 276.98
# Tokens (tokens/s/GPU) - Max: 808.45, Min: 787.5, Avg: 798.00
# Time per iteration (ms) - Max: 83220.6, Min: 80267.7, Avg: 81849.75
# Avg HIP Memory Usage: 117.00 GB (60.94%)
# Avg ROCm Memory Usage: 119.44 GB (62.21%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 3 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_3.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 279.95, Min: 277.85, Avg: 278.68
# Tokens (tokens/s/GPU) - Max: 806.6, Min: 800.45, Avg: 802.85
# Time per iteration (ms) - Max: 81591.2, Min: 80838.8, Avg: 81304.00
# Avg HIP Memory Usage: 118.39 GB (61.67%)
# Avg ROCm Memory Usage: 120.83 GB (62.94%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2.txt

# error deepep
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 291.5, Min: 270.8, Avg: 284.24
# Tokens (tokens/s/GPU) - Max: 839.75, Min: 780.3, Avg: 818.88
# Time per iteration (ms) - Max: 83990.1, Min: 76889.0, Avg: 79636.18
# Avg HIP Memory Usage: 89.11 GB (46.42%)
# Avg ROCm Memory Usage: 93.12 GB (48.50%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_vpp_2.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 305.45, Min: 289.8, Avg: 297.81
# Tokens (tokens/s/GPU) - Max: 880.0999999999999, Min: 834.8, Avg: 857.92
# Time per iteration (ms) - Max: 78507.1, Min: 73276.4, Avg: 76048.40
# Avg HIP Memory Usage: 92.88 GB (48.38%)
# Avg ROCm Memory Usage: 97.04 GB (50.55%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_2_vpp_2_precision_aware.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 302.55, Min: 297.55, Avg: 299.18
# Tokens (tokens/s/GPU) - Max: 871.65, Min: 857.25, Avg: 861.90
# Time per iteration (ms) - Max: 76433.4, Min: 74437.1, Avg: 75784.60
# Avg HIP Memory Usage: 92.88 GB (48.38%)
# Avg ROCm Memory Usage: 96.75 GB (50.40%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_2_precision_aware.txt


# OOM / crashed
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 2 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_16_mbs_2_gbs_2048_turbo_deepep_recompute_2_vpp_2_precision_aware.txt


# OOM / crashed
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_vpp_2_precision_aware.txt


# done
# Throughput (TFLOP/s/GPU) - Max: 306.3, Min: 304.3, Avg: 305.60
# Tokens (tokens/s/GPU) - Max: 882.4000000000001, Min: 876.7, Avg: 880.35
# Time per iteration (ms) - Max: 74754.5, Min: 74073.7, Avg: 74299.52
# Avg HIP Memory Usage: 128.20 GB (66.78%)
# Avg ROCm Memory Usage: 131.89 GB (68.70%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware.txt


# OOM / crashed
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 2 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_2_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware.txt

# OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_vpp_4_precision_aware.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 293.6, Min: 282.1, Avg: 289.31
# Tokens (tokens/s/GPU) - Max: 845.95, Min: 812.7, Avg: 833.52
# Time per iteration (ms) - Max: 80643.1, Min: 76633.2, Avg: 78054.52
# Avg HIP Memory Usage: 92.89 GB (48.38%)
# Avg ROCm Memory Usage: 96.92 GB (50.48%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--use_turbo_rms_norm true\
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_2_precision_aware_turbo_rms_norm.txt

# done
# Throughput (TFLOP/s/GPU) - Max: 298.7, Min: 291.95000000000005, Avg: 295.61
# Tokens (tokens/s/GPU) - Max: 860.5999999999999, Min: 841.1, Avg: 851.70
# Time per iteration (ms) - Max: 77169.3, Min: 75345.8, Avg: 76206.23
# Avg HIP Memory Usage: 128.14 GB (66.75%)
# Avg ROCm Memory Usage: 131.83 GB (68.67%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--use_turbo_rms_norm true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_turbo_rms_norm.txt


# Throughput (TFLOP/s/GPU) - Max: 324.45000000000005, Min: 322.6, Avg: 323.70
# Tokens (tokens/s/GPU) - Max: 934.7, Min: 929.4, Avg: 932.64
# Time per iteration (ms) - Max: 70513.3, Min: 69875.9, Avg: 70157.30
# Avg HIP Memory Usage: 128.18 GB (66.77%)
# Avg ROCm Memory Usage: 131.87 GB (68.69%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--enable_experimental true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_apply_rope_fusion_enable_experimental.txt


# Throughput (TFLOP/s/GPU) - Max: 325.15, Min: 322.5, Avg: 324.15
# Tokens (tokens/s/GPU) - Max: 936.75, Min: 929.2, Avg: 933.84
# Time per iteration (ms) - Max: 70526.6, Min: 69666.1, Avg: 70058.57
# Avg HIP Memory Usage: 128.13 GB (66.74%)
# Avg ROCm Memory Usage: 131.82 GB (68.66%)
# ./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
# -N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
# --train_iters 10 \
# --pipeline_model_parallel_size 16 \
# --virtual_pipeline_model_parallel_size 4 \
# --micro_batch_size 1 \
# --global_batch_size 2048 \
# --pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
# --use_turbo_deepep true \
# --moe_shared_expert_overlap false \
# --recompute_granularity full \
# --recompute_method block \
# --turbo_deepep_num_cu 80 \
# --use_precision_aware_optimizer True \
# --main_grads_dtype bf16 \
# --main_params_dtype fp16 \
# --exp_avg_dtype bf16 \
# --exp_avg_sq_dtype bf16 \
# --apply_rope_fusion true \
# --recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion.txt


# Tokens (tokens/s/GPU) - Max: 899.0, Min: 860.3499999999999, Avg: 881.59
# Throughput (TFLOP/s/GPU) - Max: 312.04999999999995, Min: 298.6, Avg: 306.01
# Time per iteration (ms) - Max: 75825.9, Min: 71670.8, Avg: 73468.73
# Avg HIP Memory Usage: 128.18 GB (66.77%)
# Avg ROCm Memory Usage: 131.86 GB (68.68%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--enable_experimental true \
#--use_turbo_rms_norm true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_apply_rope_fusion_turbo_rms_norm.txt


# Throughput (TFLOP/s/GPU) - Max: 181.55, Min: 160.5, Avg: 171.18
# Tokens (tokens/s/GPU) - Max: 726.8, Min: 642.4, Avg: 685.16
# Time per iteration (ms) - Max: 102012.0, Min: 86239.9, Avg: 94301.50
# Avg HIP Memory Usage: 122.45 GB (63.78%)
# Avg ROCm Memory Usage: 126.08 GB (65.67%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--multi_latent_attention true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_mla.txt


# RuntimeError: Training execution failed: Rank 127, node j16u37, device 7, iteration 1: Unexpected result nan (message='found NaN in local forward loss calculation')
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--moe_use_legacy_grouped_gemm = True \
#--use_turbo_grouped_mlp = False \
#--use_turbo_attention = False \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_moe_use_legacy_grouped_gemm.txt


# Throughput (TFLOP/s/GPU) - Max: 170.15, Min: 160.5, Avg: 165.06
# Tokens (tokens/s/GPU) - Max: 681.15, Min: 642.5, Avg: 660.74
# Time per iteration (ms) - Max: 101994.7, Min: 93889.5, Avg: 98332.40
# Avg HIP Memory Usage: 122.75 GB (63.94%)
# Avg ROCm Memory Usage: 126.44 GB (65.86%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--multi_latent_attention true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_mla.txt

# Throughput (TFLOP/s/GPU) - Max: 206.89999999999998, Min: 206.5, Avg: 206.75
# Tokens (tokens/s/GPU) - Max: 828.25, Min: 826.5, Avg: 827.60
# Time per iteration (ms) - Max: 79291.6, Min: 79065.1, Avg: 79139.33
# Avg HIP Memory Usage: 117.10 GB (61.00%)
# Avg ROCm Memory Usage: 119.89 GB (62.45%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--moe_use_legacy_grouped_gemm = True \
#--use_turbo_grouped_mlp = False \
#--use_turbo_attention = False \
#--multi_latent_attention true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_moe_use_legacy_grouped_gemm_mla.txt

# Runtime error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 8 \
#--virtual_pipeline_model_parallel_size 2 \
#--expert_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 32 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--recompute_num_layers 4 2>&1 | tee log_16_fp8_pp_8_ep_16_mbs_1_gbs_2048_turbo_deepep_recompute_4_vpp_2_precision_aware_rope_fusion.txt


# Runtime error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--recompute_granularity selective 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_selective_vpp_4_precision_aware_rope_fusion.txt

# Throughput (TFLOP/s/GPU) - Max: 325.05, Min: 322.6, Avg: 324.23
# Tokens (tokens/s/GPU) - Max: 936.45, Min: 929.3, Avg: 934.05
# Time per iteration (ms) - Max: 70519.0, Min: 69673.1, Avg: 69998.35
# Avg HIP Memory Usage: 127.93 GB (66.64%)
# Avg ROCm Memory Usage: 131.62 GB (68.56%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 64 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_64_cu.txt

# OOM Runtime error
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--recompute_num_layers 4 2>&1 | tee log_16_fp8_pp_16_mbs_2_gbs_2048_turbo_deepep_recompute_4_precision_aware_rope_fusion.txt


# Throughput (TFLOP/s/GPU) - Max: 321.8, Min: 320.9, Avg: 321.36
# Tokens (tokens/s/GPU) - Max: 926.95, Min: 924.45, Avg: 925.80
# Time per iteration (ms) - Max: 70872.2, Min: 70590.1, Avg: 70697.85
# Avg HIP Memory Usage: 121.79 GB (63.44%)
# Avg ROCm Memory Usage: 125.48 GB (65.36%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 2 \
#--moe_use_legacy_grouped_gemm true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_2.txt


# Throughput (TFLOP/s/GPU) - Max: 331.75, Min: 329.0, Avg: 330.44
# Tokens (tokens/s/GPU) - Max: 955.8, Min: 947.7, Avg: 951.95
# Time per iteration (ms) - Max: 69151.8, Min: 68287.7, Avg: 68712.05
# Avg HIP Memory Usage: 128.11 GB (66.73%)
# Avg ROCm Memory Usage: 131.80 GB (68.65%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# Throughput (TFLOP/s/GPU) - Max: 280.3, Min: 279.75, Avg: 280.12
# Tokens (tokens/s/GPU) - Max: 807.55, Min: 806.05, Avg: 807.09
# Time per iteration (ms) - Max: 81149.1, Min: 81014.8, Avg: 81088.57
# Avg HIP Memory Usage: 124.12 GB (64.66%)
# Avg ROCm Memory Usage: 126.98 GB (66.14%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 3 \
#--moe_use_legacy_grouped_gemm true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_3.txt


# Throughput (TFLOP/s/GPU) - Max: 183.75, Min: 183.10000000000002, Avg: 183.38
# Tokens (tokens/s/GPU) - Max: 735.5, Min: 732.95, Avg: 734.06
# Time per iteration (ms) - Max: 89321.8, Min: 88918.4, Avg: 89161.23
# Avg HIP Memory Usage: 116.80 GB (60.84%)
# Avg ROCm Memory Usage: 119.27 GB (62.13%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer true \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--moe_use_legacy_grouped_gemm true \
#--use_turbo_grouped_mlp false \
#--use_turbo_attention false \
#--multi_latent_attention true \
#--turbo_sync_free_moe_stage 2 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_1_vpp_4_precision_aware_moe_use_legacy_grouped_gemm_mla_sync_moe_2.txt


# Throughput (TFLOP/s/GPU) - Max: 312.04999999999995, Min: 304.5, Avg: 308.48
# Tokens (tokens/s/GPU) - Max: 898.9, Min: 877.3, Avg: 888.69
# Time per iteration (ms) - Max: 74704.3, Min: 71793.3, Avg: 73370.15
# Avg HIP Memory Usage: 136.18 GB (70.94%)
# Avg ROCm Memory Usage: 139.87 GB (72.86%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
# 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_32_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# Throughput (TFLOP/s/GPU) - Max: 315.35, Min: 305.7, Avg: 312.46
# Tokens (tokens/s/GPU) - Max: 908.5, Min: 880.7, Avg: 900.16
# Time per iteration (ms) - Max: 74411.9, Min: 71619.7, Avg: 72627.98
# Avg HIP Memory Usage: 146.77 GB (76.45%)
# Avg ROCm Memory Usage: 150.46 GB (78.37%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_16_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# OOM Runtime error
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids -1 \
# 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_-1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# OOM Runtime error
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 0 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_2048_turbo_deepep_recompute_0_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# Runtime error: The agent attempted to access memory beyond the largest legal address. 
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--context_parallel_size 2 \
#--pipeline_model_parallel_size 8 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_8_cp_2_mbs_1_gbs_2048_turbo_deepep_recompute_2_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime error: Bug in NCCL 
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--tensor_model_parallel_size 2 \
#--pipeline_model_parallel_size 8 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 2048 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 2 2>&1 | tee log_16_fp8_pp_8_tp_2_mbs_1_gbs_2048_turbo_deepep_recompute_2_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# Throughput (TFLOP/s/GPU) - Max: 328.15, Min: 319.4, Avg: 323.74
# Tokens (tokens/s/GPU) - Max: 945.4, Min: 920.2, Avg: 932.65
# Time per iteration (ms) - Max: 142438.6, Min: 138055.8, Avg: 140293.88
# Avg HIP Memory Usage: 128.18 GB (66.77%)
# Avg ROCm Memory Usage: 131.87 GB (68.69%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion.txt


# Throughput (TFLOP/s/GPU) - Max: 335.3, Min: 329.1, Avg: 332.79
# Tokens (tokens/s/GPU) - Max: 965.95, Min: 948.1, Avg: 958.75
# Time per iteration (ms) - Max: 138253.0, Min: 135001.8, Avg: 136389.42
# Avg HIP Memory Usage: 128.18 GB (66.77%)
# Avg ROCm Memory Usage: 131.87 GB (68.69%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--recompute_granularity full \
#--recompute_method block \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Throughput (TFLOP/s/GPU) - Max: 348.5, Min: 343.2, Avg: 346.07
# Tokens (tokens/s/GPU) - Max: 1003.9, Min: 988.7, Avg: 996.98
# Time per iteration (ms) - Max: 132574.5, Min: 130559.8, Avg: 131476.55
# Avg HIP Memory Usage: 146.20 GB (76.15%)
# Avg ROCm Memory Usage: 149.90 GB (78.08%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 16 -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_mlp_moe.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_mlp_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
-N 16 --nodelist=j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_mlp_moe.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 16 \
--virtual_pipeline_model_parallel_size 4 \
--micro_batch_size 1 \
--global_batch_size 4096 \
--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
--use_turbo_deepep true \
--moe_shared_expert_overlap false \
--turbo_deepep_num_cu 80 \
--use_precision_aware_optimizer True \
--main_grads_dtype bf16 \
--main_params_dtype fp16 \
--exp_avg_dtype bf16 \
--exp_avg_sq_dtype bf16 \
--apply_rope_fusion true \
--turbo_sync_free_moe_stage 1 2>&1 | tee log_16_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_mlp_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1_250_iters.txt
