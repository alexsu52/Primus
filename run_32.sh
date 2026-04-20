#!/bin/bash
set -x

#Throughput (TFLOP/s/GPU) - Max: 331.3, Min: 327.7, Avg: 329.80
#Tokens (tokens/s/GPU) - Max: 954.5, Min: 944.1, Avg: 950.15
#Time per iteration (ms) - Max: 69417.0, Min: 68659.0, Avg: 68975.48
#Avg HIP Memory Usage: 89.85 GB (46.80%)
#Avg ROCm Memory Usage: 95.03 GB (49.50%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
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
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: CoreDump
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 2 \
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
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: CoreDump
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
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
#--recompute_num_layers 2 2>&1 | tee log_32_fp8_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_2_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: CoreDump
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
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
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 4 2>&1 | tee log_32_fp8_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: Timeout
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--expert_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
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
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_16_ep_16_mbs_2_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: Timeout
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--expert_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
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
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 2 2>&1 | tee log_32_fp8_pp_16_ep_16_mbs_2_gbs_4096_turbo_deepep_recompute_2_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: Timeout
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--expert_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
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
#--turbo_sync_free_moe_stage 1 \
#--recompute_num_layers 4 2>&1 | tee log_32_fp8_pp_16_ep_16_mbs_2_gbs_4096_turbo_deepep_recompute_4_precision_aware_rope_fusion_sync_moe_1.txt


#Throughput (TFLOP/s/GPU) - Max: 326.5, Min: 324.9, Avg: 325.75
#Tokens (tokens/s/GPU) - Max: 940.8, Min: 936.1, Avg: 938.52
#Time per iteration (ms) - Max: 70013.3, Min: 69663.5, Avg: 69831.57
#Avg HIP Memory Usage: 57.95 GB (30.18%)
#Avg ROCm Memory Usage: 61.95 GB (32.27%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--virtual_pipeline_model_parallel_size 2 \
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
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_32_mbs_1_gbs_4096_turbo_deepep_recompute_1_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt


# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 2 \
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
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_32_mbs_2_gbs_4096_turbo_deepep_recompute_1_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
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
#--recompute_num_layers 2 2>&1 | tee log_32_fp8_pp_32_mbs_2_gbs_4096_turbo_deepep_recompute_2_precision_aware_rope_fusion_sync_moe_1.txt


#[Primus] Error: RuntimeError: Training execution failed: Rank 55, node g21u01, device 7, iteration 1: Unexpected result nan (message='found NaN in local grad norm for bucket #0 in backward pass before data-parallel communication collective') (/home/amd/mbzuai_shared/Primus/primus/core/runtime/train_runtime.py:110)[0m
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h18u[01,13],g21u[01,07,13,19,37,43],g20u[01,07,25,31,37,43],h21u[01,07,13,19,31,37],h20u[07,19,25],h17u[07,13,31],h16u[07,13,19,25,31,37] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--virtual_pipeline_model_parallel_size 2 \
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
#--recompute_num_layers 0 2>&1 | tee log_32_fp8_pp_32_mbs_1_gbs_4096_turbo_deepep_recompute_0_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt


#Excluding first and last 3 iterations from final calculations.
#Throughput (TFLOP/s/GPU) - Max: 289.0, Min: 268.2, Avg: 281.73
#Tokens (tokens/s/GPU) - Max: 832.5, Min: 772.7, Avg: 811.62
#Time per iteration (ms) - Max: 84818.4, Min: 78717.7, Avg: 80812.98
#Avg HIP Memory Usage: 57.02 GB (29.70%)
#Avg ROCm Memory Usage: 59.78 GB (31.14%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
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
#--recompute_granularity selective 2>&1 | tee log_32_fp8_pp_32_mbs_1_gbs_4096_turbo_deepep_recompute_selective_precision_aware_rope_fusion_sync_moe_1.txt


# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h20u31,j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 32 \
#--virtual_pipeline_model_parallel_size 2 \
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
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity selective 2>&1 | tee log_32_fp8_pp_32_mbs_1_gbs_4096_turbo_deepep_recompute_selective_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt


# Runtime error: Bug in NCCL 
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=h18u[01,13],g21u[01,07,13,19,37,43],g20u[01,07,25,31,37,43],h21u[01,07,13,19,31,37],h20u[07,19,25],h17u[07,13,31],h16u[07,13,19,25,31,37] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain.yaml \
#--train_iters 10 \
#--tensor_model_parallel_size 2 \
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
#--recompute_num_layers 1 2>&1 | tee log_32_fp8_pp_16_tp_2_mbs_1_gbs_4096_turbo_deepep_recompute_1_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt


# Throughput (TFLOP/s/GPU) - Max: 348.4, Min: 304.1, Avg: 323.93
# Tokens (tokens/s/GPU) - Max: 1003.7, Min: 876.2, Avg: 933.23
# Time per iteration (ms) - Max: 74793.0, Min: 65291.5, Avg: 70413.23
# Avg HIP Memory Usage: 119.57 GB (62.28%)
# Avg ROCm Memory Usage: 124.69 GB (64.95%)
# /home/amd/mbzuai_shared/Primus/log_32_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_no_core_attn_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Throughput (TFLOP/s/GPU) - Max: 356.0, Min: 305.1, Avg: 338.25
# Tokens (tokens/s/GPU) - Max: 1025.7, Min: 879.0, Avg: 974.52
# Time per iteration (ms) - Max: 74559.3, Min: 63891.3, Avg: 67508.05
# Avg HIP Memory Usage: 118.97 GB (61.97%)
# Avg ROCm Memory Usage: 124.16 GB (64.67%)
# /home/amd/mbzuai_shared/Primus/log_32_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_mlp_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

# Throughput (TFLOP/s/GPU) - Max: 355.8, Min: 310.3, Avg: 339.58
# Tokens (tokens/s/GPU) - Max: 1025.1, Min: 893.9, Avg: 978.33
# Time per iteration (ms) - Max: 73311.1, Min: 63932.9, Avg: 67194.95
# Avg HIP Memory Usage: 118.81 GB (61.89%)
# Avg ROCm Memory Usage: 124.00 GB (64.59%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,31,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
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
#--turbo_sync_free_moe_stage 1 \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

#Throughput (TFLOP/s/GPU) - Max: 354.6, Min: 200.2, Avg: 331.43
#Tokens (tokens/s/GPU) - Max: 1021.7, Min: 576.8, Avg: 954.82
#Time per iteration (ms) - Max: 227238.6, Min: 128292.6, Avg: 137702.79
#Avg HIP Memory Usage: 120.24 GB (62.63%)
#Avg ROCm Memory Usage: 125.42 GB (65.33%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,31,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_mlp_moe.yaml \
#--train_iters 250 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 8192 \
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
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_mbs_1_gbs_8192_turbo_deepep_recompute_selective_mlp_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1_250_iters.txt

#Timeout error
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--expert_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"E|(t|)*61|L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 32 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--turbo_sync_free_moe_stage 1 \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_ep_16_mbs_1_gbs_4096_turbo_deepep_recompute_selective_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

#Throughput (TFLOP/s/GPU) - Max: 73.2, Min: 71.6, Avg: 72.47
#Tokens (tokens/s/GPU) - Max: 210.8, Min: 206.4, Avg: 208.85
#Time per iteration (ms) - Max: 317507.7, Min: 310854.5, Avg: 313806.00
#Avg HIP Memory Usage: 57.31 GB (29.85%)
#Avg ROCm Memory Usage: 62.05 GB (32.32%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
#--train_iters 10 \
#--tensor_model_parallel_size 2 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
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
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_tp_2_mbs_1_gbs_4096_turbo_deepep_recompute_selective_moe_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
#--train_iters 10 \
#--tensor_model_parallel_size 2 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
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
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_tp_2_mbs_2_gbs_4096_turbo_deepep_recompute_selective_moe_vpp_2_precision_aware_rope_fusion_sync_moe_1.txt


# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,31,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,31,43],j20u[01,37],j21u[13,25,37],k20u[01,19,25,31,37,43],k21u[07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_mlp.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 8192 \
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
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_mbs_1_gbs_8192_turbo_deepep_recompute_selective_mlp_vpp_4_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
#--train_iters 250 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 8192 \
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
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#2>&1 | tee log_32_fp8_pp_16_mbs_1_gbs_8192_turbo_deepep_recompute_selective_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1_250_iters.txt



######### BF16 #########
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids 4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,36 \
#2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_16_precision_aware_rope_fusion_sync_moe_1.txt


# ./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
# -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
# -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
# --train_iters 10 \
# --pipeline_model_parallel_size 16 \
# --micro_batch_size 1 \
# --global_batch_size 4096 \
# --pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
# --use_turbo_deepep true \
# --moe_shared_expert_overlap false \
# --turbo_deepep_num_cu 80 \
# --use_precision_aware_optimizer True \
# --main_grads_dtype bf16 \
# --main_params_dtype fp16 \
# --exp_avg_dtype bf16 \
# --exp_avg_sq_dtype bf16 \
# --apply_rope_fusion true \
# --data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
# --turbo_sync_free_moe_stage 1 \
# --recompute_granularity full \
# --recompute_layer_ids 4,8,12,16,20,24,28,32 \
# 2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_8_precision_aware_rope_fusion_sync_moe_1.txt

# srun -p mbzuai_training -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] -l sg docker -c "docker ps -q | xargs -r docker stop"

# ./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
# -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
# -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
# --train_iters 10 \
# --pipeline_model_parallel_size 16 \
# --micro_batch_size 1 \
# --global_batch_size 4096 \
# --pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
# --use_turbo_deepep true \
# --moe_shared_expert_overlap false \
# --turbo_deepep_num_cu 80 \
# --use_precision_aware_optimizer True \
# --main_grads_dtype bf16 \
# --main_params_dtype fp16 \
# --exp_avg_dtype bf16 \
# --exp_avg_sq_dtype bf16 \
# --apply_rope_fusion true \
# --data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
# --turbo_sync_free_moe_stage 1 \
# --recompute_granularity full \
# --recompute_method block \
# --recompute_num_layers 1 2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_1_precision_aware_rope_fusion_sync_moe_1.txt

# srun -p mbzuai_training -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] -l sg docker -c "docker ps -q | xargs -r docker stop"

# ./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
# -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
# -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain_recompute_modules_moe.yaml \
# --train_iters 10 \
# --pipeline_model_parallel_size 16 \
# --micro_batch_size 1 \
# --global_batch_size 4096 \
# --pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
# --use_turbo_deepep true \
# --moe_shared_expert_overlap false \
# --turbo_deepep_num_cu 80 \
# --use_precision_aware_optimizer True \
# --main_grads_dtype bf16 \
# --main_params_dtype fp16 \
# --exp_avg_dtype bf16 \
# --exp_avg_sq_dtype bf16 \
# --apply_rope_fusion true \
# --data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
# --turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_moe_precision_aware_rope_fusion_sync_moe_1.txt


#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids 4,8 \
#2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_recompute_4_8_precision_aware_rope_fusion_sync_moe_1.txt


#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache16 \
#--turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_precision_aware_rope_fusion_sync_moe_1.txt


#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 4 2>&1 | tee log_32_bf16_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_4_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 2 2>&1 | tee log_32_bf16_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_2_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: OOM
# ./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
# -N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
# -- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
# --train_iters 10 \
# --pipeline_model_parallel_size 16 \
# --micro_batch_size 2 \
# --global_batch_size 4096 \
# --pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
# --use_turbo_deepep true \
# --moe_shared_expert_overlap false \
# --turbo_deepep_num_cu 80 \
# --use_precision_aware_optimizer True \
# --main_grads_dtype bf16 \
# --main_params_dtype fp16 \
# --exp_avg_dtype bf16 \
# --exp_avg_sq_dtype bf16 \
# --apply_rope_fusion true \
# --data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
# --turbo_sync_free_moe_stage 1 \
# --recompute_granularity full \
# --recompute_method block \
# --recompute_num_layers 1 2>&1 | tee log_32_bf16_pp_16_mbs_2_gbs_4096_turbo_deepep_recompute_1_precision_aware_rope_fusion_sync_moe_1.txt


#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 4 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 4 2>&1 | tee log_32_bf16_pp_16_mbs_4_gbs_4096_turbo_deepep_recompute_4_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 4 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 3 2>&1 | tee log_32_bf16_pp_16_mbs_4_gbs_4096_turbo_deepep_recompute_3_precision_aware_rope_fusion_sync_moe_1.txt

# Runtime Error: OOM
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--micro_batch_size 2 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache16 \
#--turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_32_mbs_2_gbs_4096_turbo_deepep_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 8 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*7|t*7|t*8|t*8|t*8|t*8|t*8|t*7L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache16 \
#--turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_8_mbs_1_gbs_4096_turbo_deepep_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 8 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*7|t*7|t*8|t*8|t*8|t*8|t*8|t*7L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_method block \
#--recompute_num_layers 1 2>&1 | tee log_32_bf16_pp_8_mbs_1_gbs_4096_turbo_deepep_recompute_1_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 8 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*7|t*7|t*8|t*8|t*8|t*8|t*8|t*7L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#--recompute_granularity full \
#--recompute_layer_ids 15 \
#2>&1 | tee log_32_bf16_pp_8_mbs_1_gbs_4096_turbo_deepep_recompute_15_precision_aware_rope_fusion_sync_moe_1.txt

#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--expert_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
#--turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_16_ep_4_mbs_1_gbs_4096_turbo_deepep_precision_aware_rope_fusion_sync_moe_1.txt


#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
#-N 32 --nodelist=k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
#--train_iters 10 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 2 \
#--micro_batch_size 1 \
#--global_batch_size 4096 \
#--pipeline_model_parallel_layout "\"Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L\"" \
#--use_turbo_deepep true \
#--moe_shared_expert_overlap false \
#--turbo_deepep_num_cu 80 \
#--use_precision_aware_optimizer True \
#--main_grads_dtype bf16 \
#--main_params_dtype fp16 \
#--exp_avg_dtype bf16 \
#--exp_avg_sq_dtype bf16 \
#--apply_rope_fusion true \
#--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache16 \
#--turbo_sync_free_moe_stage 1 \
#2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_4096_turbo_deepep_vvp_2_precision_aware_rope_fusion_sync_moe_1.txt

./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 32 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 16 \
--micro_batch_size 1 \
--global_batch_size 8192 \
--pipeline_model_parallel_layout "\"Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3L\"" \
--use_turbo_deepep true \
--moe_shared_expert_overlap false \
--turbo_deepep_num_cu 80 \
--use_precision_aware_optimizer True \
--main_grads_dtype bf16 \
--main_params_dtype fp16 \
--exp_avg_dtype bf16 \
--exp_avg_sq_dtype bf16 \
--apply_rope_fusion true \
--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache \
--turbo_sync_free_moe_stage 1 \
2>&1 | tee log_32_bf16_pp_16_mbs_1_gbs_8192_turbo_deepep_precision_aware_rope_fusion_sync_moe_1_250_iters.txt

./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 32 --nodelist=k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 8 \
--micro_batch_size 1 \
--global_batch_size 8192 \
--pipeline_model_parallel_layout "\"Et*7|t*7|t*8|t*8|t*8|t*8|t*8|t*7L\"" \
--use_turbo_deepep true \
--moe_shared_expert_overlap false \
--turbo_deepep_num_cu 80 \
--use_precision_aware_optimizer True \
--main_grads_dtype bf16 \
--main_params_dtype fp16 \
--exp_avg_dtype bf16 \
--exp_avg_sq_dtype bf16 \
--apply_rope_fusion true \
--data_cache_path /home/amd/datasets/c4_deepseek_subset/cache16 \
--turbo_sync_free_moe_stage 1 \
--recompute_granularity full \
--recompute_layer_ids 15 \
2>&1 | tee log_32_bf16_pp_8_mbs_1_gbs_8192_turbo_deepep_recompute_15_precision_aware_rope_fusion_sync_moe_1_250_iters.txt
