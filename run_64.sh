#!/bin/bash
set -x

#Throughput (TFLOP/s/GPU) - Max: 348.3, Min: 185.2, Avg: 328.49
#Tokens (tokens/s/GPU) - Max: 1003.3, Min: 533.6, Avg: 946.38
#Time per iteration (ms) - Max: 245624.0, Min: 130640.8, Avg: 138977.88
#Avg HIP Memory Usage: 113.61 GB (59.18%)
#Avg ROCm Memory Usage: 118.72 GB (61.84%)
#./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_fp8.yaml slurm -p mbzuai_training \
#-N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
#-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-FP8-pretrain_recompute_modules_moe.yaml \
#--train_iters 250 \
#--pipeline_model_parallel_size 16 \
#--virtual_pipeline_model_parallel_size 4 \
#--micro_batch_size 1 \
#--global_batch_size 16384 \
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
#2>&1 | tee log_64_fp8_pp_16_mbs_1_gbs_16384_turbo_deepep_recompute_selective_moe_vpp_4_precision_aware_rope_fusion_sync_moe_1_250_iters.txt

srun -p mbzuai_training -N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] -l sg docker -c "docker ps -q | xargs -r docker stop"

./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 8 \
--micro_batch_size 1 \
--global_batch_size 16384 \
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
2>&1 | tee log_64_bf16_pp_8_mbs_1_gbs_16384_turbo_deepep_precision_aware_rope_fusion_sync_moe_1_250_iters.txt

srun -p mbzuai_training -N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] -l sg docker -c "docker ps -q | xargs -r docker stop"

./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 8 \
--micro_batch_size 1 \
--global_batch_size 16384 \
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
2>&1 | tee log_64_bf16_pp_8_mbs_1_gbs_16384_turbo_deepep_recompute_15_precision_aware_rope_fusion_sync_moe_1_250_iters.txt

srun -p mbzuai_training -N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] -l sg docker -c "docker ps -q | xargs -r docker stop"

./primus-cli --config /home/amd/mbzuai_shared/Primus/prod_bf16.yaml slurm -p mbzuai_training \
-N 64 --nodelist=j16u[01,07,25,37,43],j17u[13,31,43],j18u[01,31,37,43],j19u[13,19,25,31,43],j20u[01,37],j21u[13,25,37],k20u[25,31,37,43],k21u[01,07,19,31,37,43],k16u[01,07,13,25,31,37],k17u[07,19,25,31,37,43],k18u[01,07,13,19,31],k19u[01,07,13,31,37],l16u07,l17u[31,37,43],l18u[01,19,37,43],l19u[01,13] \
-- train pretrain --config examples/megatron/configs/MI300X/deepseek_v3-BF16-pretrain.yaml \
--train_iters 250 \
--pipeline_model_parallel_size 16 \
--micro_batch_size 1 \
--global_batch_size 16384 \
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
2>&1 | tee log_64_bf16_pp_16_mbs_1_gbs_16384_turbo_deepep_precision_aware_rope_fusion_sync_moe_1_250_iters.txt