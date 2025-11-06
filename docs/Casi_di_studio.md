# Casi di studio di SimAI

## Modello Small-Scale con Alta Parallelizzazione:

**Caso di studio**: modello 7B con elevato parallelismo (TP=8, PP=4) su 512 GPU, utile per analizzare l'overhead di comunicazione con sequenze lunghe (8K tokens).

```bash
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 512 --tensor_model_parallel_size 8 --pipeline_model_parallel 4 \
--frame Megatron --global_batch 2048 \
--micro_batch 1 --seq_length 8192 --swiglu \
--use_flash_attn --enable_sequence_parallel --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt 
```

## Modello Large-Scale con ZeRO Stage 3

**Caso di studio**: modello 175B con DeepSpeed ZeRO-3 su 2048 GPU, ottimo per studiare memory efficiency e parameter prefetching in scenari extreme-scale

``` bash
sh ./scripts/megatron_workload_with_aiob.sh -m 175 \
--world_size 2048 --tensor_model_parallel_size 4 --pipeline_model_parallel 8 \
--frame DeepSpeed --global_batch 4096 \
--micro_batch 2 --seq_length 4096 --swiglu \
--use_flash_attn --zero_stage 3 --prefetch_bucket_size 50000000 \
--max_live_parameters 1000000000 --aiob_enable
```

## DeepSeek MoE con Multi-Expert Configuration

``` bash 
sh ./scripts/megatron_workload_with_aiob.sh -m 236 \
--world_size 1024 --tensor_model_parallel_size 4 --pipeline_model_parallel 4 \
--expert_model_parallel_size 8 --frame DeepSeek --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --moe_enable \
--num_experts 64 --moe_router_topk 6 --n_shared_expert 2 \
--moe_grouped_gemm --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt 
```

**Caso di studio**: DeepSeek 236B MoE con 64 esperti e routing top-6, ideale per analizzare load balancing, expert parallelism e pattern di comunicazione in architetture Mixture-of-E