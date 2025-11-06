#!/bin/bash


LD_LIBRARY_PATH="${PWD}/astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/build/lib:${LD_LIBRARY_PATH}" AS_LOG_LEVEL=DEBUG AS_NVLS_ENABLE=1 NS_GLOBAL_VALUE_IntHeader_Mode=Normal ./bin/SimAI_simulator     -t 16     -w ./workloads/None-gpt_7B-world_size512-tp2-pp8-ep1-gbs2048-mbs1-seq1024-MOE-False-GEMM-False-flash_attn-True.txt     -n "${PWD}/topo/Spectrum_OK"     -c "${PWD}/astra-sim-alibabacloud/inputs/config/SimAI.conf"