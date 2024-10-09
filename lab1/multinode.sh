#!/usr/bin/env bash

# On 104.171.200.62 (the master node)
torchrun \
--nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=104.171.200.62 --master_port=1234 \
example2.py

# On 104.171.200.182 (the worker node)
torchrun \
--nproc_per_node=2 --nnodes=1 --node_rank=1 \
--master_addr=104.171.200.62 --master_port=1234 \
example2.py