#!/bin/sh

python gen_poisson_trace.py swe-bench-base.jsonl \
	--rate 0.55 \
	--num-sessions 150 \
	--output workload.jsonl \
	--seed 42
