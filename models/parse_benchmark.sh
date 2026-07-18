#!/bin/bash

seq_lengths=( 256 )
batchsizes=( 1 2 4 8 16 32 64 128 256 512 )

for seq_length in "${seq_lengths[@]}" ; do
    for batchsize in "${batchsizes[@]}" ; do
        echo "Parsing size: ${batchsize} "
        python3 model_parser.py "decode_attn.json" ${batchsize} ${seq_length}
        python3 model_parser.py "decode_kv.json" ${batchsize} ${seq_length}
        python3 model_parser.py "prefill_attn.json" ${batchsize} ${seq_length}
        python3 model_parser.py "prefill_kv.json" ${batchsize} ${seq_length}
    done
done