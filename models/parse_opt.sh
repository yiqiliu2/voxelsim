#!/bin/bash

seq_lengths=( 1 )
batchsizes=( 1 2 4 8 16 32 64 128 256 512 )
model_sizes=( "1.3" "2.7" "6.7" "13" "30" )

for seq_length in "${seq_lengths[@]}" ; do
    for model_size in "${model_sizes[@]}" ; do
        for batchsize in "${batchsizes[@]}" ; do
            echo "Parsing size: ${batchsize} "
            python3 model_parser.py "opt-${model_size}.json" ${batchsize} ${seq_length}
        done
    done
done

batchsizes=( 1 2 4 8 16 32 64 128 256 512 )
model_sizes=( "7" "13" )

for seq_length in "${seq_lengths[@]}" ; do
    for model_size in "${model_sizes[@]}" ; do
        for batchsize in "${batchsizes[@]}" ; do
            echo "Parsing size: ${batchsize} "
            python3 model_parser.py "llama2-${model_size}.json" ${batchsize} ${seq_length}
        done
    done
done
