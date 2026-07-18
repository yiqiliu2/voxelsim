#!/bin/bash

mkdir -p parsed
mkdir -p TExpr

seq_lengths=( 128 )
batchsizes=( 1 2 4 8 16 32 64 128 256 512 )
models=( "resnet" "vit" "bert" )

for modelname in "${models[@]}" ; do
    for batchsize in "${batchsizes[@]}" ; do
        for seq_length in "${seq_lengths[@]}" ; do
            echo "Parsing ${modelname}-b${batchsize}"
            python3 model_parser.py "${modelname}.json" ${batchsize} ${seq_length}
        done
    done
done

echo "Parsing nerf-b1"
python3 model_parser.py "nerf.json" 1 1
echo "Parsing nerf-b1"
python3 model_parser.py "retnet.json" 1 1