#!/bin/bash
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
set -ex

# try these values of lambda
lambdas="0.001 0.002 0.005 0.01 0.02 0.05"

dout=24           # output dimension
db=deep1b         # use this dataset
quant=zn_79       # cross-validate using this quantizer
best_lambda=-1
best_perf="0.0000"

for lambda in $lambdas; do
    mkdir -p test_ckpt/$lambda
    python -u train.py \
      --dout $dout \
      --save_best_criterion $quant,rank=10 \
      --database $db \
      --lambda_uniform $lambda \
      --checkpoint_dir test_ckpt/$lambda |
        tee test_ckpt/$lambda.stdout

    # extract validation accuracy
    perf=$(tac test_ckpt/$lambda.stdout |
                  grep -m1 'keeping as best' |
                  grep -o '(.*>' | grep -o '[0-9\.]*')

    echo $perf

    if [[ "$perf" > "$best_perf" ]]; then
        best_perf=$perf
        best_lambda=$lambda
    fi
done

echo "Best value of lambda: $best_lambda"

python eval.py \
       --database $db \
       --quantizer $quant \
       --ckpt-path test_ckpt/$best_lambda/checkpoint.pth.best
