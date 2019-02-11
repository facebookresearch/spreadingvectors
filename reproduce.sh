#!/bin/bash
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

if [ ! -d ckpts/ ]; then
    wget http://dl.fbaipublicfiles.com/spreadingvectors/ckpt.zip
    unzip ckpt.zip
fi

echo "reproduce Catalyzer+sign in table 2"
for f in ckpts/binary/{bigann,deep1b}/ckpt_{16,32,64,128}.pth; do
    echo $f
    python eval.py --quantizer binary --ckpt-path $f
done

echo "reproduce Catalyzer+Lattice (+end2end) in table 1"
for f in ckpts/lattice/**/*.pth; do
    echo $f
    python eval.py --quantizer zn_79 --ckpt-path $f
done

echo "reproduce Catalyzer+OPQ in table 1"
for f in ckpts/lattice/**/ckpt.pth; do
    echo $f
    python eval.py --quantizer opq_64 --ckpt-path $f
done
