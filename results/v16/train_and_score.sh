#!/usr/bin/env bash
# Trains and scores v16 on the test split of processed_v8 (v14 and v15 on it: results/v15).
set -u
export MPDS_DEVICE=cpu
D8=/home/alibi/lab/processed_v8; R=results/v16
if ! python -m mpds_models.train_all $D8 /home/alibi/lab/models_v16 --curve-model extra_trees --curve-aggregate median \
    --swap-augment --mask-clipped --curve-solidus direct --train-with-val --features extended+mp \
    --extra-data /home/alibi/lab/extra_v8_orchid > $R/train_v16.log 2>&1; then echo "FAILED (see $R/train_v16.log)"; exit 1; fi
python predict.py $D8 /home/alibi/lab/models_v16 --report $R/report_v16.json --mask-clipped --fix-end-points --no-uncertainty 2>&1 | grep -v Warning > $R/predict_v16.txt
echo done
