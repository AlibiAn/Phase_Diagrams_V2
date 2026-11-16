#!/usr/bin/env bash
# Trains and scores the models of the v15 comparison on the test split of processed_v8.
# Usage: bash results/v15/train_and_score.sh <folder for the throw-away reference model>
set -u
export MPDS_DEVICE=cpu          # as v13 and v14: XGBoost builds different trees on the GPU
TMP=${1:?folder for throw-away models}
D8=/home/alibi/lab/processed_v8
R=results/v15
V14="--curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped --curve-solidus direct --train-with-val"
SCORE="--mask-clipped --fix-end-points --no-uncertainty"
run () {   # name, model folder, training options
  local name=$1 dir=$2; shift 2
  if ! python -m mpds_models.train_all $D8 "$dir" "$@" > "$R/train_$name.log" 2>&1; then
    echo "FAILED: $name (see $R/train_$name.log)"; return 1
  fi
  python predict.py $D8 "$dir" --report "$R/report_$name.json" $SCORE 2>&1 | grep -v Warning > "$R/predict_$name.txt"
  echo "done: $name"
}
run v14_on_v8 $TMP/models_v14_on_v8 $V14 --features extended                 # v14 settings, reference
run v15       /home/alibi/lab/models_v15 $V14 --features extended+mp         # v15
