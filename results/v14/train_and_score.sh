#!/usr/bin/env bash
# Trains and scores every model of the v14 comparison on the test split.
# Usage: bash results/v14/train_and_score.sh <folder for the throw-away reference models>
# The v14 models go to /home/alibi/lab/models_v14 (processed_v6 targets) and
# /home/alibi/lab/models_v14_two_liquid (processed_v7 targets).
set -u
TMP=${1:?folder for throw-away models}
D6=/home/alibi/lab/processed_v6
D7=/home/alibi/lab/processed_v7
R=results/v14
V13="--curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped"
V14="$V13 --features extended --curve-solidus direct"
SCORE="--mask-clipped --fix-end-points --no-uncertainty"

run () {   # name, dataset, model folder, training options
  local name=$1 data=$2 dir=$3; shift 3
  python -m mpds_models.train_all "$data" "$dir" "$@" > "$R/train_$name.log" 2>&1
  python predict.py "$data" "$dir" --report "$R/report_$name.json" $SCORE 2>&1 | grep -v Warning > "$R/predict_$name.txt"
  echo "done: $name"
}

run v13_rerun        $D6 $TMP/models_v13_rerun        $V13                     # must equal results/v13/report_v13.json
run v13_with_val     $D6 $TMP/models_v13_with_val     $V13 --train-with-val    # how much the extra 138 systems give
run v14_without_val  $D6 $TMP/models_v14_without_val  $V14                     # how much features + solidus give
run v14              $D6 /home/alibi/lab/models_v14   $V14 --train-with-val    # the v14 model
run v13_two_liquid   $D7 $TMP/models_v13_two_liquid   $V13                     # v13 settings on the corrected targets
run v14_two_liquid   $D7 /home/alibi/lab/models_v14_two_liquid $V14 --train-with-val
