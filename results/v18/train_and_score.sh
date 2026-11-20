#!/usr/bin/env bash
# Trains and scores v18 on the test split of processed_v8 (v17: results/v17).
# v18 = v17 + the liquidus blended 50 % with a curve whose 10 principal components
# TabPFN v2 + TabICL v2 predict (foundation_curves.py). Built with TabPFN.
# Usage: bash results/v18/train_and_score.sh [feature set, default extended+mp] [model folder] [tag]
set -u
export MPDS_DEVICE=cpu          # XGBoost on the CPU, as for v13-v17; the foundation models use the GPU
FEATURES=${1:-extended+mp}; M=${2:-/home/alibi/lab/models_v18}; TAG=${3:-v18}
D8=/home/alibi/lab/processed_v8; R=results/v18
if ! python -m mpds_models.train_all $D8 $M --curve-model extra_trees --curve-aggregate median \
    --swap-augment --mask-clipped --curve-solidus direct --train-with-val --features $FEATURES \
    --extra-data /home/alibi/lab/extra_v8_all --height-correction 0.5 \
    --foundation-blend 0.5 --foundation-components 10 > $R/train_$TAG.log 2>&1; then
  echo "FAILED (see $R/train_$TAG.log)"; exit 1; fi
python predict.py $D8 $M --report $R/report_$TAG.json --mask-clipped --fix-end-points --no-uncertainty 2>&1 \
  | grep -v Warning > $R/predict_$TAG.txt
python compare_reports.py results/v17/report_v17.json $R/report_$TAG.json --names v17 $TAG > $R/v17_vs_$TAG.txt 2>&1
python error_analysis.py $D8 $M --out $R/error_analysis_$TAG --mask-clipped --fix-end-points 2>&1 \
  | grep -v "Warning\|fillna\|_strip_unit\|self.aos" > $R/error_analysis_$TAG.txt
echo done
