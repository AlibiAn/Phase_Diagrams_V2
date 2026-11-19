#!/usr/bin/env bash
# Trains and scores v17 on the test split of processed_v8 (v14, v15: results/v15; v16: results/v16).
# v17 = v16 + liquidus height correction by TabPFN v2 + TabICL v2 (strength 0.5). Built with TabPFN.
set -u
export MPDS_DEVICE=cpu          # XGBoost on the CPU, as for v13-v16; the foundation models use the GPU
D8=/home/alibi/lab/processed_v8; R=results/v17; M=/home/alibi/lab/models_v17
if ! python -m mpds_models.train_all $D8 $M --curve-model extra_trees --curve-aggregate median \
    --swap-augment --mask-clipped --curve-solidus direct --train-with-val --features extended+mp \
    --extra-data /home/alibi/lab/extra_v8_all --height-correction 0.5 > $R/train_v17.log 2>&1; then
  echo "FAILED (see $R/train_v17.log)"; exit 1; fi
python predict.py $D8 $M --report $R/report_v17.json --mask-clipped --fix-end-points --no-uncertainty 2>&1 \
  | grep -v Warning > $R/predict_v17.txt
python compare_reports.py results/v16/report_v16.json $R/report_v17.json --names v16 v17 > $R/v16_vs_v17.txt 2>&1
python error_analysis.py $D8 $M --out $R/error_analysis --mask-clipped --fix-end-points 2>&1 \
  | grep -v "Warning\|fillna\|_strip_unit\|self.aos" > $R/error_analysis.txt
echo done
