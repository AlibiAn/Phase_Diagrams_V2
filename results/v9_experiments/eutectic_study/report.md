# Eutectic study
1242 systems (train + val), 5×5-fold CV; p-values: Nadeau–Bengio corrected paired t-test against XGBoost (M4).

## Classification (eutectic exists)
| model                |   f1_mean |   f1_std |   f1_vs_xgb |   f1_p_value |   f1_tuned_mean |   f1_tuned_vs_xgb |   f1_tuned_p_value |   roc_auc_mean |   roc_auc_vs_xgb |   roc_auc_p_value |   pr_auc_mean |   brier_mean |   train_f1_mean |   overfit_gap_f1 |   acc_high_disagreement_mean |   acc_rest_mean |   seconds_mean |
|:---------------------|----------:|---------:|------------:|-------------:|----------------:|------------------:|-------------------:|---------------:|-----------------:|------------------:|--------------:|-------------:|----------------:|-----------------:|-----------------------------:|----------------:|---------------:|
| XGBoost (M4)         |     0.903 |    0.009 |       0.000 |      nan     |           0.904 |             0.000 |            nan     |          0.853 |            0.000 |           nan     |         0.957 |        0.125 |           1.000 |            0.097 |                        0.625 |           0.841 |         11.991 |
| XGBoost (defaults)   |     0.899 |    0.011 |      -0.004 |        0.431 |           0.902 |            -0.003 |              0.511 |          0.850 |           -0.003 |             0.524 |         0.957 |        0.132 |           1.000 |            0.101 |                        0.666 |           0.834 |          0.891 |
| RandomForest         |     0.903 |    0.009 |       0.001 |        0.915 |           0.900 |            -0.004 |              0.385 |          0.858 |            0.005 |             0.359 |         0.958 |        0.113 |           1.000 |            0.097 |                        0.606 |           0.837 |          1.284 |
| ExtraTrees           |     0.903 |    0.008 |      -0.000 |        0.966 |           0.900 |            -0.004 |              0.461 |          0.845 |           -0.008 |             0.346 |         0.954 |        0.116 |           1.000 |            0.097 |                        0.589 |           0.838 |          1.080 |
| HistGradientBoosting |     0.899 |    0.009 |      -0.004 |        0.419 |           0.904 |            -0.000 |              0.904 |          0.853 |           -0.000 |             0.996 |         0.956 |        0.129 |           1.000 |            0.101 |                        0.670 |           0.837 |          3.537 |
| Logistic (linear)    |     0.834 |    0.024 |      -0.068 |        0.000 |         nan     |           nan     |            nan     |          0.823 |           -0.030 |             0.027 |         0.945 |        0.163 |           0.861 |            0.026 |                        0.719 |           0.758 |          0.019 |

## Learning curve (held-out ROC-AUC by training size)
| model        |   198 |   397 |   595 |   794 |   993 |   596 |   795 |   994 |
|:-------------|------:|------:|------:|------:|------:|------:|------:|------:|
| XGBoost (M4) | 0.789 | 0.837 | 0.832 | 0.842 | 0.852 | 0.850 | 0.862 | 0.864 |
| RandomForest | 0.804 | 0.835 | 0.817 | 0.840 | 0.837 | 0.854 | 0.866 | 0.871 |
| ExtraTrees   | 0.780 | 0.822 | 0.807 | 0.826 | 0.836 | 0.847 | 0.855 | 0.859 |

## Position regressors (MAE)
| target             | model                      |   mean |   std |
|:-------------------|:---------------------------|-------:|------:|
| composition_at_pct | mean of training rows      |   27.6 |   0.7 |
| composition_at_pct | XGBoost (M4)               |   20.4 |   1.2 |
| composition_at_pct | RandomForest               |   20.2 |   0.7 |
| composition_at_pct | ExtraTrees                 |   19.6 |   1.1 |
| temperature_K      | mean of training rows      |  436.9 |  31.8 |
| temperature_K      | fixed fraction of lower Tm |  249.0 |   8.0 |
| temperature_K      | XGBoost (M4)               |  161.9 |  16.9 |
| temperature_K      | RandomForest               |  161.3 |  12.0 |
| temperature_K      | ExtraTrees                 |  156.0 |  13.7 |

## Most important property groups
| model        | group                                   |   importance |
|:-------------|:----------------------------------------|-------------:|
| RandomForest | thermo__Yang omega                      |       0.0055 |
| RandomForest | thermo__Miedema_deltaH_inter            |       0.0052 |
| RandomForest | thermo__Miedema_deltaH_amor             |       0.0049 |
| RandomForest | phys__melting_point__ratio              |       0.0040 |
| RandomForest | thermo__Tm_asymmetry                    |       0.0031 |
| RandomForest | comp__MagpieData avg_dev CovalentRadius |       0.0024 |
| RandomForest | phys__boiling_point__ratio              |       0.0023 |
| RandomForest | thermo__Yang delta                      |       0.0020 |
| XGBoost (M4) | thermo__Miedema_deltaH_inter            |       0.0327 |
| XGBoost (M4) | thermo__Tm_asymmetry                    |       0.0254 |
| XGBoost (M4) | thermo__Yang omega                      |       0.0153 |
| XGBoost (M4) | comp__MagpieData range CovalentRadius   |       0.0093 |
| XGBoost (M4) | phys__boiling_point__ratio              |       0.0093 |
| XGBoost (M4) | thermo__Miedema_deltaH_amor             |       0.0073 |
| XGBoost (M4) | phys__group__ratio                      |       0.0071 |
| XGBoost (M4) | phys__melting_point__abs_diff           |       0.0059 |
