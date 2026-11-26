# Error analysis: /home/alibi/lab/models_v18_clean, test split (138 systems)

- Liquidus MAE 94.6 K over 12,863 points; mean signed error +3.5 K (predicted too high on average).
- The worst 10 % of systems carry 33 % of all liquidus error; median system 70 K, worst 728 K (In_Nb).
- Systems least like the training data (top quarter by distance) have liquidus MAE 156 K, the most similar quarter 58 K.
- Relative to the higher melting point the error is 3.7 % (median); its correlation with that melting point is -0.25, against +0.20 in kelvin (near 0 = high-melting systems are only hard because their temperatures are bigger).
- Clean systems (complete diagram, no volatile element, not monotectic; 85 of 137): liquidus MAE 82 K mean, 63 K median. Volatile-element systems (35): 134 K mean.
- Diagram type wrong: MAE 121 K vs 92 K when right.

## By data group (groups overlap; clean = none of the others)
incomplete = under half the composition range with liquidus data, or a gap of 10 at% or more; volatile element = an element boiling below 1300 K
                  systems  liquidus_mean_K  liquidus_median_K  solidus_mean_K  share_of_error_pct
group                                                                                            
all                   137             98.6               70.4           141.1               100.0
clean                  85             82.0               63.2           139.1                51.6
incomplete             16            145.7              111.6           236.1                17.3
volatile element       35            133.9               91.7           174.5                34.7
monotectic             23            141.1               94.1           130.3                24.0

## Data quality: coverage, gaps and straight-line data
Per system, by the share of the composition range with liquidus data:
          systems  mean_MAE_K  median_MAE_K  share_of_error_pct
coverage                                                       
< 50 %          8       176.0         167.3                10.4
50-80 %         7       120.9          93.0                 6.3
80-95 %        13       123.3         130.6                11.9
95-100 %      109        88.5          62.3                71.4

Per system, by the largest stretch of composition with no liquidus data:
                 systems  mean_MAE_K  median_MAE_K  share_of_error_pct
largest_gap_pct                                                       
< 2 at%              108        88.2          62.1                70.6
2-10 at%              12       109.4         106.9                 9.7
10-30 at%              3       166.2         196.5                 3.7
> 30 at%              14       154.4         104.1                16.0

Per data point, by how that part of the curve was drawn (straight = the source points were joined by a straight line over that many at%; such stretches are interpolated, not measured):
                    points  MAE_K  share_of_points_pct  share_of_error_pct
point_class                                                               
smooth curve          9834   96.0                 76.5                77.6
straight 5-15 at%      648   63.5                  5.0                 3.4
straight 15-30 at%     128   19.1                  1.0                 0.2
straight > 30 at%      736   68.3                  5.7                 4.1
flat plateau          1429  114.2                 11.1                13.4
next to a gap           88  174.3                  0.7                 1.3

## Main kind of error per system
          systems  mean_MAE_K  share_of_error_pct
kind                                             
eutectic       21        64.4                10.0
edges           7       105.1                 5.4
offset         26       184.6                35.6
shape          83        79.7                49.0

## By true diagram type (K)
                             liquidus_MAE_K        liquidus_bias_K       solidus_MAE_K       
                                     median   mean          median  mean        median   mean
topology                                                                                     
complex                                93.0  186.0           -47.0  82.0         193.0  222.0
eutectic_with_intermetallics           67.0   86.0            10.0  10.0         108.0  149.0
isomorphous                            24.0   50.0             4.0 -13.0          47.0   61.0
monotectic                             94.0  141.0             1.0 -57.0          61.0  130.0
simple_eutectic                        54.0   59.0            25.0  38.0          74.0   89.0

## Error near the eutectic (eutectic systems)
- 0-2 at% from it: 103 K (396 points)
- 2-5 at% from it: 98 K (575 points)
- 5-10 at% from it: 92 K (949 points)
- 10-20 at% from it: 82 K (1,787 points)
- 20-40 at% from it: 85 K (2,994 points)
- > 40 at% from it: 77 K (3,781 points)

## Error by steepness of the true liquidus
- 0-5 K/at%: 80 K (4,607 points)
- 5-10 K/at%: 78 K (2,182 points)
- 10-20 K/at%: 95 K (2,989 points)
- 20-40 K/at%: 129 K (2,174 points)
- 40-80 K/at%: 139 K (490 points)
- > 80 K/at%: 217 K (145 points)

## Elements in the worst systems (at least 3 systems)
         systems  mean_MAE_K
element                     
Nb             4       225.0
In             5       208.0
S              5       203.0
V              6       171.0
W              5       168.0
Li             5       163.0
Te             5       159.0
H              3       152.0
Hg             3       149.0
Sb             5       140.0
Zn            10       133.0
Eu             4       132.0

## Properties that go with high error (Spearman)
                                  property  spearman  p_value
               feature: phys__ideal_dev_50     0.495      0.0
               feature: phys__ideal_dev_60     0.480      0.0
feature: comp__MagpieData avg_dev MeltingT     0.480      0.0
  feature: comp__MagpieData range MeltingT     0.480      0.0
    feature: phys__melting_point__abs_diff     0.477      0.0
                  melting-point difference     0.477      0.0
            feature: phys__ideal_eut_depth    -0.476      0.0
       feature: phys__melting_point__ratio     0.472      0.0
             feature: thermo__Tm_asymmetry     0.472      0.0
               feature: phys__ideal_dev_40     0.471      0.0
             feature: phys__ideal_mean_dev     0.451      0.0
               feature: phys__ideal_dev_75     0.429      0.0
               feature: phys__ideal_dev_25     0.418      0.0
  distance to the nearest training systems     0.405      0.0
          feature: el__HeatFusion__absdiff     0.388      0.0

## Worst 15 systems
 pair                     topology           predicted_topology  liquidus_MAE_K  liquidus_bias_K   kind  nn_distance
In_Nb                      complex                      complex           728.5            728.4 offset         14.4
Ga_Mo                      complex                      complex           393.9            393.1 offset         13.6
Cr_Eu                   monotectic                   monotectic           343.5           -343.3 offset          8.9
 S_Zn                   monotectic                   monotectic           329.0           -304.9  edges         16.9
 Ir_P eutectic_with_intermetallics eutectic_with_intermetallics           328.5            328.1 offset         18.9
 Hg_S                   monotectic                   monotectic           326.7           -326.6 offset         14.1
 H_Li                   monotectic eutectic_with_intermetallics           287.9           -287.9 offset         33.0
 V_Zn eutectic_with_intermetallics                      complex           277.3            265.1 offset          8.5
 Te_Y                      complex eutectic_with_intermetallics           258.3            258.3 offset         12.4
Sb_Te eutectic_with_intermetallics eutectic_with_intermetallics           250.5            250.4 offset         12.3
 W_Zr eutectic_with_intermetallics eutectic_with_intermetallics           250.1           -238.3  shape         10.2
 K_Zn                   monotectic                   monotectic           242.8            241.3 offset         13.0
 In_V                   monotectic                   monotectic           240.4           -240.4 offset         10.9
Mn_Se                   monotectic                   monotectic           236.7            216.0  shape         10.0
 Cd_V                   monotectic                      complex           221.0           -208.0  edges         11.6
