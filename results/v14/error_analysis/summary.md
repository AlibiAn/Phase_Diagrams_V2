# Error analysis: /home/alibi/lab/models_v14, test split (138 systems)

- Liquidus MAE 112.9 K over 12,616 points; mean signed error -6.9 K (predicted too low on average).
- The worst 10 % of systems carry 39 % of all liquidus error; median system 86 K, worst 1917 K (Cd_V).
- Systems least like the training data (top quarter by distance) have liquidus MAE 181 K, the most similar quarter 71 K.
- Relative to the higher melting point the error is 4.4 % (median); its correlation with that melting point is -0.27, against +0.14 in kelvin (near 0 = high-melting systems are only hard because their temperatures are bigger).
- Clean systems (complete diagram, no volatile element, not monotectic; 91 of 137): liquidus MAE 99 K mean, 69 K median. Volatile-element systems (35): 198 K mean.
- Diagram type wrong: MAE 199 K vs 107 K when right.

## By data group (groups overlap; clean = none of the others)
incomplete = under half the composition range with liquidus data, or a gap of 10 at% or more; volatile element = an element boiling below 1300 K
                  systems  liquidus_mean_K  liquidus_median_K  solidus_mean_K  share_of_error_pct
group                                                                                            
all                   137            129.7               85.8           194.7               100.0
clean                  91             98.6               68.6           173.5                50.5
incomplete             16            163.2              125.2           230.5                14.7
volatile element       35            198.4              111.8           207.9                39.1
monotectic             14            363.0              254.0           421.8                28.6

## Data quality: coverage, gaps and straight-line data
Per system, by the share of the composition range with liquidus data:
          systems  mean_MAE_K  median_MAE_K  share_of_error_pct
coverage                                                       
< 50 %         11       335.1         137.5                20.7
50-80 %         8       152.9         123.1                 6.9
80-95 %        12       146.7         123.3                 9.9
95-100 %      106       104.7          69.4                62.5

Per system, by the largest stretch of composition with no liquidus data:
                 systems  mean_MAE_K  median_MAE_K  share_of_error_pct
largest_gap_pct                                                       
< 2 at%              105       104.5          68.6                61.7
2-10 at%              11       127.2         107.0                 7.9
10-30 at%              3       200.4         195.2                 3.4
> 30 at%              18       266.7         137.2                27.0

Per data point, by how that part of the curve was drawn (straight = the source points were joined by a straight line over that many at%; such stretches are interpolated, not measured):
                    points  MAE_K  share_of_points_pct  share_of_error_pct
point_class                                                               
smooth curve         10544  114.6                 83.6                84.9
straight 5-15 at%      669   70.4                  5.3                 3.3
straight 15-30 at%     171  110.6                  1.4                 1.3
straight > 30 at%      856   70.5                  6.8                 4.2
flat plateau           263  242.5                  2.1                 4.5
next to a gap          113  226.7                  0.9                 1.8

## Main kind of error per system
          systems  mean_MAE_K  share_of_error_pct
kind                                             
eutectic       22        77.7                 9.6
edges           6        70.7                 2.4
offset         32       270.3                48.7
shape          77        90.8                39.3

## By true diagram type (K)
                             liquidus_MAE_K        liquidus_bias_K        solidus_MAE_K       
                                     median   mean          median   mean        median   mean
topology                                                                                      
complex                               123.0  175.0           -43.0   44.0         166.0  339.0
eutectic_with_intermetallics           67.0   94.0            11.0    5.0         109.0  155.0
isomorphous                            34.0   52.0            13.0   -5.0          46.0   60.0
monotectic                            254.0  363.0          -106.0 -224.0         215.0  422.0
simple_eutectic                        91.0  123.0            23.0  -22.0          87.0  170.0

## Error near the eutectic (eutectic systems)
- 0-2 at% from it: 119 K (396 points)
- 2-5 at% from it: 117 K (575 points)
- 5-10 at% from it: 111 K (949 points)
- 10-20 at% from it: 105 K (1,783 points)
- 20-40 at% from it: 109 K (2,965 points)
- > 40 at% from it: 98 K (3,716 points)

## Error by steepness of the true liquidus
- 0-5 K/at%: 103 K (3,854 points)
- 5-10 K/at%: 102 K (2,314 points)
- 10-20 K/at%: 105 K (3,134 points)
- 20-40 K/at%: 136 K (2,300 points)
- 40-80 K/at%: 163 K (566 points)
- > 80 K/at%: 239 K (164 points)

## Elements in the worst systems (at least 3 systems)
         systems  mean_MAE_K
element                     
V              6       570.0
Cd             4       505.0
In             5       279.0
S              5       247.0
Li             5       221.0
Nb             4       194.0
H              3       190.0
Hg             3       187.0
W              5       182.0
Zn            10       178.0
Sr             4       152.0
Bi             3       128.0

## Properties that go with high error (Spearman)
                                      property  spearman  p_value
                   feature: phys__ideal_dev_50     0.391      0.0
                   feature: phys__ideal_dev_60     0.391      0.0
      distance to the nearest training systems     0.390      0.0
                feature: phys__ideal_eut_depth    -0.372      0.0
                   feature: phys__ideal_dev_40     0.366      0.0
                 feature: phys__ideal_mean_dev     0.365      0.0
                   feature: phys__ideal_dev_75     0.361      0.0
                 feature: thermo__Tm_asymmetry     0.352      0.0
           feature: phys__melting_point__ratio     0.352      0.0
                        feature: el__is_bcc__B     0.350      0.0
   feature: comp__MagpieData mode AtomicWeight    -0.350      0.0
feature: comp__MagpieData minimum AtomicWeight    -0.350      0.0
      feature: comp__MagpieData minimum Number    -0.348      0.0
         feature: comp__MagpieData mode Number    -0.348      0.0
    feature: comp__MagpieData avg_dev MeltingT     0.347      0.0

## Worst 15 systems
 pair                     topology           predicted_topology  liquidus_MAE_K  liquidus_bias_K   kind  nn_distance
 Cd_V                   monotectic                      complex          1917.1          -1917.1 offset         11.0
 In_V              simple_eutectic                      complex           730.4           -730.4 offset         10.2
In_Nb                      complex                      complex           603.6            603.3 offset         13.6
 S_Zn                   monotectic                   monotectic           488.7           -481.8 offset         15.9
 Hg_S                   monotectic                   monotectic           424.7           -424.7 offset         13.8
 H_Li                   monotectic                   monotectic           371.0           -371.0 offset         32.8
 Ir_P eutectic_with_intermetallics eutectic_with_intermetallics           358.6            358.2 offset         18.6
Li_Rh eutectic_with_intermetallics eutectic_with_intermetallics           335.8           -314.4  shape         16.0
Nd_Sr                   monotectic                   monotectic           319.7            317.3 offset         13.1
Ba_Be eutectic_with_intermetallics                   monotectic           296.7           -275.4 offset         16.4
 K_Zn                   monotectic eutectic_with_intermetallics           266.9            266.1 offset         12.4
Mn_Se                   monotectic                   monotectic           266.1             64.2  shape          9.4
Ga_Mo                      complex                      complex           262.9            262.1 offset         12.8
Li_Sb eutectic_with_intermetallics eutectic_with_intermetallics           243.3           -241.8 offset         10.0
Ni_Tl                   monotectic                   monotectic           241.9           -241.5  shape         10.5
