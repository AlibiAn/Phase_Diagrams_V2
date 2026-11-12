# Error analysis: /home/alibi/lab/models_v14_two_liquid, test split (138 systems)

- Liquidus MAE 103.5 K over 12,864 points; mean signed error -9.4 K (predicted too low on average).
- The worst 10 % of systems carry 34 % of all liquidus error; median system 72 K, worst 504 K (In_Nb).
- Systems least like the training data (top quarter by distance) have liquidus MAE 156 K, the most similar quarter 62 K.
- Relative to the higher melting point the error is 4.0 % (median); its correlation with that melting point is -0.19, against +0.22 in kelvin (near 0 = high-melting systems are only hard because their temperatures are bigger).
- Clean systems (complete diagram, no volatile element, not monotectic; 85 of 137): liquidus MAE 86 K mean, 67 K median. Volatile-element systems (35): 146 K mean.
- Diagram type wrong: MAE 123 K vs 104 K when right.

## By data group (groups overlap; clean = none of the others)
incomplete = under half the composition range with liquidus data, or a gap of 10 at% or more; volatile element = an element boiling below 1300 K
                  systems  liquidus_mean_K  liquidus_median_K  solidus_mean_K  share_of_error_pct
group                                                                                            
all                   137            108.3               71.7           162.9               100.0
clean                  85             85.6               67.2           151.4                49.0
incomplete             16            171.3              162.1           247.4                18.5
volatile element       35            146.3              102.3           188.8                34.5
monotectic             23            174.0              105.1           200.0                27.0

## Data quality: coverage, gaps and straight-line data
Per system, by the share of the composition range with liquidus data:
          systems  mean_MAE_K  median_MAE_K  share_of_error_pct
coverage                                                       
< 50 %          8       195.4         171.7                10.5
50-80 %         7       158.3         138.5                 7.5
80-95 %        13       165.1         131.2                14.5
95-100 %      109        92.0          66.9                67.5

Per system, by the largest stretch of composition with no liquidus data:
                 systems  mean_MAE_K  median_MAE_K  share_of_error_pct
largest_gap_pct                                                       
< 2 at%              108        91.8          65.8                66.8
2-10 at%              12       145.1         112.4                11.7
10-30 at%              3       236.1         198.3                 4.8
> 30 at%              14       176.8         115.4                16.7

Per data point, by how that part of the curve was drawn (straight = the source points were joined by a straight line over that many at%; such stretches are interpolated, not measured):
                    points  MAE_K  share_of_points_pct  share_of_error_pct
point_class                                                               
smooth curve          9827  102.9                 76.4                76.0
straight 5-15 at%      648   61.7                  5.0                 3.0
straight 15-30 at%     128   20.8                  1.0                 0.2
straight > 30 at%      736   75.8                  5.7                 4.2
flat plateau          1437  143.1                 11.2                15.5
next to a gap           88  179.8                  0.7                 1.2

## Main kind of error per system
          systems  mean_MAE_K  share_of_error_pct
kind                                             
eutectic       25        91.2                15.4
edges           6        45.4                 1.8
offset         26       200.6                35.1
shape          80        88.4                47.7

## By true diagram type (K)
                             liquidus_MAE_K        liquidus_bias_K        solidus_MAE_K       
                                     median   mean          median   mean        median   mean
topology                                                                                      
complex                               131.0  169.0           -33.0   35.0         190.0  295.0
eutectic_with_intermetallics           70.0   95.0             4.0    5.0         106.0  155.0
isomorphous                            32.0   49.0            16.0   -0.0          50.0   55.0
monotectic                            105.0  174.0           -14.0 -103.0         142.0  200.0
simple_eutectic                        47.0   61.0            17.0   42.0          87.0   96.0

## Error near the eutectic (eutectic systems)
- 0-2 at% from it: 111 K (396 points)
- 2-5 at% from it: 108 K (575 points)
- 5-10 at% from it: 101 K (949 points)
- 10-20 at% from it: 92 K (1,787 points)
- 20-40 at% from it: 97 K (2,994 points)
- > 40 at% from it: 85 K (3,782 points)

## Error by steepness of the true liquidus
- 0-5 K/at%: 93 K (4,610 points)
- 5-10 K/at%: 88 K (2,181 points)
- 10-20 K/at%: 103 K (2,989 points)
- 20-40 K/at%: 129 K (2,174 points)
- 40-80 K/at%: 150 K (489 points)
- > 80 K/at%: 224 K (145 points)

## Elements in the worst systems (at least 3 systems)
         systems  mean_MAE_K
element                     
S              5       235.0
V              6       227.0
H              3       226.0
Li             5       205.0
Hg             3       182.0
In             5       176.0
W              5       175.0
Nb             4       170.0
Eu             4       167.0
Cr             5       165.0
Zn            10       146.0
Cd             4       134.0

## Properties that go with high error (Spearman)
                                  property  spearman  p_value
feature: comp__MagpieData avg_dev MeltingT     0.475      0.0
  feature: comp__MagpieData range MeltingT     0.475      0.0
                  melting-point difference     0.473      0.0
    feature: phys__melting_point__abs_diff     0.473      0.0
       feature: phys__melting_point__ratio     0.459      0.0
             feature: thermo__Tm_asymmetry     0.459      0.0
               feature: phys__ideal_dev_50     0.455      0.0
            feature: phys__ideal_eut_depth    -0.446      0.0
               feature: phys__ideal_dev_60     0.440      0.0
               feature: phys__ideal_dev_40     0.437      0.0
             feature: phys__ideal_mean_dev     0.411      0.0
          feature: el__HeatFusion__absdiff     0.397      0.0
               feature: phys__ideal_dev_75     0.392      0.0
               feature: phys__ideal_dev_25     0.391      0.0
            feature: el__cohesive__absdiff     0.360      0.0

## Worst 15 systems
 pair                     topology           predicted_topology  liquidus_MAE_K  liquidus_bias_K     kind  nn_distance
In_Nb                      complex                      complex           503.8            503.7    shape         13.6
 S_Zn                   monotectic                   monotectic           442.8           -437.3   offset         15.9
 Cd_V                   monotectic                      complex           422.7           -421.5    shape         11.0
 H_Li                   monotectic                   monotectic           414.9           -414.9   offset         32.8
 Hg_S                   monotectic                   monotectic           414.6           -414.6   offset         13.8
Cr_Eu                   monotectic                   monotectic           408.1           -408.0   offset          8.8
 Ir_P eutectic_with_intermetallics eutectic_with_intermetallics           352.8            352.4 eutectic         18.6
Ni_Tl                   monotectic                   monotectic           333.9           -333.7   offset         10.5
 In_V                   monotectic                   monotectic           324.5           -324.5   offset         10.2
Ba_Be eutectic_with_intermetallics                   monotectic           313.9           -295.9   offset         16.4
Ga_Mo                      complex                      complex           280.7            280.0   offset         12.8
 P_Zn eutectic_with_intermetallics                   monotectic           274.1           -268.9   offset         16.2
Li_Rh eutectic_with_intermetallics eutectic_with_intermetallics           269.8           -247.4    shape         16.0
 Te_Y                      complex eutectic_with_intermetallics           254.7            254.7   offset         11.2
 V_Zn eutectic_with_intermetallics                      complex           251.3            240.0   offset          7.9
