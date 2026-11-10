# Error analysis: /home/alibi/lab/models_v11_trees, test split (138 systems)

- Liquidus MAE 133.3 K over 12,616 points; mean signed error -7.6 K (predicted too low on average).
- The worst 10 % of systems carry 38 % of all liquidus error; median system 104 K, worst 1993 K (Cd_V).
- Systems least like the training data (top quarter by distance) have liquidus MAE 190 K, the most similar quarter 123 K.
- Relative to the higher melting point the error is 5.4 % (median); its correlation with that melting point is -0.34, against +0.07 in kelvin (near 0 = high-melting systems are only hard because their temperatures are bigger).
- Diagram type wrong: MAE 227 K vs 122 K when right.

## Data quality: coverage, gaps and straight-line data
Per system, by the share of the composition range with liquidus data:
          systems  mean_MAE_K  median_MAE_K  share_of_error_pct
coverage                                                       
< 50 %         11       345.8         203.4                18.6
50-80 %         8       181.6         155.0                 7.1
80-95 %        12       142.7         119.2                 8.3
95-100 %      106       127.7          92.1                66.0

Per system, by the largest stretch of composition with no liquidus data:
                 systems  mean_MAE_K  median_MAE_K  share_of_error_pct
largest_gap_pct                                                       
< 2 at%              105       127.9          92.1                65.5
2-10 at%              11       117.1         107.1                 6.3
10-30 at%              3       226.2         197.1                 3.3
> 30 at%              18       283.7         177.7                24.9

Per data point, by how that part of the curve was drawn (straight = the source points were joined by a straight line over that many at%; such stretches are interpolated, not measured):
                    points  MAE_K  share_of_points_pct  share_of_error_pct
point_class                                                               
smooth curve         10544  138.2                 83.6                86.6
straight 5-15 at%      669   81.0                  5.3                 3.2
straight 15-30 at%     171   84.2                  1.4                 0.9
straight > 30 at%      856   83.1                  6.8                 4.2
flat plateau           263  218.4                  2.1                 3.4
next to a gap          113  245.8                  0.9                 1.7

## Main kind of error per system
          systems  mean_MAE_K  share_of_error_pct
kind                                             
eutectic       14        95.1                 6.5
edges           2       173.0                 1.7
offset         38       252.5                46.8
shape          83       111.3                45.0

## By true diagram type (K)
                             liquidus_MAE_K        liquidus_bias_K        solidus_MAE_K       
                                     median   mean          median   mean        median   mean
topology                                                                                      
complex                               125.0  196.0           -40.0   -1.0         280.0  321.0
eutectic_with_intermetallics           78.0  103.0            19.0   15.0         128.0  165.0
isomorphous                            18.0   48.0             1.0  -16.0          44.0   53.0
monotectic                            253.0  376.0          -129.0 -237.0         297.0  418.0
simple_eutectic                       138.0  206.0            46.0  -33.0         113.0  192.0

## Error near the eutectic (eutectic systems)
- 0-2 at% from it: 144 K (396 points)
- 2-5 at% from it: 142 K (575 points)
- 5-10 at% from it: 133 K (949 points)
- 10-20 at% from it: 125 K (1,783 points)
- 20-40 at% from it: 129 K (2,965 points)
- > 40 at% from it: 118 K (3,716 points)

## Error by steepness of the true liquidus
- 0-5 K/at%: 121 K (3,854 points)
- 5-10 K/at%: 135 K (2,314 points)
- 10-20 K/at%: 128 K (3,134 points)
- 20-40 K/at%: 149 K (2,300 points)
- 40-80 K/at%: 175 K (566 points)
- > 80 K/at%: 229 K (164 points)

## Elements in the worst systems (at least 3 systems)
         systems  mean_MAE_K
element                     
V              6       662.0
Cd             4       524.0
In             5       272.0
S              5       240.0
Cr             5       233.0
Li             5       209.0
Zn            10       185.0
Pb             8       183.0
H              3       181.0
Au             6       180.0
Hg             3       176.0
Nb             4       172.0

## Properties that go with high error (Spearman)
                                      property  spearman  p_value
      feature: comp__MagpieData minimum Number    -0.380      0.0
         feature: comp__MagpieData mode Number    -0.380      0.0
feature: comp__MagpieData minimum AtomicWeight    -0.379      0.0
   feature: comp__MagpieData mode AtomicWeight    -0.379      0.0
                 feature: thermo__Tm_asymmetry     0.353      0.0
           feature: phys__melting_point__ratio     0.353      0.0
                       feature: phys__Z__ratio     0.351      0.0
            feature: comp__MagpieData mode Row    -0.351      0.0
         feature: comp__MagpieData minimum Row    -0.351      0.0
             feature: phys__atomic_mass__ratio     0.344      0.0
        feature: thermo__Miedema_deltaH_ss_min     0.335      0.0
            feature: comp__MagpieData mean Row    -0.332      0.0
      feature: comp__MagpieData range MeltingT     0.326      0.0
    feature: comp__MagpieData avg_dev MeltingT     0.326      0.0
        feature: phys__melting_point__abs_diff     0.322      0.0

## Worst 15 systems
 pair                     topology           predicted_topology  liquidus_MAE_K  liquidus_bias_K   kind  nn_distance
 Cd_V                   monotectic                      complex          1993.2          -1993.2 offset          6.6
 In_V              simple_eutectic                      complex           806.5           -806.5 offset          5.4
 Er_V                      complex                   monotectic           655.7           -627.5  shape          3.5
 S_Zn                   monotectic eutectic_with_intermetallics           563.1           -561.6 offset         10.2
Cr_Pb              simple_eutectic              simple_eutectic           511.9           -462.3  shape          9.0
In_Nb                      complex                      complex           478.1            477.9  shape          7.2
Au_Co              simple_eutectic              simple_eutectic           394.0            394.0 offset          8.0
 Ir_P eutectic_with_intermetallics eutectic_with_intermetallics           381.8            381.4 offset         10.8
Au_Ni              simple_eutectic              simple_eutectic           379.2            379.2 offset          7.0
Ba_Be eutectic_with_intermetallics                   monotectic           331.0           -314.0 offset         13.2
Li_Rh eutectic_with_intermetallics eutectic_with_intermetallics           325.9           -300.9  shape         11.9
 Hg_S                   monotectic                   monotectic           320.9           -320.9 offset         11.3
Ni_Tl                   monotectic                   monotectic           320.0           -319.3 offset          8.1
Cr_Eu                   monotectic                   monotectic           288.9           -288.7  edges          4.9
Mn_Se                   monotectic                   monotectic           285.3            241.5  shape          7.6
