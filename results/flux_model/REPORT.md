# Flux model (results/flux_model)

## Identifiable structure
14 free / 23 fixed. Structural zeros of the rate laws: 0. Fisher information of the 14 analysed free parameters: full rank True (smallest singular value ratio 1.17e-04), max |r| 0.897. Kept free by design (not identifiable): none.
```
           ident_class      value  sensitivity  cv_percent  ci_ratio_97.5_over_2.5  practically_identifiable profile_class
param                                                                                                                     
v_max_1           free    80.2561       0.0000  13005.4882                  1.3827                     False     one_sided
Ks_g6p_pgi        free     0.0050       0.0000  28818.0085               3731.3802                     False          flat
kcat_f_2          free   978.1748       0.0023    226.0584                  2.1669                     False     one_sided
kcat_f_3          free   974.4076       0.0986     11.8237                  1.4294                     False     one_sided
Ks_fbp_4          free     0.0010       0.0092    139.5108                  1.0191                     False     one_sided
kcat_f_4          free     2.6973       0.1754      5.6898                  1.3349                      True  identifiable
kcat_f_5          free  6865.9168       0.0005  11496.4799                  2.4938                     False          flat
kcat_f_6          free   622.6176       0.0002  24721.9831                  4.5869                      True     one_sided
kcat_f_7          free   875.8411       0.0016   3254.8590                  1.4509                     False     one_sided
Ks_adp_7          free     0.0147       0.0004   9351.9710                289.5988                     False          flat
kcat_f_8          free   651.6838       0.0018   6193.1373                  3.7663                      True     one_sided
Ks_3pg_8          free     0.1005       0.0011  15095.4289               1449.7295                     False          flat
kcat_f_9          free     6.7110       0.0786     57.6544                126.8115                     False  identifiable
Ks_2pg_9          free     0.0010       0.0043   2271.2778                233.4808                     False     one_sided
```

## Fixing path
```
 iter      stage            rule      param action  partner       r  loading  obj_after
    0 structural structural_zero   Ks_nad_6  fixed      NaN     NaN  -0.5448    10.9087
    0 structural structural_zero   Ks_pgp_7  fixed      NaN     NaN  -0.6845    10.9087
    0 structural structural_zero   Kp_pgp_6  fixed      NaN     NaN   0.7066    10.9087
    0 structural structural_zero   Ks_3pg_7  fixed      NaN     NaN   0.7010    10.9087
    0 structural structural_zero   Ks_g3p_6  fixed      NaN     NaN   0.6604    10.9087
    0 structural structural_zero   Kp_fbp_3  fixed      NaN     NaN  -0.6172    10.9087
    0 structural structural_zero   Kp_g3p_4  fixed      NaN     NaN   0.7071    10.9087
    0 structural structural_zero   Ks_f6p_3  fixed      NaN     NaN   0.7071    10.9087
    1  practical             svd  Kp_nadh_6  fixed      NaN     NaN  -1.0000    10.9087
    2  practical             svd   Kp_g3p_5  fixed      NaN     NaN  -1.0000    10.9087
    3  practical             svd      Ka1_1  fixed      NaN     NaN  -0.5884    10.9087
    4  practical             svd    K_g6p_1  fixed      NaN     NaN  -1.0000    10.9087
    5  practical             svd   Ks_atp_7  fixed      NaN     NaN   0.9997    10.9087
    6  practical             svd      Ka3_1  fixed      NaN     NaN  -0.7779    10.9087
    7  practical             svd Kp_f6p_pgi  fixed      NaN     NaN  -1.0000    10.9087
    8  practical             svd  Kp_dhap_4  fixed      NaN     NaN  -0.9998    10.9087
    9  practical             svd   Ks_2pg_8  fixed      NaN     NaN   0.9999    10.9087
   10  practical             svd    Ks_pi_6  fixed      NaN     NaN  -0.9999    10.9087
   11  practical             svd   Ks_pep_9  fixed      NaN     NaN  -0.9997    10.9087
   12  practical             svd      Ka2_1  fixed      NaN     NaN  -0.9610    10.9087
   13  practical             svd  Ks_dhap_5  fixed      NaN     NaN  -0.6640    10.9087
   14  practical             svd   Ks_atp_3  fixed      NaN     NaN  -0.7103    10.9087
   15  practical     correlation   Kp_adp_3  fixed kcat_f_3 -0.9996      NaN    10.9087
```

## Bootstrap
5000/5000 resamples converged; objective median 10.289 (95% interval 2.669-23.891; estimate 10.909). Practically identifiable: kcat_f_4, kcat_f_6, kcat_f_8. Objective = 3.53 + 6.01 n_KO18 + 1.14 n_KO05 (R2 0.98).

## Fit
Objective 10.9087 (misfit s2 2.41). Metabolites RMSE 0.114 (mean predictor 0.158, R2 0.477); LI fluxes RMSE 0.359 (mean predictor 0.727, R2 0.756). Parameters on a bound: Ks_fbp_4.

## Leave-one-condition-out cross-validation
Metabolites RMSE 0.206 (training-mean predictor 0.166); LI fluxes RMSE 1.404 (training-mean predictor 0.762). Folds failed: 0; closure solves failed: 3.

