# Flux model, repaired pipeline (results_v4/flux_model)

## Identifiability first
Final structure: 11 free / 26 fixed, reduction status `converged`; final point = re_estimated (obj 10.9429; reduction anchor 10.9203; acceptance limit 10.9639).
Practical criteria on the Schur complement (4 kept_free_by_rule parameters as nuisance): rank 7/7 at eigenvalue cut 1e-08 (singular ratio 1e-4), max |r| 0.833, no |r|>0.9 pair. Output-level classes (bound-free / active set): {'identifiable': 7, 'bound_identified': 1, 'non_identifiable': 3}.
Flags: {'K_g6p_1': ['fd_unstable'], 'Ks_fbp_4': ['at_bound'], 'kcat_f_5': ['fd_unstable'], 'kcat_f_6': ['fd_unstable']}.
```
     param       ident_class     output_class       flags     value  influence  cv_percent             cv_basis  bootstrap_cv_%  boot_ci_ratio profile_class
   K_g6p_1 kept_free_by_rule non_identifiable fd_unstable   10.0011     0.0033   2625.8165 nuisance_conditional          0.0060         1.0002          flat
Ks_g6p_pgi      identifiable     identifiable         NaN    0.1972     0.0150    854.2581                schur        170.4527      3016.0330          flat
  kcat_f_2      identifiable     identifiable         NaN  917.4236     0.0586    212.1389                schur         17.0263         2.0154     one_sided
  kcat_f_3 kept_free_by_rule bound_identified         NaN  960.0879     0.6284     11.2956 nuisance_conditional          8.6154         1.3471     one_sided
  Ks_fbp_4           flagged     identifiable    at_bound    0.0010     0.0834    139.2210                schur       1080.5528         1.0789     one_sided
  kcat_f_4      identifiable     identifiable         NaN    2.6996     2.0170      5.6846                schur        649.8026         1.2673  identifiable
  kcat_f_5           flagged     identifiable fd_unstable 5578.9209     0.0048   2625.4333                schur         17.5673         1.7377          flat
  kcat_f_6 kept_free_by_rule non_identifiable fd_unstable  537.7572     0.0062   1356.2411 nuisance_conditional         18.1508         1.7644          flat
  kcat_f_7 kept_free_by_rule non_identifiable         NaN  720.5240     0.0089    858.6368 nuisance_conditional         14.3610         1.7091          flat
  kcat_f_8      identifiable     identifiable         NaN  561.6466     0.0086   1118.1235                schur         18.4400         2.2895     one_sided
  kcat_f_9      identifiable     identifiable         NaN    6.8908     0.4212     20.5148                schur        384.9436        93.0074  identifiable
```

kaug covariance: PSD True, 0 negative variances (final/covariance_crosscheck.csv). It disagrees with the Schur-FIM SDs (ratios 0.2-4, 3.5e4 for K_g6p_1), so CVs are indicative only.

### Which rule fixed which parameter
```
     param  iter                rule  influence_at_fix     class_at_fix
  Ks_atp_3     0    structural_exact           0.68654 non_identifiable
  Kp_adp_3     0    structural_exact           0.68633 non_identifiable
  Kp_g3p_4     0    structural_exact           0.00000 non_identifiable
  Ks_g3p_6     0    structural_exact           0.00060 non_identifiable
  Ks_nad_6     0    structural_exact           0.00060 non_identifiable
 Kp_nadh_6     0    structural_exact           0.00000 non_identifiable
  Ks_pgp_7     0    structural_exact           0.00278 non_identifiable
  Ks_3pg_7     0    structural_exact           0.00002 non_identifiable
  Kp_pgp_6     0 structural_pointdep           0.00000 non_identifiable
   v_max_1     0   structural_output           0.00199 non_identifiable
     Ka1_1     0   structural_output           0.00069 non_identifiable
     Ka2_1     0   structural_output           0.00048 non_identifiable
     Ka3_1     0   structural_output           0.00075 non_identifiable
  Ks_f6p_3     0   structural_output           0.61506 non_identifiable
  Kp_fbp_3     0   structural_output           0.61435 non_identifiable
  Kp_g3p_5     0   structural_output           0.00000 non_identifiable
   Ks_pi_6     0   structural_output           0.00035 non_identifiable
  Ks_adp_7     0   structural_output           0.00434 non_identifiable
  Ks_atp_7     0   structural_output           0.00002 non_identifiable
 Kp_dhap_4     1                 svd           0.00000                 
  Ks_pep_9     2                 svd           0.00011                 
Kp_f6p_pgi     3                 svd           0.00006                 
  Ks_2pg_8     4                 svd           0.00003                 
 Ks_dhap_5     5                 svd           0.00082                 
  Ks_3pg_8     6                corr           0.00321                 
  Ks_2pg_9     7                flag           0.01020                 
```
Output-level fixes: all 10 were `non_identifiable` (null also with the active set), none only bound_identified, at the iteration-0 round-2 anchor. kcat_f_3 is `bound_identified` at the final point (kept_free_by_rule).

## Bootstrap (N=1000, case resampling, seed 0, warm start at the final point)
1000/1000 converged (10 needed the monotone retry). Objective: median 10.49, 95% [2.93, 23.94]; obj ~ 2.96 + 6.68 n_KO18 + 1.43 n_KO05 (R2 0.49); KO18 still dominates (6.68 per copy; 6.68 in results_v3/flux_model).
Caveat: for parameters the data cannot move (kcat_f_5/6/7/8, K_g6p_1) the bootstrap median equals the warm start to 7 digits: the narrow percentile interval reflects parking at the start, not information, so they are reported as not practically identifiable. They stay in the free set because the rules always keep them free (kcat_f_* and the last free parameter of each rate law carry the biology of the reaction), not because the data identify them. Only kcat_f_4 (and partly kcat_f_9) are credible.
```
     param    anchor    median      p2.5     p97.5  cv_percent  ci_ratio_97.5_over_2.5  frac_at_theta_bound  parked  practically_identifiable
   K_g6p_1   10.0011   10.0011   10.0005   10.0023      0.0060                  1.0002                0.000    True                     False
Ks_g6p_pgi    0.1972    0.1817    0.0014    4.1920    170.4527               3016.0330                0.000   False                     False
  kcat_f_2  917.4236  915.4729  495.9429  999.5072     17.0263                  2.0154                0.146   False                     False
  kcat_f_3  960.0879  959.4024  742.2871  999.9613      8.6154                  1.3471                0.221   False                     False
  Ks_fbp_4    0.0010    0.0010    0.0010    0.0011   1080.5528                  1.0789                0.970   False                     False
  kcat_f_4    2.6996    2.7118    2.4564    3.1129    649.8026                  1.2673                0.000   False                      True
  kcat_f_5 5578.9209 5578.9210 5266.6294 9151.7070     17.5673                  1.7377                0.000    True                     False
  kcat_f_6  537.7572  537.7573  512.0732  903.5211     18.1508                  1.7644                0.000    True                     False
  kcat_f_7  720.5240  720.5240  580.0862  991.4175     14.3610                  1.7091                0.038    True                     False
  kcat_f_8  561.6466  561.6466  383.1601  877.2501     18.4400                  2.2895                0.000    True                     False
  kcat_f_9    6.8908    6.9079    5.2324  486.6490    384.9436                 93.0074                0.000   False                     False
```

## Profiles (31 points, +/-0.5 decades, continuation, tight)
Thresholds 0.0873 (and 0.2081 misfit scaled). Re-fits start without duals, so profile points sit about 0.03 above the best objective: crossings are uncertain at the level of 1/3 of the threshold.
```
     param  upper_decades_lr  lower_decades_lr     class_lr  class_lr_s2  n_failed
   K_g6p_1               NaN               NaN         flat         flat         0
Ks_g6p_pgi               NaN               NaN         flat         flat         0
  kcat_f_2               NaN             0.169    one_sided    one_sided         0
  kcat_f_3               NaN             0.054    one_sided    one_sided         0
  Ks_fbp_4             0.233               NaN    one_sided         flat         0
  kcat_f_4             0.014             0.016 identifiable identifiable         0
  kcat_f_5               NaN               NaN         flat         flat         0
  kcat_f_6               NaN               NaN         flat         flat         0
  kcat_f_7               NaN               NaN         flat         flat         0
  kcat_f_8               NaN             0.481    one_sided         flat         0
  kcat_f_9             0.149             0.110 identifiable identifiable         0
```

## Fit
rmse_met 0.1151 (training-mean baseline 0.1576), LI flux rmse 0.3596 (baseline 0.7270), s2 2.38, drains at a bound 8/110. Stage0 (37 free): obj 11.745. Comparison with results_v3/flux_model (11 free): obj 10.9, s2 2.4, rmse_met 0.115, LI flux 0.360.

## LOOCV (22 folds, RF03 included)
Model is WORSE than the training-mean baselines: metabolites rmse 0.2035 vs 0.1659, LI flux 1.376 vs 0.762 (all folds); closure-success folds only: 0.2197 vs 0.1757 and 1.414 vs 0.823. Known limitation (pts_fraction closure ill-posed, S invertible).
Fold status: {'multistart_ok': 8, 'above_bound': 8, 'anchor_ok': 6}; the restricted anchor lies outside the training-only bounds in 11/22 folds. above_bound folds: KO03, KO04, KO05, KO15, KO16, KO19, KO20 have the anchor outside the training bounds (bound unattainable); KO10 has it inside and is above the bound by 0.007 (ipopt warm-start drift).
```
   h   fold_status  obj_fold   bound  anchor_inside_training_bounds  n_anchor_values_outside_bounds  closure_success
KO02 multistart_ok   11.4469 11.4503                           True                               0             True
KO03   above_bound   11.6823 11.4542                          False                               1             True
KO04   above_bound   12.1898 11.4570                          False                              32             True
KO05   above_bound   12.5765  9.9319                          False                               3             True
KO07 multistart_ok   11.3879 11.4113                           True                               0             True
KO08 multistart_ok   11.3910 11.4173                          False                               3            False
KO10   above_bound   11.4418 11.4344                           True                               0             True
KO11     anchor_ok   11.0636 11.0655                           True                               0             True
KO12 multistart_ok   11.4145 11.4382                           True                               0             True
KO13 multistart_ok   11.2818 11.3036                           True                               0             True
KO14     anchor_ok   11.0694 11.0832                           True                               0            False
KO15   above_bound   11.2467 11.2043                          False                              15            False
KO16   above_bound   11.3864 11.3779                          False                              12             True
KO17     anchor_ok   11.0148 11.0192                          False                               6             True
KO18     anchor_ok    4.9893  5.0199                          False                              11             True
KO19   above_bound   11.8035 11.2673                          False                               9             True
KO20   above_bound   12.2306 10.6830                          False                              21             True
KO21 multistart_ok   11.3898 11.3955                          False                               2             True
KO22 multistart_ok   11.3689 11.3883                           True                               0            False
KO23 multistart_ok   11.3109 11.3278                           True                               0             True
KO24     anchor_ok   11.1273 11.1665                           True                               0             True
RF03     anchor_ok   11.2853 11.3013                           True                               0             True
```

## Known limitations
closure ill-posed (S invertible, pts_fraction closure); PGI thermodynamic floor (the data violate the PGI driving force in 6 conditions, reported only); isozymes ignored; cofactor bounds (Pi literature range); ipopt warm-start drift of about 0.3% (kept_anchor semantics, no duals); profile and bootstrap behaviour of parameters at a bound (Ks_fbp_4 at its lower bound).

Figures: figures/fig2_composite.png, figures/fixing_path.png, figures/landscape.png, figures/paper/fig1_workflow.png, figures/paper/fig2_characterization.png, figures/paper/figS_bootstrap_all.png, figures/paper/figS_parameter_evolution.png, figures/paper/figS_parity_stages.png, figures/paper/figS_profiles_all.png, figures/pipeline_diagram.png.
