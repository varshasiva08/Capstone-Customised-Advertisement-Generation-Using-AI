# Final analysis (80 units, 52 profiles, 272 rated images)

## Rater agreement

| label | agreement_% | kappa | kappa_95CI_over_profiles |
|---|---|---|---|
| gender (raw labels) | 92.3 | 0.83 | [0.74, 0.91] |
| ethnicity (8 labels) | 56.2 | 0.47 | [0.33, 0.60] |
| age, exact decade | 47.8 | 0.35 | [0.27, 0.41] |
| age, within 10 years | 90.1 |  |  |

## Target match and agreement by target group

| target | images | profiles | rater1_match_% | rater2_match_% | raters_agree_% | kappa_match |
|---|---|---|---|---|---|---|
| African American | 31 | 4 | 100.0 | 100.0 | 100.0 | nan |
| East Asian | 22 | 4 | 95.5 | 100.0 | 95.5 | 0.0 |
| Latin American | 42 | 12 | 16.7 | 31.0 | 52.4 | -0.28 |
| Middle Eastern | 60 | 12 | 43.3 | 50.0 | 83.3 | 0.67 |
| South Asian | 90 | 8 | 75.6 | 64.4 | 60.0 | 0.06 |
| White | 27 | 12 | 88.9 | 55.6 | 66.7 | 0.27 |
| ALL | 272 | 52 | 65.1 | 62.1 | 72.1 | 0.4 |

## Gender tie-break

21 images where raters 1 and 2 disagreed on gender; majority (rater 3) equals the target on 21 of 21.

| image | target | rater1 | rater2 | rater3 | majority |
|---|---|---|---|---|---|
| main_r0000 | male | male | female | male | male |
| main_r0010 | male | male | female | male | male |
| main_r0016 | male | male | female | male | male |
| main_r0020 | male | male | female | male | male |
| main_r0035 | male | male | female | male | male |
| main_r0051 | male | male | female | male | male |
| main_r0052 | male | male | female | male | male |
| main_r0056 | male | male | female | male | male |
| main_r0076 | male | male | female | male | male |
| main_r0087 | male | male | female | male | male |
| main_r0091 | male | male | female | male | male |
| main_r0093 | male | male | female | male | male |
| main_r0118 | male | male | female | male | male |
| main_r0142 | male | male | female | male | male |
| main_r0144 | male | male | female | male | male |
| extra1_r0060 | male | female | male | male | male |
| extra1_r0054 | male | male | female | male | male |
| extra1_r0000 | male | female | male | male | male |
| extra1_r0030 | male | female | male | male | male |
| extra1_r0071 | male | female | male | male | male |
| extra2_r0009 | male | female | male | male | male |

## All-axis success on final images, % (units)

| grader | vanilla | resample | cpdc |
|---|---|---|---|
| DeepFace | 50.0 | 72.5 | 71.2 |
| Rater 1 | 65.0 | 68.8 | 71.2 |
| Rater 2 | 57.5 | 56.2 | 61.3 |
| Both agree | 48.8 | 47.5 | 52.5 |
| mean generator calls | 1.0 | 2.28 | 2.12 |

## Contrasts, percentage points, 95% CI over profiles

| grader | contrast | diff_pp | ci_low | ci_high | profiles | excludes_zero |
|---|---|---|---|---|---|---|
| DeepFace | cpdc - vanilla | 22.1 | 12.5 | 32.7 | 52 | yes |
| DeepFace | cpdc - resample | -4.8 | -15.4 | 5.1 | 52 | no |
| DeepFace | resample - vanilla | 26.9 | 16.0 | 39.1 | 52 | yes |
| Rater 1 | cpdc - vanilla | 6.7 | 0.6 | 14.1 | 52 | yes |
| Rater 1 | cpdc - resample | 3.2 | -4.5 | 11.9 | 52 | no |
| Rater 1 | resample - vanilla | 3.5 | -5.1 | 12.2 | 52 | no |
| Rater 2 | cpdc - vanilla | 4.5 | -6.7 | 15.7 | 52 | no |
| Rater 2 | cpdc - resample | 4.5 | -7.7 | 16.7 | 52 | no |
| Rater 2 | resample - vanilla | -0.0 | -10.6 | 10.6 | 52 | no |
| Both agree | cpdc - vanilla | 2.9 | -6.1 | 11.9 | 52 | no |
| Both agree | cpdc - resample | 3.8 | -6.4 | 14.1 | 52 | no |
| Both agree | resample - vanilla | -1.0 | -9.0 | 7.1 | 52 | no |

## Unit-level discordance (exact McNemar)

| grader | comparison | cpdc_only | other_only | mcnemar_p |
|---|---|---|---|---|
| DeepFace | cpdc vs resample | 6 | 7 | 1.0 |
| DeepFace | cpdc vs vanilla | 17 | 0 | 0.0 |
| Rater 1 | cpdc vs resample | 5 | 3 | 0.727 |
| Rater 1 | cpdc vs vanilla | 8 | 3 | 0.227 |
| Rater 2 | cpdc vs resample | 12 | 8 | 0.503 |
| Rater 2 | cpdc vs vanilla | 9 | 6 | 0.607 |
| Both agree | cpdc vs resample | 9 | 5 | 0.424 |
| Both agree | cpdc vs vanilla | 9 | 6 | 0.607 |

## By batch, % all axes

| batch | df_all cpdc | df_all resample | df_all vanilla | all1 cpdc | all1 resample | all1 vanilla | all2 cpdc | all2 resample | all2 vanilla | both cpdc | both resample | both vanilla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| extra1 | 66.7 | 79.2 | 41.7 | 50.0 | 45.8 | 41.7 | 29.2 | 25.0 | 20.8 | 20.8 | 16.7 | 16.7 |
| extra2 | 75.0 | 37.5 | 37.5 | 37.5 | 62.5 | 50.0 | 87.5 | 62.5 | 87.5 | 37.5 | 37.5 | 50.0 |
| main | 72.9 | 75.0 | 56.2 | 87.5 | 81.2 | 79.2 | 72.9 | 70.8 | 70.8 | 70.8 | 64.6 | 64.6 |

## Sensitivity: dropping suspected default-click rows

| grader | contrast | diff_pp | ci_low | ci_high | profiles | excludes_zero |
|---|---|---|---|---|---|---|
| Rater 1 (drop 43 suspect final images) | cpdc - vanilla | 6.4 | 0.0 | 14.4 | 44 | no |
| Rater 1 (drop 43 suspect final images) | cpdc - resample | 5.2 | -1.9 | 13.3 | 45 | no |
| Rater 1 (drop 43 suspect final images) | resample - vanilla | 1.1 | -7.8 | 10.0 | 45 | no |
| Rater 2 (drop 18 suspect final images) | cpdc - vanilla | 2.4 | -9.0 | 13.9 | 48 | no |
| Rater 2 (drop 18 suspect final images) | cpdc - resample | 2.4 | -9.9 | 14.6 | 49 | no |
| Rater 2 (drop 18 suspect final images) | resample - vanilla | -2.1 | -12.8 | 8.3 | 48 | no |

## DeepFace ethnicity verdict vs each rater

| target | images | DeepFace_pass_% | r1_match_% | r1_agree_% | r1_kappa | r2_match_% | r2_agree_% | r2_kappa |
|---|---|---|---|---|---|---|---|---|
| African American | 31 | 96.8 | 100.0 | 96.8 | 0.0 | 100.0 | 96.8 | 0.0 |
| East Asian | 22 | 100.0 | 95.5 | 95.5 | 0.0 | 100.0 | 100.0 | nan |
| Latin American | 42 | 50.0 | 16.7 | 52.4 | 0.05 | 31.0 | 42.9 | -0.14 |
| Middle Eastern | 60 | 28.3 | 43.3 | 58.3 | 0.12 | 50.0 | 55.0 | 0.1 |
| South Asian | 90 | 23.3 | 75.6 | 43.3 | 0.11 | 64.4 | 45.6 | 0.06 |
| White | 27 | 85.2 | 88.9 | 88.9 | 0.51 | 55.6 | 63.0 | 0.2 |
| ALL | 272 | 49.3 | 65.1 | 62.9 | 0.26 | 62.1 | 59.2 | 0.19 |

* South Asian, DeepFace vs rater 1: kappa 0.11 [0.03, 0.26] over 8 profiles; DeepFace passes 19/68 human-accepted and 2/22 human-rejected images
* South Asian, DeepFace vs rater 2: kappa 0.06 [-0.04, 0.29] over 8 profiles; DeepFace passes 15/58 human-accepted and 6/32 human-rejected images
* human-human kappa on 'matches target' overall: 0.40; DeepFace-human: 0.26 (r1), 0.19 (r2)
