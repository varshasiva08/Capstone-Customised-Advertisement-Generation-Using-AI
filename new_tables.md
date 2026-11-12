### Table A. Experimental setup

- Units (profile x repeat): **48** from **24** distinct profiles; arms: vanilla, resample, cpdc; up to 4 generator calls per unit; attempt 0 is shared by the three arms.
- Images rated by one human rater: **145** (all unique attempt images).
- Units per target ethnicity:

| ethnicity | profiles | units |
|---|---|---|
| African American | 4 | 12 |
| East Asian | 4 | 12 |
| Latin American | 4 | 4 |
| Middle Eastern | 4 | 4 |
| South Asian | 4 | 12 |
| White | 4 | 4 |

### Table B. Arm comparison graded by the DeepFace verifier (optimistic: the verifier also chose the final image)

| arm | images/unit | ethnicity | gender | age | all axes |
|---|---|---|---|---|---|
| vanilla | 1.0 | 72.9% | 91.7% | 72.9% | 56.2% |
| resample | 2.06 | 87.5% | 100.0% | 87.5% | 75.0% |
| cpdc | 1.96 | 83.3% | 97.9% | 89.6% | 72.9% |

### Table C. The same final images graded by three different graders (n = 48 units per arm)

| arm | images/unit | DeepFace: all axes | Human: all axes (age +/-10y) | DeepFace: eth+gender | CLIP: eth+gender | Human: eth+gender |
|---|---|---|---|---|---|---|
| vanilla | 1.0 | 56.2% | 79.2% | 68.8% | 83.3% | 85.4% |
| resample | 2.06 | 75.0% | 81.2% | 87.5% | 87.5% | 89.6% |
| cpdc | 1.96 | 72.9% | 87.5% | 81.2% | 85.4% | 91.7% |

Differences in percentage points with 95% CI (bootstrap over profiles):

| contrast | DeepFace all axes | Human all axes | CLIP eth+gender | Human eth+gender |
|---|---|---|---|---|
| resample - vanilla | +18.8 [+8.3, +30.8] | +2.1 [-4.8, +10.0] | +4.2 [+0.0, +10.9] | +4.2 [-3.8, +12.5] |
| cpdc - vanilla | +16.7 [+8.3, +26.0] | +8.3 [+2.0, +16.7] | +2.1 [+0.0, +6.5] | +6.2 [+0.0, +13.6] |
| cpdc - resample | -2.1 [-9.6, +4.3] | +6.2 [+0.0, +13.6] | -2.1 [-7.4, +0.0] | +2.1 [+0.0, +7.1] |

Units where CPDC differs from resample on human all-axes: better in **3**, worse in **0**, tied in **45** of 48.

### Table D. Ethnicity verifiers against the human label, by target group

| target | n images | human: reads as target | DeepFace passes | CLIP passes | DeepFace agrees with human | CLIP agrees with human |
|---|---|---|---|---|---|---|
| African American | 31 | 100% | 97% | 100% | 97% | 100% |
| East Asian | 22 | 95% | 100% | 95% | 95% | 91% |
| Latin American | 10 | 20% | 40% | 30% | 80% | 90% |
| Middle Eastern | 19 | 37% | 37% | 58% | 58% | 79% |
| South Asian | 56 | 80% | 20% | 93% | 36% | 80% |
| White | 7 | 100% | 100% | 29% | 100% | 29% |
| All | 145 | 78% | 56% | 83% | 67% | 84% |

Reference: answering "matches" for every image would agree with the human on 77.9% of images. Cohen's kappa with the human: DeepFace 0.29, CLIP 0.50.

### Table D2. Gender and age axes against the human label

| axis | n images | human: matches | DeepFace passes | CLIP matches | human: within 10 years |
|---|---|---|---|---|---|
| gender (target: female) | 89 | 99% | 76% | 100% |  |
| gender (target: male) | 56 | 100% | 100% | 100% |  |
| gender (target: all) | 145 | 99% | 86% | 100% |  |
| age (target: 30s) | 47 |  | 89% | n/a | 98% |
| age (target: 50s) | 98 |  | 51% | n/a | 95% |
| age (target: all) | 145 |  | 63% | n/a | 96% |

Exact-decade age agreement between the human and the target: 44% (within one decade: 96%). CLIP has no age output.

### Table E. South Asian target images: can the verifiers tell good from bad? (n = 56; human-yes 45, human-no 11)

| verifier | passes of human-yes | passes of human-no | AUC of P(target) | kappa |
|---|---|---|---|---|
| DeepFace (lenient South Asian rule) | 10/45 | 1/11 | 0.54 | 0.06 |
| CLIP ViT-L/14 zero-shot | 43/45 | 9/11 | 0.46 | 0.18 |

Human-no South Asian images by profile id: p000 (1), p001 (5), p003 (5) (profiles are 30s/50s female/male South Asian).

### Table F. Can a combined verifier beat the raw ones? (ethnicity axis, leave-one-profile-out)

| verifier | accuracy (all 145) | accuracy (South Asian) | AUC (all) | AUC (South Asian) |
|---|---|---|---|---|
| always "matches" | 77.9% | 80.4% | 0.50 | 0.50 |
| DeepFace pass/fail | 66.9% | 35.7% | 0.70 | 0.57 |
| CLIP top class = target | 84.1% | 80.4% | 0.79 | 0.46 |
| logistic: CLIP P(target) (leave-one-profile-out) | 79.3% | 78.6% | 0.70 | 0.44 |
| logistic: CLIP + DeepFace (leave-one-profile-out) | 84.8% | 78.6% | 0.82 | 0.52 |

### Table G. Cost per success (all axes; final selected image)

| arm | images generated | DeepFace successes | images per DeepFace success | human successes | images per human success |
|---|---|---|---|---|---|
| vanilla | 48 | 27 | 1.78 | 38 | 1.26 |
| resample | 99 | 36 | 2.75 | 39 | 2.54 |
| cpdc | 94 | 35 | 2.69 | 42 | 2.24 |

### Table H. Recovery of units whose first image failed the DeepFace verifier

| group | arm | units failing at attempt 0 (DeepFace) | fixed by the end (DeepFace) | judged OK by human at the end | mean images |
|---|---|---|---|---|---|
| South Asian targets | vanilla | 9 | 0 (0%) | 6 (67%) | 1.00 |
| South Asian targets | resample | 9 | 3 (33%) | 7 (78%) | 3.67 |
| South Asian targets | cpdc | 9 | 3 (33%) | 8 (89%) | 3.22 |
| other targets | vanilla | 12 | 0 (0%) | 8 (67%) | 1.00 |
| other targets | resample | 12 | 6 (50%) | 8 (67%) | 3.25 |
| other targets | cpdc | 12 | 5 (42%) | 10 (83%) | 3.17 |

### Table I. Does ethnicity wording help? Retry images in units whose attempt-0 image failed DeepFace's ethnicity check

| retry type | retry images | ethnicity passes (DeepFace) | human says ethnicity matches |
|---|---|---|---|
| cpdc, ethnicity wording applied | 21 | 3 (14%) | 14 (67%) |
| cpdc, other axis wording | 9 | 3 (33%) | 4 (44%) |
| resample (no wording change) | 35 | 9 (26%) | 23 (66%) |

DeepFace pass rate of the image produced in each CPDC correction round, by the axis that was corrected:

| correction axis (cpdc rounds) | rounds | ethnicity pass | gender pass | age pass |
|---|---|---|---|---|
| age | 18 | 78% | 56% | 50% |
| ethnicity | 21 | 14% | 100% | 86% |
| gender | 7 | 71% | 57% | 43% |
