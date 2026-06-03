# NFL Draft Prediction — Competition Progress Log

## Overview

**Goal:** Predict the probability that a college football player gets drafted into the NFL, based on physical performance test results (NFL Combine stats) and player info.

**Metric:** ROC AUC (Area Under the ROC Curve)
- Score ranges from 0.5 (random guessing) to 1.0 (perfect)
- A score of 0.83 means the model correctly ranks a drafted player above an undrafted player 83% of the time

**Deadline:** June 12th 11AM UTC

---

## Scores Summary

| Version | Notebook | Public Score | Key Change |
|---|---|---|---|
| Baseline | `baseline.ipynb` | 0.80792 | RandomForest, raw features only |
| V1 | `improved_submissionV1.ipynb` | 0.83403 | LightGBM + feature engineering |
| V2 | `improved_submissionV2.ipynb` | 0.8152 ❌ | More features — caused data leakage |
| V3 | `improved_submissionV3.ipynb` | 0.82228 | Removed leaky features |
| V4 | `improved_submissionV4.ipynb` | 0.83364 | V1 features + Optuna tuning |
| V5 | `improved_submissionV5.ipynb` | 0.83382 | V3 features + Optuna tuning |
| V6 | `improved_submissionV6.ipynb` | **0.83523** ← best | V1+V3 features + Optuna + 10-fold |
| V6 ensemble | `submission_V6_ensemble.csv` | TBD | Average of V1 + V4 + V6 |

---

## Baseline — `baseline.ipynb`

**Public Score: 0.80792**

### What it does
- Loads `train.csv` and `test.csv`
- Encodes categorical columns (`Player_Type`, `Position_Type`, `Position`) with `LabelEncoder`
  - This converts text categories like "offense" or "WR" into numbers so the model can use them
- Trains a **RandomForestClassifier** with `StratifiedKFold` (5 folds)
  - StratifiedKFold ensures each fold has the same ratio of drafted/undrafted players
- Evaluates with **ROC AUC**
- Saves `submission.csv`

### Features used
- Raw numeric: `Year`, `Age`, `Height`, `Weight`, `Sprint_40yd`, `Vertical_Jump`, `Bench_Press_Reps`, `Broad_Jump`, `Agility_3cone`, `Shuttle`
- Encoded categoricals: `Player_Type`, `Position_Type`, `Position`

### Limitations
- **RandomForest is weaker than gradient boosting** for tabular data — it doesn't handle missing values well and is generally less accurate
- **No handling of missing values** — many players skip certain drills (e.g. kickers rarely do the bench press). The model treats missing values as zeros, which is misleading
- **No feature engineering** — the raw numbers are used as-is, without any transformation that might reveal more signal

---

## V1 — `improved_submissionV1.ipynb`

**Public Score: 0.83403** ✅ Best so far (+0.026 from baseline)

### What changed from Baseline

#### 1. Model: RandomForest → LightGBM
LightGBM is a gradient boosting framework that is generally much stronger than RandomForest on tabular data. Key advantages:
- Handles missing values natively (no need to impute)
- Learns complex non-linear patterns more efficiently
- Uses early stopping: training stops automatically when the validation AUC stops improving, preventing overfitting

#### 2. New Feature: BMI
`BMI = Weight / Height²`

Body Mass Index captures the relationship between a player's size and weight. A player who is very heavy for their height might be a powerful lineman, while a lean player might be a fast receiver. This single number summarizes body composition in a way that raw height and weight separately cannot.

#### 3. New Features: Missing-value flags
For each performance column (Sprint, Vertical, Bench, etc.), we create a binary flag:
- `Sprint_40yd_missing = 1` if the player did NOT run the 40-yard dash
- `Sprint_40yd_missing = 0` if they did

**Why this helps:** Skipping a drill is not random — it's often a strategic decision. For example, kickers and punters almost never do the bench press. Offensive linemen rarely skip the bench press. The pattern of which drills a player skips tells the model something about their position and likelihood of being drafted.

We also create `n_missing` = total number of drills skipped, which gives the model a summary of how incomplete a player's testing profile is.

#### 4. New Features: Position-relative z-scores
For each performance column, we compute how the player compares to **other players at the same position**:

`Sprint_40yd_pos_zscore = (player_sprint - mean_sprint_for_position) / std_sprint_for_position`

**Why this helps:** A 4.4-second 40-yard dash means something completely different for a wide receiver (fast) vs an offensive lineman (extremely fast). Without this normalization, the model can't distinguish between "fast for a WR" and "fast for an OL". By computing z-scores within each position group, we give the model a fair comparison.

### Why it improved
The combination of a stronger model (LightGBM) and more informative features (BMI, missing flags, position z-scores) gave the model much better signal to work with. The jump from 0.80792 to 0.83403 (+0.026) is significant.

### Output
- `submission_V1.csv`

---

## V2 — `improved_submissionV2.ipynb`

**Public Score: 0.8152** ❌ Score dropped (-0.019 from V1)

### What changed from V1
Added 5 new features on top of V1:

#### 1. Position percentile rank
`Sprint_40yd_pos_rank` = what percentile is this player's sprint time within their position group (e.g. top 10%?)

#### 2. Speed score
`speed_score = 1/Sprint + 1/Shuttle + 1/Agility`

Combines three speed-related drills into a single composite score. Lower times = faster = better, so we invert them before summing.

#### 3. School target encoding
`school_draft_rate` = the historical draft rate of players from that school (e.g. Alabama players get drafted 70% of the time)

#### 4. Age relative to position
`age_pos_zscore` = how old/young is this player compared to others at the same position

#### 5. Interaction features
- `height_x_weight` = Height × Weight (captures overall body size)
- `sprint_x_vertical` = Sprint × Vertical Jump (captures explosive athleticism)
- `sprint_x_broad` = Sprint × Broad Jump (captures speed + power)

### Why it got worse — Data Leakage

**Data leakage** means the model accidentally "sees" information it shouldn't have during training, making it appear to perform well on training data but fail on new data.

Two features caused leakage:

**`school_draft_rate`:** This was computed using the `Drafted` column (the target variable) from the training data, then applied to both train and test. The model learned "Alabama players get drafted a lot" — but this is circular reasoning. It's using the answer to predict the answer. On new data (the test set), this pattern doesn't generalize because the test set has different players.

**`{col}_pos_rank`:** This was computed on the full dataset (train + test combined). This means the model saw information from the test set during training, which is not allowed.

**The warning sign:** The OOF AUC was 0.84351 (looked great!) but the public score was 0.8152 (much worse). When OOF AUC is significantly higher than public score, it almost always means data leakage.

### Output
- `submission_V2.csv`

---

## V3 — `improved_submissionV3.ipynb`

**Public Score: 0.82228** ↑ Recovering from V2 (+0.007 from V2)

### What changed from V2

**Removed the two leaky features:**
- ❌ `school_draft_rate` — removed because it uses the target variable to create a feature (circular)
- ❌ `{col}_pos_rank` — removed because it was computed on the full dataset including test data

**Kept the safe features from V2:**
- ✅ `speed_score` — safe because it only uses raw performance numbers, no labels
- ✅ `age_pos_zscore` — safe because it only uses age and position, no labels
- ✅ `height_x_weight`, `sprint_x_vertical`, `sprint_x_broad` — safe interactions

Plus all V1 features (BMI, missing flags, z-scores).

### Why still below V1
The extra features (speed score, age z-score, interactions) are not adding enough signal to beat V1's clean baseline. The model has more features but they don't provide enough new information. The default LightGBM hyperparameters may also not be optimal for this larger feature set.

### Output
- `submission_V3.csv`

---

## V4 — `improved_submissionV4.ipynb`

**Public Score: 0.83364** (just below V1's 0.83403)

### What changed from V1
- **Same features as V1** (clean, no leakage)
- **Added Optuna hyperparameter tuning** (50 trials)

### What is Optuna?
Optuna is a hyperparameter optimization framework that uses **Bayesian optimization** to efficiently search for the best model settings. Instead of trying all combinations (grid search) or random combinations (random search), Optuna learns from previous trials to focus on the most promising parameter regions.

### Hyperparameters tuned and what they control

| Parameter | What it controls | Search Range |
|---|---|---|
| `learning_rate` | How fast the model learns — lower = more careful but slower | 0.01 – 0.1 |
| `num_leaves` | Complexity of each tree — more leaves = more complex model | 20 – 150 |
| `min_child_samples` | Minimum data points per leaf — higher = less overfitting | 10 – 50 |
| `feature_fraction` | % of features used per tree — adds randomness, reduces overfitting | 0.5 – 1.0 |
| `bagging_fraction` | % of data used per tree — adds randomness, reduces overfitting | 0.5 – 1.0 |
| `bagging_freq` | How often to apply bagging | 1 – 10 |
| `lambda_l1` | L1 regularization — penalizes large weights, reduces overfitting | 1e-8 – 10.0 |
| `lambda_l2` | L2 regularization — penalizes large weights, reduces overfitting | 1e-8 – 10.0 |

### Results
- OOF AUC: **0.83299** (fold scores: 0.79913, 0.86780, 0.87685, 0.79372, 0.83580)
- Public Score: **0.83364**
- OOF AUC ≈ Public Score → confirms no data leakage
- Slightly below V1 (0.83403) — the default LightGBM params were already close to optimal for V1's feature set

### Output
- `submission_V4.csv`

---

## V5 — `improved_submissionV5.ipynb`

**Public Score: 0.83382** (just below V1's 0.83403)

### What changed from V3
- **Same features as V3** (leak-free, with extra safe features)
- **Added Optuna hyperparameter tuning** (50 trials, same search space as V4)

### Why we tried this
V3 had good features but used default hyperparameters. The hypothesis was: maybe the extra features in V3 (speed score, age z-score, interactions) are genuinely useful, but the model needs better tuning to take advantage of them.

### Results
- Public Score: **0.83382**
- Better than V3 (0.82228) — tuning did help the V3 feature set
- Still slightly below V1 (0.83403) — the extra features still aren't adding enough signal

### Key insight
All versions with tuning (V4, V5) are converging around 0.833–0.834, very close to V1. This suggests:
1. The V1 feature set is already quite good
2. The extra features in V3 are not hurting but not helping much either
3. The next step should combine everything: V1 + V3 features + tuning + more folds + ensemble

### Output
- `submission_V5.csv`

---

## Key Lessons Learned

### 1. Data Leakage is dangerous
Computing features using the target variable (`Drafted`) or using the test set during feature engineering causes the model to overfit. The OOF AUC looks great but the public score drops.

**Safe features:** statistics computed only from input features (z-scores, group means by position)
**Leaky features:** target encoding (using `Drafted` to compute school rates), ranks computed on full dataset

**Warning sign:** OOF AUC >> Public Score = suspect leakage

### 2. OOF AUC is your best friend
The Out-of-Fold AUC is a reliable estimate of public score — **as long as there is no data leakage**. When OOF AUC ≈ Public Score, the model is honest and generalizing well.

### 3. Position-relative features matter
A 4.4s sprint is elite for a wide receiver but unremarkable for a defensive back. Raw numbers without context are less informative than numbers relative to position peers. Z-scores within position groups give the model this context.

### 4. Missing values carry information
Players who skip drills are not random — kickers skip bench press, quarterbacks skip certain agility drills. The pattern of missing data is itself a signal about the player's role and draft likelihood.

### 5. Improvement workflow
```
Submit baseline → Add features → Change model → Tune hyperparameters → Create new features → Repeat
```

---

---

## V6 — `improved_submissionV6.ipynb`

**Expected to beat V1 (0.83403)**

### Strategy: Best of everything

#### 1. Features: V1 + V3 (all leak-free)
- BMI, missing flags, n_missing, position z-scores (from V1)
- speed_score, age_pos_zscore, height_x_weight, sprint_x_vertical, sprint_x_broad (from V3)
- Total: 32 features

#### 2. Optuna hyperparameter tuning (50 trials)
- Same search space as V4/V5
- Tuning now runs on 10-fold CV for a more reliable signal

#### 3. 10-fold StratifiedKFold (up from 5-fold)
- Each fold uses ~2500 training samples and ~278 validation samples
- More stable OOF estimate, less variance between runs
- Each test prediction is averaged over 10 models instead of 5

#### 4. Ensemble: average V6 + V1 + V4
- V1 (0.83403) and V4 (0.83364) are the two best clean submissions
- Averaging diverse predictions reduces variance and typically improves AUC
- Saves as `submission_V6_ensemble.csv`

### Results
- Public Score: **0.83523** ✅ New best (+0.00120 from V1)
- Confirms that combining V1+V3 features + Optuna tuning + 10-fold CV all contributed

### Outputs
- `submission_V6.csv` — V6 model alone
- `submission_V6_ensemble.csv` — average of V1 + V4 + V6 (not yet submitted)

---

## File Structure

```
competition/
├── baseline.ipynb                  ← original baseline (untouched)
├── improved_submissionV1.ipynb     ← LightGBM + basic features
├── improved_submissionV2.ipynb     ← more features (had leakage)
├── improved_submissionV3.ipynb     ← fixed leakage
├── improved_submissionV4.ipynb     ← V1 features + Optuna tuning
├── improved_submissionV5.ipynb     ← V3 features + Optuna tuning
├── improved_submissionV6.ipynb     ← V1+V3 features + Optuna + 10-fold + ensemble
├── PROGRESS_LOG.md                 ← this file
├── submission.csv                  ← baseline submission
├── submission_V1.csv
├── submission_V2.csv
├── submission_V3.csv
├── submission_V4.csv
├── submission_V5.csv
├── submission_V6.csv
├── submission_V6_ensemble.csv      ← average of V1 + V4 + V6
└── input/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```
