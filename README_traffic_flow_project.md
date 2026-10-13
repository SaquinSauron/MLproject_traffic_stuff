# Machine Learning for Traffic Flow Prediction in Urban Areas

## 1. Project overview

This project uses **XGBoost**, a tree-based gradient-boosting algorithm, to classify traffic congestion levels from historical New York City traffic-speed readings recorded during April 2016.

The central question is:

> Can information about a road and time of day, together with recent congestion history, help classify the road's current congestion level?

The project compares a time-and-road baseline against models that use recent congestion and speed history. It also compares the best model with simple baselines to check whether the additional modelling is useful.

**Important scope note:** the current experiment is a **10-road, 28-day sample**, not a model trained on all NYC roads. The code chooses the 10 most frequently observed road links in the first input chunk. Results should therefore be interpreted as results on this experiment's sample, not as city-wide performance.

## 2. Dataset

- **Source file:** `data/april2016.csv.zip`
- **Period:** April 2016; the experiment retains the first 28 days from the start date detected by the script.
- **Raw archive:** 1,141,383 traffic records are scanned in chunks.
- **Selected roads:** 10 links selected by observation frequency in the first source chunk.
- **Aggregation:** readings are grouped into five-minute bins and averaged per road.
- **Usable modelling rows:** 63,751 after creating the shared history features and removing rows with missing required lag/history values.
- **Split:** chronological, with 44,322 training rows, 9,783 validation rows, and 9,646 test rows.

The raw data provides speed and travel-time observations rather than official congestion categories. Consequently, the congestion classes below are **project-defined labels derived from speed**, not official Department of Transportation labels.

### Main source fields

| Field | Meaning / use |
|---|---|
| `DataAsOf` | Timestamp of the traffic observation |
| `linkId` | Road/link identifier |
| `Speed` | Observed speed; used to derive the target label |
| `TravelTime` | Travel time; used for input-row validity checks, not as a model feature |
| `Status` | Source status field; rows with negative status are filtered out |

The script converts timestamps and numeric fields, removes rows with invalid/missing road IDs or timestamps, and keeps only rows with positive speed, positive travel time, and non-negative status. It then aggregates the selected links into five-minute mean-speed observations.

## 3. Defining congestion classes

Because the source does not provide official congestion labels, the project calculates a **speed z-score separately for each road**, using that road's training-period mean and standard deviation:

\[
z = \frac{\text{speed} - \text{training mean speed}}{\text{training standard deviation}}
\]

A positive z-score means speed is above that road's training average; a negative z-score means it is below average.

| Class | Label | Speed z-score |
|---:|---|---|
| 0 | Clear | \(z \geq 2\) |
| 1 | Light | \(1 \leq z < 2\) |
| 2 | Moderate | \(0 \leq z < 1\) |
| 3 | Heavy | \(-1 \leq z < 0\) |
| 4 | Severe | \(z < -1\) |

The speed reference statistics are fitted using the training period only, helping prevent validation/test observations from defining the class thresholds. These categories are a useful experimental proxy, but they should not be presented as officially measured congestion labels.

## 4. Feature engineering and model variants

All model variants are evaluated on the same rows so that their scores can be compared fairly within this experiment.

### Model A — XGBoost baseline

Features:
- Road identifier
- Day of week
- Hour of day

This model asks how much congestion classification is possible using road identity and time information alone.

### Model B — XGBoost + previous congestion

Adds:
- Congestion class exactly 5 minutes earlier (`congestion_5m_ago`)

This tests whether the immediately preceding congestion state improves the classification.

### Model C — XGBoost + congestion at 5, 15 and 30 minutes earlier

Adds:
- `congestion_5m_ago`
- `congestion_15m_ago`
- `congestion_30m_ago`

This is the **best model by held-out test accuracy** in the recorded experiment. It tests whether a short history of congestion is more useful than a single previous observation.

### Model D — XGBoost + congestion and speed trends

Adds the congestion history above, plus:
- Speed 5, 15 and 30 minutes earlier
- Mean speed over the preceding 30 minutes
- Speed standard deviation over the preceding 30 minutes

The rolling speed features are calculated from prior observations, with the speed series shifted before rolling, so the current target observation is not included in its own historical features.

### Handling missing history

Each road is reindexed to a complete five-minute timeline before lag features are made. Therefore, “5 minutes earlier” refers to the previous five-minute time bin, not merely the previous available source row. If a required historical value is missing, that row is removed from the shared modelling sample.

## 5. Training and evaluation design

- **Time-based split:** data is divided chronologically into training, validation, and test periods; it is not randomly shuffled.
- **Hyperparameter selection:** 12 XGBoost configurations are evaluated for each feature set. Tree count, maximum depth, learning rate, and minimum child weight are varied.
- **Model selection:** validation accuracy is the primary selection metric, with macro F1 used to break ties.
- **Final evaluation:** the held-out test period is used for the reported test scores, not for choosing hyperparameters.
- **Rolling one-step evaluation:** predictions use road/time information and historical readings available at or before the timestamp being classified. As time advances, newly observed history can be used. This is **not** a fixed 15-minute or 30-minute-ahead forecast made without receiving new data.

The current script's split boundaries and run details are recorded in `april2016_results/run_metadata.json`; the validation search is in `april2016_results/validation_search.csv`.

## 6. Results

### Recorded results on the held-out test set

| Model | Test accuracy | Macro F1 | Validation accuracy |
|---|---:|---:|---:|
| XGBoost baseline: road + weekday + hour | 61.45% | 0.460 | 67.37% |
| XGBoost + previous congestion | 79.59% | 0.697 | 82.55% |
| **XGBoost + congestion at 5/15/30 minutes** | **79.87%** | **0.700** | 82.92% |
| XGBoost + congestion and speed trends | 79.84% | 0.659 | **83.03%** |
| Persistence baseline: keep previous congestion class | 78.31% | **0.707** | — |
| Majority-class baseline | 46.77% | 0.127 | — |

**Main observations**

1. Adding the previous congestion class improves test accuracy from **61.45% to 79.59%** (+18.14 percentage points).
2. Using congestion at 5, 15, and 30 minutes earlier gives the best test accuracy, **79.87%**, an improvement of **18.42 percentage points** over the baseline.
3. The richer speed-trend model has the highest validation accuracy (**83.03%**) but does not have the highest test accuracy. This is why the held-out test result matters.
4. The persistence baseline reaches **78.31% accuracy** and a macro F1 of **0.707**, which is higher than the best XGBoost model's macro F1. Traffic often remains in the same state for a short time, so a strong model should be compared with this simple strategy.
5. The best XGBoost model improves accuracy over persistence by only **1.56 percentage points**. This is a more realistic measure of the added value of modelling than comparing only against the weaker time-and-road baseline.
6. The test set is class-imbalanced: class 2 (moderate) is the largest class at about **46.77%**, while class 0 (clear) is only about **0.52%**. Accuracy alone can hide poor performance on less common classes; macro F1 gives each class equal weight.

### What are the two simple baselines?

- **Persistence:** predicts that the current congestion class is the same as the previous five-minute congestion class. It is not a trained machine-learning model. It checks whether XGBoost beats the natural tendency of traffic conditions to persist.
- **Majority class:** always predicts the most frequent class, regardless of road or time. Its 46.77% test accuracy is a basic reference point. Its very low macro F1 (0.127) reflects its failure to recognise the other classes.

Results apply to this sampled data, label definition, split, and rolling evaluation setup. They should not be compared directly with results from a different project unless the roads, target labels, feature information, split, and evaluation protocol are aligned.

## 7. Repository structure

```text
MLproject_traffic_stuff/
├── data/
│   └── april2016.csv.zip
├── april2016_results/
│   ├── README.md
│   ├── metrics.csv
│   ├── results.png
│   ├── run_metadata.json
│   ├── sampled_traffic.csv
│   ├── training_speed_reference.csv
│   ├── validation_search.csv
│   ├── test_predictions.csv
│   └── trained XGBoost model files (.ubj)
├── train_april.py
├── requirements.txt
├── run_on_windows.bat
└── README.txt
```

The `april2016_results/` directory contains outputs from the recorded run. Running the training script regenerates these results and may overwrite existing outputs. Back up the folder first if you want to preserve the current results.

## 8. How to run the project on Windows

### Requirements

- Python 3 installed (Python 3.10+ is a sensible choice for a fresh setup)
- Internet connection for installing packages
- Enough disk space for the input archive and generated outputs

### Option A — easiest: use the provided batch file

1. Extract the project ZIP to a folder.
2. Open the extracted `MLproject_traffic_stuff` folder.
3. Double-click `run_on_windows.bat`.

The batch file creates a local `.venv` if needed, installs the pinned dependencies from `requirements.txt`, and runs `train_april.py`. When the run finishes, inspect `april2016_results/metrics.csv`, `april2016_results/results.png`, and `april2016_results/README.md`.

### Option B — run from PowerShell

Open PowerShell in the project root (the folder containing `train_april.py`) and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_april.py
```

If PowerShell blocks activation scripts, you can skip activation and call the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_april.py
```

### Option C — macOS/Linux

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_april.py
```

### Expected outputs

After a successful run, inspect:

- `april2016_results/metrics.csv` — test and validation metrics for the model variants and baselines.
- `april2016_results/results.png` — comparison chart.
- `april2016_results/validation_search.csv` — hyperparameter-search results.
- `april2016_results/test_predictions.csv` — held-out predictions.
- `april2016_results/run_metadata.json` — run settings and split information.
- `april2016_results/training_speed_reference.csv` — training-period speed references used for labels.
- `april2016_results/*.ubj` — saved XGBoost models.

**Reproducibility note:** the README reports the recorded run's metrics. Running again can produce small differences due to environment/library versions or changes to code/data; use the newly generated `metrics.csv` as the result of that run.

## 9. Two-member presentation and demo split

The following division gives each member a clear, connected part. Aim for roughly equal speaking time and practise the hand-off.

### Member 1 — problem, data, preprocessing, and label design

**Explain**
1. Problem statement and objective: classify current congestion into five levels from historical traffic observations.
2. Dataset source and scope: April 2016, raw archive size, 10 selected links, and five-minute aggregation.
3. Why labels must be created: the raw data contains speed readings, not official congestion classes.
4. Explain the z-score formula and the five label thresholds.
5. Explain the chronological train/validation/test split and why future information must not be used to build training labels/features.

**Demo**
1. Show `data/april2016.csv.zip` in the repository.
2. Open `train_april.py` and point out the data loading/cleaning, road selection, five-minute aggregation, label creation, and chronological split sections.
3. Open `april2016_results/training_speed_reference.csv` and explain that the per-road training statistics are used to derive the classes.
4. Show the beginning of `april2016_results/sampled_traffic.csv` to illustrate the processed observations and time-based features.

**Hand-off sentence:** “Now that we have a clean, time-ordered dataset and defined congestion labels, Member 2 will explain the models, results, and what the comparison tells us.”

### Member 2 — features, XGBoost, results, and limitations

**Explain**
1. Why XGBoost was used: it is a gradient-boosted tree model suitable for tabular features.
2. Compare the four feature sets: time/road baseline; previous congestion; 5/15/30-minute congestion history; congestion plus speed trends.
3. Explain the validation-based hyperparameter search and why the test set is held back until final evaluation.
4. Present the metrics table, focusing on the improvement from 61.45% baseline accuracy to 79.87% for the multi-lag model.
5. Explain persistence and majority-class baselines. Be candid that persistence is strong and the best XGBoost model improves its accuracy by only 1.56 percentage points.
6. Mention class imbalance, the project-defined labels, the 10-road sample, and that this is rolling current-state classification rather than a fixed-horizon forecast.

**Demo**
1. Open `april2016_results/metrics.csv` and compare the models.
2. Show `april2016_results/results.png` for a visual comparison.
3. Open `april2016_results/validation_search.csv` to show that hyperparameters were selected using validation results.
4. Open `april2016_results/test_predictions.csv` and explain that the predictions are evaluated against held-out labels.
5. If time permits, run `python train_april.py` (or `run_on_windows.bat`) and show that the outputs are regenerated.

### Suggested live-demo order (about 5–7 minutes)

1. **Member 1 (2–3 min):** repository/data → cleaning and aggregation → label creation → chronological split.
2. **Member 2 (3–4 min):** feature variants → metrics chart/table → persistence comparison → predictions/output files.
3. Keep a backup of the existing `april2016_results/` folder and the already-generated metrics/chart. A full training run may take time, so do not depend on live training to show the results.

## 10. Limitations and future improvements

- The current experiment uses only 10 selected links, so performance on all roads is unknown.
- Congestion labels are derived from per-road speed z-scores and are not official ground-truth labels.
- The classes are imbalanced, particularly the very small clear class.
- Persistence is a strong baseline; additional models should be judged against it, not only against the time-and-road baseline.
- This setup classifies the current five-minute bin using available history. A future project version could define and evaluate a clear future horizon, such as predicting congestion 15 minutes ahead.
- An expanded experiment could include more roads, with the same preprocessing, label definitions, chronological split principles, and baselines held constant for a fair comparison.

## 11. Quick viva answers

**Why use XGBoost?**  
It is a gradient-boosted decision-tree method that can model nonlinear relationships in tabular features such as road, time, and lagged traffic history.

**Why aggregate into five-minute bins?**  
It provides a consistent time interval for comparing observations and constructing precise 5-, 15-, and 30-minute history features.

**Why use a chronological split?**  
Traffic prediction is time-dependent. Training on later observations and testing on earlier ones would give an unrealistic evaluation.

**Why not use current speed as a model feature?**  
Current speed is used to derive the target class. Feeding it into the model would reveal the information that defines the label and would make the evaluation misleading.

**Why is persistence accuracy high?**  
Congestion often changes gradually, so the previous class is frequently still correct five minutes later.

**What is the best model in this run?**  
XGBoost with congestion classes from 5, 15, and 30 minutes earlier, at 79.87% test accuracy and 0.700 macro F1.

**Does the project predict traffic 30 minutes into the future?**  
No. The 30-minute feature is historical input. The current setup classifies the current five-minute observation using past data; it does not forecast the state 30 minutes ahead.

**What is the key limitation?**  
The results are based on a 10-road sample and proxy labels derived from speed, so they should not be treated as city-wide performance against official congestion ground truth.
