# POI Routine / Travel Decomposition

Built on top of **[RecBole](https://github.com/RUCAIBox/RecBole)**. `recbole/` is the
original RecBole with a few evaluation metrics added ([RecBole modifications](#recbole-modifications)).

> Zhao, W. X. et al. *RecBole: Towards a Unified, Comprehensive and Efficient
> Framework for Recommendation Algorithms.* CIKM 2021.

We split check-ins into **routine** and **travel**, then run two experiments.

| | What it does | Pipeline |
|---|---|---|
| **Preprocessing** | Label raw check-ins as routine/travel and build datasets | `preprocess/` |
| **EXP1** | Compare recommendation performance on three datasets: travel / routine / raw | RecBole |
| **EXP2** | travelset (set of venues on the next trip) prediction + framework metrics | Standalone implementation |

```
preprocess/   raw check-ins -> routine/travel labeling -> dataset build  (output to dataset/)
exp1/         EXP1
exp2/         EXP2
configs/      experiment configs
runtime.py    seed list · automatic GPU selection · seed averaging
run.sh        run script
```

---

## Installation

```bash
conda create -n poi python=3.9
conda activate poi
pip install -r requirements.txt
pip install "scipy<1.13"
```

In a CUDA environment the GPU is found and used automatically; otherwise it runs on CPU.

---

## Data preparation

The datasets are not distributed due to copyright. Download the originals and build them with `preprocess/`.

### 1. Download the originals

From the **Global-scale Check-in Dataset with User Social Networks** (WWW 2019) on the
[Foursquare Dataset page](https://sites.google.com/site/yangdingqi/home/foursquare-dataset),
move just these two files as-is. No conversion is needed.

```
dataset/foursquare/raw_POIs.txt
dataset/foursquare/dataset_WWW_Checkins_anonymized.txt
```

For Gowalla, put `loc-gowalla_totalCheckins.txt.gz` from
[SNAP](https://snap.stanford.edu/data/loc-gowalla.html) in `dataset/` (optional).

### 2. Routine/travel labeling

```
UVR(v) = unique visitors / total check-ins             higher = travel
RVR(v) = share of visitors who visited 2+ times        higher = routine
```

Stage 1 uses the mean of per-venue values within each Foursquare category (venues with fewer than 10 check-ins excluded). Gowalla has no categories, so the same is done per 0.5° grid bin; everything else is identical.

| Stage | Condition | Label |
|---|---|---|
| Stage 1 (category) A | RVR ≥ 0.5 **AND** UVR ≤ 0.25 | all routine |
| Stage 1 (category) B | RVR < 0.5 **AND** UVR > 0.25 | all travel |
| Stage 2 (user-venue pair, C/E only) | visits ≥ 2 **AND** ≥ 4× the average of other users at the same venue | routine |
| Stage 2, single-visitor venue | visits ≥ 4 | routine |

### 3. Thresholds

The defaults are **UVR 0.25, RVR 0.5**. To change them, edit the values at the top of
[`preprocess/daily_label.py`](preprocess/daily_label.py) and rerun the build.

```python
RVR_THRESHOLD = 0.5
UVR_THRESHOLD = 0.25
REL_TAU = 4.0            # stage-2 threshold. If you change UVR, setting it to 1/UVR is recommended
```

### 4. Build

```bash
python preprocess/venue_category_stats.py --countries US,JP,DE,SG,KR   # category UVR/RVR

python preprocess/build_travel.py    --scopes US,JP,DE,SG,KR   # EXP1 travel
python preprocess/build_routine.py   --scopes US,JP,DE,SG,KR   # EXP1 routine
python preprocess/build_raw.py       --scopes US,JP,DE,SG,KR   # EXP1 raw (k-core only, no labeling)
python preprocess/build_travelset.py --scopes US,JP,DE,SG,KR   # EXP2

python preprocess/build_tourist_label.py --scopes US,JP,DE,SG,KR   # (optional) for the tourist-purity metric
```

All three arms drop venues without a category or with invalid coordinates, then apply 5-core.

| File | What it does |
|---|---|
| `daily_label.py` | Labeling rules and thresholds (used by every build script) |
| `atomic_common.py` | Shared logic: k-core, coordinate validation, writing RecBole files, etc. |
| `venue_category_stats.py` | Per-category UVR/RVR |
| `build_travel.py` / `build_routine.py` / `build_raw.py` | The three EXP1 arms |
| `build_travelset.py` | EXP2 data (last 3 travel check-ins = test, the one before = valid) |
| `build_tourist_label.py` | Tourist-purity labels (optional) |
| `build_gowalla.py` | Gowalla version, grid bins instead of categories (optional) |
| `dataset_stats.py` / `export_category_stats.py` | For inspecting statistics |

---

## Running

Edit only the settings at the top of `run.sh`, then run it.

```bash
bash run.sh --list     # list the experiments that would run
bash run.sh
```

```bash
DATA_ROOT=dataset
EXPS="exp1 exp2"
SCOPES="kr"
EXP1_ARMS="travel routine raw"
MODELS="sasrec lightgcn userfreq userrandom random popularity"
SEEDS=                 # empty = 556-565
```

Every experiment runs with **seeds 556–565 (10 seeds)** and saves their average in a single JSON.
Rules that use no randomness (`userfreq`, `popularity`) are computed only once.

---

## Results

`results/exp{1,2}/<dataset>-<model>-<timestamp>.json`

```json
{
  "meta":        { "dataset": ..., "model": ..., "seeds": [556, ..., 565], "topk": [5,10,20] },
  "config":      { ... },
  "data":        { "n_users": ..., "n_items": ..., ... },
  "metrics":     { seed mean },
  "metrics_std": { seed standard deviation },
  "per_seed":    [ { "seed": 556, "metrics": { ... }, "checkpoint": ... }, ... ]
}
```

Rule baselines go under `metrics.by_rule.<rule>`. EXP2 additionally has `ground_truth`,
the same metrics computed on the ground-truth set itself.

**EXP1** — RecBole split LS + TO (each user's last check-in is the test target), full-sort evaluation.

| Key | Meaning |
|---|---|
| `recall@k`, `ndcg@k` | Accuracy |
| `test_revisit_rate` | Share of test targets that are previously visited venues |
| `revisitamp@k` | Share of previously visited venues in recommendations / the user's original revisit rate |
| `category_coverage@k` | Recommended categories / original categories |
| `unseen_cat_rate@k` | Share of venues of a kind the user has never visited |
| `popularity_internal@k` | Popularity measured on the training portion |
| `tourist_purity@k` | Only when `TOURIST=on` |

**EXP2** — Train next-item prediction on the prefix, evaluate against the set of venues on the next trip (2.8 on average).

| Key | Meaning |
|---|---|
| `recall@k`, `ndcg@k` | Accuracy (averaged per user) |
| `recall_revisit@k` + `recall_novel@k` | Hits split into previously visited / novel (same for `ndcg`) |
| `revisitamp@k`, `category_coverage@k`, `unseen_cat_rate@k` | Same definitions as EXP1 |
| `locinside@k` | Share of recommendations inside the user's existing activity radius |
| `loccompact@k` | How clustered the k recommendations are (km) |

---

## Models · Hyperparameters

| Name | Description |
|---|---|
| `sasrec`, `lightgcn` | Trained models |
| `userfreq` / `userrandom` | Venues in the user's history, by visit count / at random |
| `random` / `popularity` | Uniformly random / most popular in the training portion |

| | EXP1 SASRec | EXP1 LightGCN | EXP2 |
|---|---|---|---|
| seeds | 556–565 | 556–565 | 556–565 |
| epochs / stopping_step | 300 / 10 | 200 / 10 | 100 / 10 |
| train_batch_size | 2048 | 8192 | 512 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| hidden / layers / heads | 64 / 2 / 2 | 64 / 2 / — | 64 / 2 / 2 |
| dropout | 0.5 | — | 0.2 |
| topk | [5, 10, 20] | [5, 10, 20] | [5, 10, 20] |

Config files: `configs/`, `recbole/properties/model/`. Values set in `run.sh` variables take precedence.


---

## RecBole modifications

Metric classes added to `recbole/evaluator/metrics.py`: `CategoryCoverage`,
`PersonalRepeat`, `Serendipity`, `RevisitAmp`, `UnseenCategoryRate`, `TouristPurity`.
`recbole/evaluator/collector.py` additionally collects `rec.users` for these metrics.
