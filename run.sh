#!/usr/bin/env bash
#  ─────────────────────────────────────────────────────────────────────
#  Experiment runner script
#
#      bash run.sh              # run with the settings below as-is
#      bash run.sh --list       # only list which experiments would run, then exit
#
#  If this is your first time, build the datasets first. See "Building the datasets"
#  in README.md.  If there is nothing under dataset/, this script skips everything.
#
#  Output
#      results/<rq>/[<combination>/]<dataset>-<model>-<timestamp>.json
#      saved/<rq>/[<combination>/]<dataset>-<model>/         checkpoints
#  If DATA_ROOT is dataset/ itself, results accumulate directly without the <combination> level.
#  ─────────────────────────────────────────────────────────────────────

# ======================= Settings =======================

#  The GPU is picked automatically (the one with the most free memory, or CPU if none).

#  Directory that holds the datasets.
#      dataset                       when built with a single threshold
#      dataset/uvr0.25-rvr0.50       when stored separately per threshold combination
DATA_ROOT=dataset

#  Experiments to run:  exp1 | exp2  (space-separated, in order)
#      exp1  performance comparison of the three arms travel/routine/raw   (RecBole evaluation path)
#      exp2  travelset set prediction + framework metrics                  (standalone pipeline)
EXPS="exp1 exp2"

#  Scopes. The <scope> part of foursquare-<scope>-<arm>.
SCOPES="kr"

#  Arms to compare in exp1.
#      travel  travel only   routine  routine only   raw  all (k-core only)
EXP1_ARMS="travel routine raw"

#  Models.
#      sasrec lightgcn                          trained
#      userfreq userrandom random popularity    rules without training (processed in one go)
MODELS="sasrec lightgcn userfreq userrandom random popularity"

#  Tourist purity (tourist_purity):  on | off
#  Turning it on requires dataset/venue_tourist_label/.
#      python dataset/build_tourist_label.py --scopes KR
TOURIST=off

#  Skip training if a checkpoint exists. Leave empty to always train from scratch.
REUSE=1

# ------- If left empty, the values below come from the config files -------
#      exp1 -> configs/common.yaml (+ model-family yaml)
#      exp2 -> configs/exp2.yaml
#  If SEEDS is empty, 556-565 (10 seeds). The result JSON stores the mean over seeds.
SEEDS=
TOPK=
EPOCHS=
STOPPING_STEP=
TRAIN_BATCH_SIZE=
EVAL_BATCH_SIZE=
LEARNING_RATE=

# ------- exp2 model parameters (for exp1, the model yaml handles these) -------
HIDDEN_SIZE=
N_LAYERS=
N_HEADS=
DROPOUT=
MAX_LEN=

#  Automatic adjustment of exp1's eval_batch_size.
#      SASRec's full-sort evaluation builds a score matrix of size eval_batch_size x n_items.
#      On datasets with many items, using the config default of 4096 as-is causes
#      CUDA OOM.  If EVAL_BATCH_SIZE is filled in, this automatic adjustment is ignored.
AUTO_EVAL_BS=1

#  Python executable. If using a conda environment, point to the python inside it.
PY=python

# ===================== End of settings =====================


set -u
cd "$(dirname "$0")" || exit 1
T0=$(date +%s)

LIST=0
[ "${1:-}" = "--list" ] && LIST=1

RULE_MODELS=" userfreq userrandom random popularity "
TOUR=""
[ "$TOURIST" = "on" ] && TOUR="--tourist"
COMBO=$(basename "$DATA_ROOT")
[ "$COMBO" = "dataset" ] && COMBO=""

FAILED=""
N_OK=0
N_FAIL=0

#  Print "--key value" only when a value is set. If empty, print nothing so that
#  the Python-side default (None) is kept, and then the config file value is used.
opt () {
  if [ -n "$2" ]; then printf ' %s %s' "$1" "$2"; fi
}

auto_eval_bs () {
  local f="$1/$2/$2.item" n
  [ -f "$f" ] || { echo ""; return; }
  n=$(( $(wc -l < "$f") - 1 ))
  if   [ "$n" -gt 200000 ]; then echo 1024
  elif [ "$n" -gt 100000 ]; then echo 2048
  else echo ""; fi
}

miss () {
  if [ -d "$1/$2" ]; then return 1; fi
  N_FAIL=$((N_FAIL + 1))
  FAILED="${FAILED}${FAILED:+$'\n'}  ${2}  (dataset not found: $1/$2)"
  echo "!!! dataset not found: $1/$2"
  return 0
}

run () {
  local name="$1"; shift
  if [ "$LIST" = "1" ]; then
    echo "    $name"
    return 0
  fi
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo ">>> ${name}   $(date '+%m-%d %H:%M:%S')"
  echo "────────────────────────────────────────────────────────────────"
  #  Don't redirect to a log file; stream straight to the screen. That way the tqdm progress bar is visible.
  #  To keep it in a file:  bash run.sh 2>&1 | tee run.log
  if "$@"; then
    N_OK=$((N_OK + 1))
  else
    N_FAIL=$((N_FAIL + 1))
    FAILED="${FAILED}${FAILED:+$'\n'}  ${name}"
    echo "!!! failed: ${name}"
  fi
}

exp1_one () {
  local root="$1" d="$2"
  miss "$root" "$d" && return
  local bs="$EVAL_BATCH_SIZE"
  if [ -z "$bs" ] && [ "$AUTO_EVAL_BS" = "1" ]; then bs=$(auto_eval_bs "$root" "$d"); fi

  local rules="" m name cf
  for m in $MODELS; do
    case "$RULE_MODELS" in *" $m "*) rules="${rules:+$rules,}$m"; continue ;; esac
    case "$m" in
      sasrec)   name=SASRec;   cf="configs/common.yaml configs/sequential.yaml" ;;
      lightgcn) name=LightGCN; cf="configs/common.yaml configs/general.yaml configs/graph.yaml" ;;
      *) echo "!!! unknown model for exp1: $m"; continue ;;
    esac
    # shellcheck disable=SC2046
    run "${d}-${name}" \
      $PY exp1/run.py -m "$name" -d "$d" --config_files="$cf" \
        --data_path "$root" $TOUR ${REUSE:+--reuse} \
        $(opt --seeds "$SEEDS") $(opt --topk "$TOPK") \
        $(opt --epochs "$EPOCHS") $(opt --stopping_step "$STOPPING_STEP") \
        $(opt --train_batch_size "$TRAIN_BATCH_SIZE") \
        $(opt --eval_batch_size "$bs") \
        $(opt --learning_rate "$LEARNING_RATE")
  done
  #  The rule baselines read the dataset only once and run the rules in one go.
  if [ -n "$rules" ]; then
    # shellcheck disable=SC2046
    run "${d}-baselines" \
      $PY exp1/baselines.py -d "$d" \
        --config_files="configs/common.yaml configs/sequential.yaml" \
        --data_path "$root" --rules "$rules" $TOUR \
        $(opt --seeds "$SEEDS") $(opt --topk "$TOPK") \
        $(opt --eval_batch_size "$bs")
  fi
}

exp2_one () {
  local root="$1" d="$2"
  miss "$root" "$d" && return
  local rules="" m
  for m in $MODELS; do
    case "$RULE_MODELS" in *" $m "*) rules="${rules:+$rules,}$m"; continue ;; esac
    case "$m" in sasrec|lightgcn) ;; *) echo "!!! unknown model for exp2: $m"; continue ;; esac
    # shellcheck disable=SC2046
    run "${d}-${m}" \
      $PY exp2/train.py -d "$d" -m "$m" --data_root "$root" \
        $TOUR ${REUSE:+--reuse} \
        $(opt --seeds "$SEEDS") $(opt --topk "$TOPK") \
        $(opt --epochs "$EPOCHS") $(opt --stopping_step "$STOPPING_STEP") \
        $(opt --train_batch_size "$TRAIN_BATCH_SIZE") \
        $(opt --eval_batch_size "$EVAL_BATCH_SIZE") \
        $(opt --learning_rate "$LEARNING_RATE") \
        $(opt --hidden_size "$HIDDEN_SIZE") $(opt --n_layers "$N_LAYERS") \
        $(opt --n_heads "$N_HEADS") $(opt --dropout "$DROPOUT") \
        $(opt --max_len "$MAX_LEN")
  done
  if [ -n "$rules" ]; then
    # shellcheck disable=SC2046
    run "${d}-baselines" \
      $PY exp2/baselines.py -d "$d" --data_root "$root" --rules "$rules" $TOUR \
        $(opt --seeds "$SEEDS") $(opt --topk "$TOPK") \
        $(opt --eval_batch_size "$EVAL_BATCH_SIZE")
  fi
}

# ──────────────────────────────────────────────────────────────────────
[ "$LIST" = "1" ] && echo "Experiments to run:"
for RQ in $EXPS; do
  [ "$LIST" = "1" ] && echo "  [$RQ]  DATA_ROOT=$DATA_ROOT"
  for SC in $SCOPES; do
    if [ "$RQ" = "exp1" ]; then
      for ARM in $EXP1_ARMS; do exp1_one "$DATA_ROOT" "foursquare-${SC}-${ARM}"; done
    else
      exp2_one "$DATA_ROOT" "foursquare-${SC}-travelset"
    fi
  done
done

if [ "$LIST" = "1" ]; then
  echo
  echo "To actually run them, omit --list:  bash run.sh"
  exit 0
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo "  finished   $(date '+%m-%d %H:%M:%S')   (elapsed $((($(date +%s) - T0) / 60)) min)"
echo "  succeeded ${N_OK} / failed ${N_FAIL}"
if [ "$N_FAIL" -gt 0 ]; then
  echo
  echo "  failed experiments:"
  echo "$FAILED"
  echo
  echo "  To rerun only the experiments above, narrow SCOPES / EXP1_ARMS / MODELS"
  echo "  and run bash run.sh again (with REUSE=1, succeeded ones are skipped)."
fi
echo "════════════════════════════════════════════════════════════════"
exit "$N_FAIL"
