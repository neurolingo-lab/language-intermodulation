#!/bin/sh
#SBATCH --job-name=syntax-filtICA
#SBATCH --time=3:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=/home/gercek/worker-logs/syntax-filt_ICA-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB

7z x /home/gercek/livenv.zip -o/tmp/
source /tmp/gercek_li_venv/bin/activate
uv pip install -e /home/gercek/Projects/mne-bids -e /home/gercek/Projects/mne-bids-pipeline --config-settings editable_mode=strict
mne_bids_pipeline --config $HOME/scratch/syntax_im/mnebids_pipeline_config.py  --subject $PIPELINE_SUB --session 01 --steps preprocessing/frequency_filter,preprocessing/regress_artifact,preprocessing/fit_ica,preprocessing/find_ica_artifacts --task $PIPELINE_TASK
