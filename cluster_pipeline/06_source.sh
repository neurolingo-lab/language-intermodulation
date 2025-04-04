#!/bin/sh
#SBATCH --job-name=syntax-sourcemodel
#SBATCH --time=3:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=/home/gercek/worker-logs/syntax-source-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB

module load FreeSurfer/7.3.2-centos7_x86_64
7z x /home/gercek/livenv.zip -o/tmp/
source /tmp/gercek_li_venv/bin/activate
uv pip install -e /home/gercek/Projects/mne-bids -e /home/gercek/Projects/mne-bids-pipeline --config-settings editable_mode=strict
mne_bids_pipeline --config $HOME/scratch/syntax_im/mnebids_pipeline_config.py  --subject $PIPELINE_SUB --session 01 --steps sensor/make_evoked,sensor/make_cov,source/make_bem_surfaces,source/make_bem_solution,source/setup_source_space --task $PIPELINE_TASK
