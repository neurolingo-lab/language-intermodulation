#!/bin/sh
#SBATCH --job-name=syntax-recon
#SBATCH --time=12:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=/home/gercek/worker-logs/syntax-reconst-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

7z x /home/gercek/livenv.zip -o/tmp/
source /tmp/gercek_li_venv/bin/activate
module load FreeSurfer/7.3.2-centos7_x86_64
mne_bids_pipeline --config $HOME/scratch/syntax_im/mnebids_pipeline_config.py  --subject $PIPELINE_SUB --steps freesurfer
