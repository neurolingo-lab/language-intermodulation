# How to run this pipeline for a new subject via `mne-bids-pipeline`

## Step 1: Adding data to BIDS
Ingest subject raw MEG data into the BIDS dataset using `mne-bids`. You will need to take the raw data into the dataset via the following command:

```bash
mne_bids raw_to_bids --bids_root PATH/TO/BIDS/ROOT --subject_id SUBJECT --session_id SESSION --task TASK
```

This will add the scan to the BIDS dataset and all appropriate `.tsv` and `.json` files. You can also specify the acquisition if you have separate ones etc. See the [full documentation for mne_bids CLI](https://mne.tools/mne-bids/stable/generated/cli.html#mne-bids-raw-to-bids).

Next you will need to convert the DICOM file to the standard format for MNE using [`dcm2bids`](https://unfmontreal.github.io/Dcm2Bids/3.2.0/tutorial/first-steps/#populating-the-config-file), which can be done with the following command in the syntax dataset:

```bash
dcm2bids -d PATH/TO/DICOM.zip -p SUBJECT -c path/to/bids/root/code/folder --auto_extract_entities
```

This will result in either a temporary directory in your current working dir which contains `sub-SUBJECT` or a new folder in the BIDS root for your subject where the converted `.nii.gz` file is. 

If the file is in a temporary directory, go ahead and move the converted image and any `.json` files to the BIDS directory for that subject/session.

Next we will need to write the calibration and crosstalk files, if you have them, to the subject directory. MNE-BIDS supports this luckily:

```bash
mne_bids calibration_to_bids --file PATH/TO/CALFILE.dat --bids_root PATH/TO/BIDS/ROOT --subject_id SUBJECT --session_id SESSION
mne_bids crosstalk_to_bids --file PATH/TO/CROSSTALK.fif --bids_root PATH/TO/BIDS/ROOT --subject_id SUBJECT --session_id SESSION
```

Empty-room data must now be written to the BIDS dataset as well. You will add this data to a dummy subject called "emptyroom" (yes, I know, but this is the canonical way to do it in bids). The session ID will be a ISO-formatted date without any separators, e.g. March 5 2025 would become `20250305`.

```bash
mne_bids raw_to_bids --bids_root PATH/TO/BIDS/ROOT --subject_id emptyroom --session_id ISODATE
```

Now you are ready for the pipeline!

## Step 2: FreeSurfer reconstruction via pipeline

Now you'll need to run the reconstruction step of the pipeline before aligning the fiducials. The pipeline batch script `freesurfer_sub.sh` will do this for you on a SLURM cluster, or alternatively you can run the following command:


```bash
mne_bids_pipeline --config PATH/TO/CONFIG/FILE.py --bids_root PATH/TO/BIDS/ROOT --subject_id SUBJECT --steps freesurfer
```

You must either define your own config file or use the one for the syntax project. 

This will create a FreeSurfer subjects directory in `bids_root/derivatives/freesurfer` which will be used going forwards.

Once the reconstruction is done, you should have a full folder tree of outputs called `sub-SUBJECT` in the above directory.

## Step 3: Coordinate frame alignment (fiducials)

You'll now need to correct the fiducials for the MRI image to match where they were set in the MEG recording. You will do this via the MNE coregistration GUI in Python:

```python
import mne
import mne_bids as mnb
from pathlib import Path


sub = "SUBJECT"
ses = "SESSION"
task = "TASK"
# Put other BIDS keys here, such as acquisition

bids_root = Path("PATH/TO/BIDS/ROOT")
rawbids = mnb.BIDSPath(subject=sub, session=ses, task=task, root=bids_root, datatype="meg", suffix="meg", extension=".fif")
raw = mnb.read_raw_bids(rawbids)

fs_subjects_dir = bids_root / "derivatives/freesurfer/"
t1_fname = fs_subjects_dir / f"sub-{sub}" / "mri/T1.mgz"

mne.gui.coregistration()  # Here you will perform the below steps before continuing

# After writing the trans and fiducials files make landmarks
trans_fname = fs_subjects_dir / f"sub-{sub}" / f"bem/sub-{sub}-fiducials.fif"
trans = mne.read_trans(trans_fname)

landmarks = mnb.get_anat_landmarks(t1_fname, info=raw.info, trans=trans, fs_subject=f"sub-{sub}", fs_subjects_dir=fs_subjects_dir)

# Write the final fiducials to the anatomical landmarks of the BIDS dataset
# Note: We set "overwrite" to True because the original MRI we used for freesurfer is still in this BIDS directory

t1w_bids_path = mnb.BIDSPath(subject=sub, session=ses, root=bids_root, datatype="anat", suffix="T1w")
outpath = mnb.write_anat(image=t1_fname, bids_path=t1w_bids_path, landmarks=landmarks, verbose=True, overwrite=True)
```

With this your MRI data should be prepared for the final steps involving source reconstruction!

## Step 4: Preprocessing the data

Now we will apply preprocessing steps to the data such as maxfiltering and filtering out line noise. First we will run data quality checks and compute the head position for maxfilter. These steps are two separate batch files, `dataquality_sub.sh` and `headpos_sub.sh`, which should be run in that order. The head position in particular takes a very long time to run.


Once you have the head position figured out, you can then run the maxfilter step in `maxfilter_sub.sh` to perform the signal subspace correction on your data. Like the head position calculation, this is a very slow operation. Particularly if you have a long recording.

You can now, after doing the filtering, perform a final notch filter and low/high-pass on the data (as specified in the pipeline config file). This is done together with the ICA fitting for your data, which will fit ICA components that you can then choose to accept/reject as EOG/ECG artifacts. The file `icareg_sub.sh` performs all of these steps.

**At this point, some manual intervention is required.** You should have been reading the log outputs from each of these steps as you ran them, and you will see that the last step will automatically mark some ICA components as bad due to artifacts. You will need to manually check these in the report file that the pipeline outputs. Note which components are truly artifacts, and then open up the `...proc-ica_components.tsv` file in `bids_root/derivatives/sub-SUBJECT/ses-SESSION/meg/`. Here you will go through the components one-by-one and make sure you agree with what it automatically included/excluded.

Once you're happy with the components to regress out, save the changes to this file. You can now move on! From here on out most steps take only a few minutes to run, so no batch scripts are provided.

We will now make epochs from the data (again, check your pipeline config file to be sure your events are correct) and regress out the ICA components. **From here on out arguments for the BIDS root directory, subject, task, and session are omitted for brevity as "standard arguments".

```bash
mne_bids_pipeline [...standard arguments] --steps preprocessing/make_epochs,preprocessing/apply_ica,preprocessing/ptp_reject
```

This final operation will produce the `proc-clean_raw.fif` and `proc-clean_epo.fif` files needed for analysis by removing the components and doing any peak-to-peak channel rejection.

##  Step 5: Basic sensor analyses

Now the pipeline will output a few basic sensor-level analyses. These are really just to give you an eye of sea for the data and future examination.

You can just run the whole set with:

```bash
mne_bids_pipeline [..standard arguments] --steps sensor
```

which will execute them all as a bundle. The full set, if all enabled in the pipeline config, involves getting evoked data per condition, decoding pairs of conditions on whole epochs and with a sliding window, TFR computation, CSP decoding, and noise-covariance estimation.

The final step, which computes the covariance, is important for source analyses so make sure it runs!

## Step 5: Source analyses

Finally, we can run the full source pipeline to compute the forward and inverse solutions. The associated step for this is `--steps source`, which will run the full pipeline.


Good luck!
