from argparse import ArgumentParser

import mne
import mne_bids as mnb


def coreg_dialogue(subject, session, task, bids_root):
    """Run the MNE coregistration GUI."""
    # Set up the BIDS path and subject/session/task identifiers
    bids_path = mnb.BIDSPath(subject=subject, session=session, task=task, root=bids_root)
    rawpath = bids_path.copy().update(extension=".fif", suffix="meg", datatype="raw")

    # Load the raw data
    raw = mne.io.read_raw_bids(rawpath)

    try:
        import easygui

        transpath = bids_root / "sourcedata" / f"sub-{subject}_trans.fif"
        easygui.msgbox(
            "Please run the coregistration GUI and save the results.\n\n"
            f"Results should be saved to {transpath}."
        )
    except ImportError:
        print(
            "Please run the coregistration GUI and save the results.\n\n"
            f"Results should be saved to {transpath}."
        )

    # Run the coregistration GUI
    mne.gui.coregistration(
        inst=raw,
        subject="sub-" + subject,
        subjects_dir=bids_root / "derivatives" / "freesurfer",
        block=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run MNE coregistration GUI")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    parser.add_argument("--task", type=str, required=True, help="Task ID")
    parser.add_argument("--bids_root", type=str, required=True, help="BIDS root directory")

    args = parser.parse_args()

    # Call the coregistration function with the provided arguments
    coreg_dialogue(args.subject, args.session, args.task, args.bids_root)
