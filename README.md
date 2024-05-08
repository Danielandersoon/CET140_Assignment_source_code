# CET140_Assignment_source_code
 Source code for the CET140 first-year specialist project. This is a classification model to hopefully allow for the detection of PD patients by their fMRI and rs-eeg measurements

The solution takes *.nii (NIFTI) images as the input for the fMRI and *.bdf as the input type for the EEG data
These should be as such relative to the source directory

/</br>
├─ Source/</br>
├─ Datasets/</br>
│  ├─ MRI/</br>
│  │  ├─ HC/</br>
│  │  ├─ PD/</br>
│  ├─ EEG/</br>
│  │  ├─ HC/</br>
│  │  ├─ PD/</br>

Dependencies:
you will need the following pip packages:
numpy;
tensorflow;
nibabel;
scipy;
matplotlib;
tqdm;
mne;


