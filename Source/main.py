import os
import MRIpreprocess
import MRI_3DCNN
import MRI_DenseNet
import EEG_3DCNN
import resource

# Set the maximum memory usage to 24 GB (24 * 1024 MB)
resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 * 1024 * 1024, resource.RLIM_INFINITY))


if __name__ == '__main__':
    # Global Variables
    _FMRI_DATA_PATH_ = './Datasets/Mri/Unnormalized'
    _FMRI_STRIPPED_DATA_PATH_ = './Datasets/Mri/Stripped'
    _FMRI_NORMALIZED_DATA_PATH_ = './Datasets/Mri/Normlaized'
    _EEG_DATA_PATH_ = './Datasets/EEG/Unnormalized'
    

    #MRI_3DCNN.train_image_classifier(_FMRI_NORMALIZED_DATA_PATH_)
    #MRI_DenseNet.train_image_classifier(_FMRI_NORMALIZED_DATA_PATH_)
    EEG_3DCNN.train_eeg_classifier(_EEG_DATA_PATH_)