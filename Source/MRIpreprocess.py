import os
import nibabel as nib
from nibabel import nifti1
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from deepbrain import Extractor
import intensity_normalization as inorm
from intensity_normalization.normalize import zscore

# Global Variables
_FMRI_DATA_PATH_ = './Datasets/Mri/Unnormalized'
_FMRI_STRIPPED_DATA_PATH_ = './Datasets/Mri/Stripped'
_FMRI_NORMALIZED_DATA_PATH_ = './Datasets/Mri/Normlaized'

# MRI Data Pre-Processing
def skull_strip(input_file_path, output_file_path):
    """
    Function to perform skull stripping using the fsqc library on MRI images in separate "HC" and "PD" subfolders.

    Args:
    - input_folder (str): Path to the input folder containing "HC" and "PD" subfolders.
    - output_folder (str): Path to save the skull-stripped MRI images in separate "HC_stripped" and "PD_stripped" subfolders.
    """
    # Create output folders if they don't exist
    output_hc_folder = os.path.join(output_file_path,"HC")
    output_pd_folder = os.path.join(output_file_path,"PD")
    os.makedirs(output_hc_folder, exist_ok=True)
    os.makedirs(output_pd_folder, exist_ok=True)
    print("Stripping sub-dir found/created - ",output_hc_folder , " and ", output_pd_folder)
    
    # Iterate through subfolders
    for subfolder in ["HC", "PD"]:
        input_subfolder = os.path.join(input_file_path, subfolder)
        output_subfolder = output_hc_folder if subfolder == "HC" else output_pd_folder
        print("input sub-dir found - ", input_subfolder)
        # Iterate through files in subfolder
        for file_name in os.listdir(input_subfolder):
            print(file_name," located")
            if file_name.endswith(".nii"):
                print("file type = *.nii")
                input_file_ = os.path.join(input_subfolder, file_name)
                output_file_path = os.path.join(output_subfolder, file_name)
                
                # Load the input MRI image
                img = nifti1.load(input_file_)
                img_data = img.get_fdata()
                print("nii data shape: ", img_data.shape)


                # Initialize the skull stripping tool
                ext = Extractor()
                
                try:
                    if len(img_data.shape) == 4:  # If the MRI data has 4 dimensions
                    # Select the middle volume (or any other suitable approach)
                        middle_index = img_data.shape[3] // 2
                        # Extract a single volume from the 4D MRI data
                        single_volume_data = img_data[:, :, :, middle_index]
                        # Reshape the data to have a single channel
                        single_volume_data = single_volume_data[..., np.newaxis]
                        skull_stripped_img_data = ext.run(single_volume_data)
                    else:
                        # Process the MRI data normally
                        skull_stripped_img_data = ext.run(img_data)

                    # Create a new Nifti1Image object with the skull stripped data and the affine from the original image
                    skull_stripped_img_ = nib.Nifti1Image(skull_stripped_img_data, img.affine)
                    nib.save(skull_stripped_img_, output_file_path)
                    print(f"Skull stripping successful: {file_name}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

def normalize_intensity(input_folder, output_folder):
    """
    Function to normalize the intensity of MRI 

    Args:
    - input_folder (str): Path to the input folder containing MRI images.
    - output_folder (str): Path to save the normalized MRI images.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through files in input folder
    for file_name in os.listdir(input_folder):
        print(file_name," located")
        if file_name.endswith(".nii"):
            print("file type = *.nii")
            input_file_ = os.path.join(input_folder, file_name)
            output_file_ = os.path.join(output_folder, file_name)

            normalizer = zscore.ZScoreNormalize()
            img = normalizer.load_image(input_file_)
            normalized_img = normalizer.normalize_image(img)
            nib.save(normalized_img, output_file_)

def mri_preprocessing():
    """
    Function to preprocess MRI data.
    """
    # Code for skull stripping using SynthStrip
    skull_strip(_FMRI_DATA_PATH_, _FMRI_STRIPPED_DATA_PATH_)

    # Normalize intensity
    normalize_intensity(_FMRI_STRIPPED_DATA_PATH_+"/HC", _FMRI_NORMALIZED_DATA_PATH_+"/HC")
    normalize_intensity(_FMRI_STRIPPED_DATA_PATH_+"/PD", _FMRI_NORMALIZED_DATA_PATH_+"/PD")