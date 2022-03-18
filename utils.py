import os
import SimpleITK as sitk
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage.exposure import equalize_adapthist
from skimage.io import imread, imsave
from skimage.transform import resize
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
from shutil import copy, rmtree
from tqdm.notebook import tqdm
import cv2
import json

def maybe_make_dir(path):
    if(not(os.path.isdir(path))):
        os.makedirs(path)

def read_nifti(path):
    # Reader
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    # Path
    reader.SetFileName(path)
    # Image
    img = reader.Execute();
    img = sitk.GetArrayFromImage(img)[0]
    # Return image
    return img

def adapt_2d_and_apply_clahe(img):
    '''3D images to 2D and apply CLAHE'''
    if(len(img.shape)>2):
        img = rgb2gray(img)
    img = equalize_adapthist(img)  # CLAHE
    return img    

def convert_2d_image_array_to_nifti(img, output_filename_truncated: str, spacing=(999, 1, 1), transform=None, is_seg: bool = False):
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    img = imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

def overlay_image_and_label_nifti(img_path,lbl_path,configs,index=0,info={},resize_img=True,font_size=1):
    '''Overlays image and label with an alpha (default 0.3)
    Inputs:
        - img_path
        - lbl_path
        - configs
        - info: dict - {case_id, cohort, age_yrs} - Contains information about the case. If empty, nothing would be printed out.
    Outputs:
        - overlay: ndarray - image with label overlaid'''
    # Set-up
    if(info):
        case_id = info['case_id']
        cohort = info['cohort']
        age_yrs = info['age_yrs']
    DIM = configs['DIM_IMG_IN_COLLAGE']
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
     # image
    reader.SetFileName(img_path)
    img = reader.Execute();
    img = sitk.GetArrayFromImage(img)[0]
    if(resize_img):
        img = resize(img,(DIM,DIM))
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    # label
    reader.SetFileName(lbl_path)
    lbl = reader.Execute()
    lbl = sitk.GetArrayFromImage(lbl)[0]
    max_label = np.max(lbl)
    lbl[lbl>0] = 255  # Do not harm the image in the resize operation
    if(resize_img):
        lbl = resize(lbl,(512,512))
        lbl = cv2.normalize(lbl,None,0,max_label,cv2.NORM_MINMAX).astype(np.uint8)
    # overlay
    overlay = label2rgb(lbl, image=img, bg_label=0, colors=['red'])  # Labels: 0 - background (no color), 1 - Lungs (red)
    if(resize_img):
        overlay = resize(overlay,(DIM,DIM,3))
    overlay = cv2.normalize(overlay,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    # If info, display text
    if(info):
        overlay = cv2.putText(overlay,f"{case_id} ({cohort})",(int(overlay.shape[1]*0.03),int(overlay.shape[0]*0.9)), configs['FONT'], font_size*0.75, configs['RGB_COLOR'], font_size*1, cv2.LINE_AA)
        overlay = cv2.putText(overlay,f"Age (yrs): {age_yrs}",(int(overlay.shape[1]*0.03),int(overlay.shape[0]*0.96)), configs['FONT'], font_size*0.75, configs['RGB_COLOR'], font_size*1, cv2.LINE_AA)
        overlay = cv2.putText(overlay,f"{index+1}",(int(overlay.shape[1]*0.9),int(overlay.shape[0]*0.1)), configs['FONT'], font_size*1, configs['RGB_COLOR'], 2, cv2.LINE_AA)
    return overlay

def preprocess_with_clahe(img,img_shape):
    '''Function that preprocesses images - Resize to --IMG_SHAPE--, apply CLAHE and normalize images. If image is RGB -> to grayscale.'''
    if(len(img.shape)>2):
        img = rgb2gray(img)
    img = resize(img,img_shape)
    img = equalize_adapthist(img)  # CLAHE
    img = normalize_img(img)
    return img

def normalize_img(img):
    epsilon = 1e-10
    img_norm = (img-np.mean(img))/(np.std(img)+epsilon)
    return img_norm

def xywhn2xywh_bbox(X,Y,H,W,img_height,img_width):
    x = int(np.round(img_width*(X-W/2)))
    y = int(np.round(img_height*(Y-H/2)))
    w = int(np.round(img_width*W))
    h = int(np.round(img_height*H))
    return (x,y,w,h)

def crop_img(img,coords):
    '''coords: (x,y,h,w)'''
    x,y,h,w = coords
    img_cropped = img[y:y+h,x:x+w]
    return img_cropped

def maybe_remove_jupyter_checkpoints(path_dir):
    for root, subdirs, files in os.walk(path_dir):
        if(os.path.basename(root)=='.ipynb_checkpoints'):
            shutil.rmtree(root)