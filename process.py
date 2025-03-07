# -*- coding: utf-8 -*-
"""
@author: Daniel Capellán-Martín <daniel.capellan@upm.es>
"""

# TODO: Patch extraction
# TODO: Correct LATMM position - more posterior

from utils import *
from functions_main import *
import argparse
import tensorflow as tf
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process AP and/or LAT CXR pediatric images. img_AP and img_LAT folders should contain analogous images. If only AP or LAT images are introduced (not both), no matching between views will be done.')
    parser.add_argument("--csv", action='store', required=True, default=None, help="[.csv file] Comma-separated CSV containing information about the images. Possible columns: ['case_id','img_path_AP','img_path_LAT'].")
    parser.add_argument("--seg_model", action='store', required=True, default=None, choices=['nnunet','medt','gatedaxialunet'], help="Model used for segmenting the lungs. Choose among: ['nnunet','medt','gatedaxialunet'].")
    parser.add_argument("--output_folder", "-o", action='store', default=None, help="[str] Output folder. Required.")
    parser.add_argument("--fformat", "-f", action='store', default=None, choices=['jpg','png','nifti'], help="[str] JPG, PNG, NIFTI (.nii.gz) formats allowed. Possible choices: ['jpg','png','nifti'].")
    parser.add_argument("--no_clahe", "-nc", action='store_false', default=True, required=False, help="If flag, CLAHE will not be applied.")
    parser.add_argument("--yolo_conf_frontal", "-ycf", action='store', default=0.5, required=False, help="YOLO confidence threshold for frontal images.")
    parser.add_argument("--yolo_conf_lateral", "-ycl", action='store', default=0.5, required=False, help="YOLO confidence threshold for lateral images.")
    args = parser.parse_args()

    csv_path = args.csv
    seg_model = args.seg_model
    output_folder = args.output_folder
    file_format_orig = args.fformat
    apply_clahe = args.no_clahe
    BASE_DIR = os.getcwd()
    MEDT_DIR = os.path.join(os.getcwd(),'Medical-Transformer')

    # File format definition
    file_format = file_format_orig
    if file_format_orig=='nifti': file_format='nii.gz'
    file_format = '.'+file_format

    # Run checks
    if(not(os.path.isabs(output_folder))):
        if ('./' in output_folder):
            output_folder = output_folder.replace('./','')
        output_folder = os.path.join(BASE_DIR,output_folder)
        maybe_make_dir(output_folder)
    df, views = check_and_adapt_csv(csv_path,file_format)

    # Define paths
    paths = {
        'preprocessed': os.path.join(output_folder,'preprocessed'),
        'yolo_in': os.path.join(output_folder, 'yolo_in'),
        'yolo_out': os.path.join(output_folder, 'yolo_out'),
        'cropped': os.path.join(output_folder, 'cropped'),
        'cropped_clahe': os.path.join(output_folder, 'cropped_clahe'),
        'nnunet_in': os.path.join(output_folder, 'nnunet_in'),
        'nnunet_out': os.path.join(output_folder, 'nnunet_predicted'),
        'medt_in': os.path.join(output_folder, 'medt_in'),
        'medt_out': os.path.join(output_folder, 'medt_predicted'),
        'gatedaxialunet_in': os.path.join(output_folder, 'gatedaxialunet_in'),
        'gatedaxialunet_out': os.path.join(output_folder, 'gatedaxialunet_predicted'),
        'orientation_corrected': os.path.join(output_folder, 'orientation_corrected'),
        'regions': os.path.join(output_folder, 'regions'), 
        'yolo_models': {
            'AP': os.path.join(BASE_DIR,'yolov5_weights','AP_pTB_yolov5_weights_v12112021.pt'),
            'LAT': os.path.join(BASE_DIR,'yolov5_weights','LAT_pTB_yolov5_weights_v09022022.pt')
        },
        'medt_models':{
            'AP':{
                'medt': os.path.join(BASE_DIR,'medt_weights','AP','medt','17022022_145805_medtfinal_model.pth'),
                'gatedaxialunet': os.path.join(BASE_DIR,'medt_weights','AP','gatedaxialunet','17022022_140431_gatedaxialunetfinal_model.pth'),
            },
            'LAT':{
                'medt': os.path.join(BASE_DIR,'medt_weights','LAT','medt','17022022_145402_medtfinal_model.pth'),
                'gatedaxialunet': os.path.join(BASE_DIR,'medt_weights','LAT','gatedaxialunet','17022022_140619_gatedaxialunetfinal_model.pth'),
            }
        },
        'lat_cnn_model': os.path.join(BASE_DIR,'lat_cnn_resnet_model','LAT_orientation_pTBResNetBAdam_saved-model-200-1.00_best.h5')
    }
    
    # Process
    # Steps 0-1: With a for loop
    views_bkup = views.copy()
    for index, row in df.iterrows():
        # Info
        print(f"Preprocessing case {index}: {row['case_id']}")
        # Restore views
        views = views_bkup.copy()
        # Get paths
        if('AP' in views):
            img_path_AP = row['img_path_AP']
        if('LAT' in views):
            img_path_LAT = row['img_path_LAT']
        # At this point, paths should be properly defined
        # Step 0. Read image(s)
        if(file_format_orig in ['jpg','png']):
            if('AP' in views):
                if not (str(img_path_AP).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_AP==""):
                    img_AP = imread(img_path_AP)
                else:
                    views.remove('AP')
            if('LAT' in views):
                if not (str(img_path_LAT).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_LAT==""):
                    img_LAT = imread(img_path_LAT)
                else:
                    views.remove('LAT')
        elif(file_format_orig in ['nifti']):
            if('AP' in views):
                if not (str(img_path_AP).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_AP==""):
                    img_AP = read_nifti(img_path_AP)
                else:
                    views.remove('AP')
            if('LAT' in views):
                if not (str(img_path_LAT).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_LAT==""):
                    img_LAT = read_nifti(img_path_LAT)
                else:
                    views.remove('LAT')
        # Step 1. Preprocess & save
        for view in views:
            # Check if image is already preprocessed
            maybe_make_dir(os.path.join(paths['preprocessed'],view))
            maybe_make_dir(os.path.join(paths['yolo_in'],view))
            if(view=='AP'):
                if(file_format_orig in ['jpg','png']):
                    if check_existing_path(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_AP))) and check_existing_path(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_AP))):
                        continue
                elif(file_format_orig in ['nifti']):
                    if check_existing_path(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_AP)[:-7]+'.jpg')) and check_existing_path(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_AP)[:-7]+'.jpg')):
                        continue
            elif(view=='LAT'):
                if(file_format_orig in ['jpg','png']):
                    if check_existing_path(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_LAT))) and check_existing_path(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_LAT))):
                        continue
                elif(file_format_orig in ['nifti']):
                    if check_existing_path(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_LAT)[:-7]+'.jpg')) and check_existing_path(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_LAT)[:-7]+'.jpg')):
                        continue
            if(view=='AP'):
                img = img_AP.copy()
            elif(view=='LAT'):
                img = img_LAT.copy()
            # CLAHE
            if(apply_clahe):
                img_clahe = adapt_2d_and_apply_clahe(img)
            else:
                img_clahe = img
            img_clahe_resize = resize(img_clahe,(512,512))
            if(apply_clahe):
                # Only needed if CLAHE is applied
                img_clahe = cv2.normalize(img_clahe,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            img_clahe_resize = cv2.normalize(img_clahe_resize,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            if(view=='AP'):
                if(file_format_orig in ['jpg','png']):
                    imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_AP)),img_clahe)
                    imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_AP)),img_clahe_resize)
                elif(file_format_orig in ['nifti']):
                    imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_AP)[:-7]+'.jpg'),img_clahe)
                    imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_AP)[:-7]+'.jpg'),img_clahe_resize)
            elif(view=='LAT'):
                if(file_format_orig in ['jpg','png']):
                    imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_LAT)),img_clahe)
                    imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_LAT)),img_clahe_resize)
                elif(file_format_orig in ['nifti']):
                    imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_LAT)[:-7]+'.jpg'),img_clahe)
                    imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_LAT)[:-7]+'.jpg'),img_clahe_resize)
    # Checkpoint dataset.csv
    df.to_csv(os.path.join(output_folder,'dataset.csv'))
    # Step 2. Crop with YOLO (batch)
    os.chdir(os.path.join(BASE_DIR,'yolov5'))
    for view in views:
        if view == 'AP':
            yolo_thresh = args.yolo_conf_frontal
        elif view == 'LAT':
            yolo_thresh = args.yolo_conf_lateral
        YOLO_MODEL_PATH = paths['yolo_models'][view]
        DIM = 512
        CONF = yolo_thresh
        IMG_DIR_FOR_YOLO = os.path.join(paths['yolo_in'],view)
        PROJECT_OUT = os.path.join(paths['yolo_out'])
        NAME = view
        MAX_DETECTIONS = 1
        if not os.path.exists(os.path.join(PROJECT_OUT,NAME)):
            os.system(f"python detect.py --weights {YOLO_MODEL_PATH} --img {DIM} --conf {CONF} --source {IMG_DIR_FOR_YOLO} --project {PROJECT_OUT} --name {NAME} --exist-ok --max-det {MAX_DETECTIONS} --save-txt --save-conf")
        else:
            print(f"YOLO predictions for view {view} already exist. Skipping.")
    ## 2.1. Restore base path
    os.chdir(BASE_DIR)
    ## Step 2.2. Crop images with YOLOv5 predictions
    err_idx = []
    views_bkup = views.copy()
    for index, row in df.iterrows():
        # Info
        print(f"Cropping case {index}: {row['case_id']}")
        # Restore views
        views = views_bkup.copy()
        # Get paths
        if('AP' in views):
            img_path_AP = row['img_path_AP']
        if('LAT' in views):
            img_path_LAT = row['img_path_LAT']
        # At this point, paths should be properly defined
        # Step 2.2.0. Read image(s)
        if(file_format_orig in ['jpg','png']):
            if('AP' in views):
                if not (str(img_path_AP).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_AP==""):
                    img_AP = imread(img_path_AP)
                else:
                    views.remove('AP')
            if('LAT' in views):
                if not (str(img_path_LAT).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_LAT==""):
                    img_LAT = imread(img_path_LAT)
                else:
                    views.remove('LAT')
        elif(file_format_orig in ['nifti']):
            if('AP' in views):
                if not (str(img_path_AP).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_AP==""):
                    img_AP = read_nifti(img_path_AP)
                else:
                    views.remove('AP')
            if('LAT' in views):
                if not (str(img_path_LAT).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or img_path_LAT==""):
                    img_LAT = read_nifti(img_path_LAT)
                else:
                    views.remove('LAT')
        # Step 2.2.1. Crop & save
        for view in views:
            # Check if image is already cropped
            maybe_make_dir(os.path.join(paths['cropped'],view))
            maybe_make_dir(os.path.join(paths['cropped_clahe'],view))
            if(file_format_orig in ['jpg','png']):
                if check_existing_path(os.path.join(paths['cropped'],view,row[f"fname_{view}_without_ext"]+file_format)):
                    continue
                if check_existing_path(os.path.join(paths['cropped_clahe'],view,row[f"fname_{view}_without_ext"]+file_format)):
                    continue
            if(file_format_orig in ['nifti']):
                if check_existing_path(os.path.join(paths['cropped'],view,row[f"fname_{view}_without_ext"])):
                    continue
                if check_existing_path(os.path.join(paths['cropped_clahe'],view,row[f"fname_{view}_without_ext"])):
                    continue
            YOLO_IMG_PATH = os.path.join(paths['yolo_out'],view)
            YOLO_TXT_PATH = os.path.join(YOLO_IMG_PATH,'labels')
            if(view=='AP'):
                img = img_AP.copy()
            elif(view=='LAT'):
                img = img_LAT.copy()
            bbox_txt_file = [y for y in sorted(glob.glob(os.path.join(YOLO_TXT_PATH,'*.txt'))) if row[f"fname_{view}_without_ext"] in y]
            if(bbox_txt_file and len(bbox_txt_file)==1):
                bbox_txt_file = bbox_txt_file[0]
            else:
                f = row[f"img_path_{view}"]
                print(f"No YOLO prediction for file {f} and view {view}. Skipping. This case will not be considered in further steps.")
                err_idx.append(index)
                continue
            # Get coordinates and probability from TXT
            with open(bbox_txt_file) as f:
                lines = f.readlines()
            X,Y,W,H = [float(c) for c in lines[0].split(' ')][1:5]
            prob = float(lines[0].split(' ')[-1]) # probability
            (x,y,w,h) = xywhn2xywh_bbox(X,Y,H,W,img_height=img.shape[0],img_width=img.shape[1])
            # Crop image
            img_cropped = crop_img(img,(x,y,h,w))
            # CLAHE
            img_cropped_clahe = adapt_2d_and_apply_clahe(img_cropped)
            img_cropped_clahe = cv2.normalize(img_cropped_clahe,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            # Save images
            if(file_format_orig in ['jpg','png']):
                imsave(os.path.join(paths['cropped'],view,row[f"fname_{view}_without_ext"]+file_format),img_cropped)
                imsave(os.path.join(paths['cropped_clahe'],view,row[f"fname_{view}_without_ext"]+file_format),img_cropped_clahe)
            if(file_format_orig in ['nifti']):
                convert_2d_image_array_to_nifti_without_zeros(img_cropped, os.path.join(paths['cropped'],view,row[f"fname_{view}_without_ext"]))
                convert_2d_image_array_to_nifti_without_zeros(img_cropped_clahe, os.path.join(paths['cropped_clahe'],view,row[f"fname_{view}_without_ext"]))
    ## Step 2.3 Remove those cases which reported errors
    df = df.drop(err_idx)
    # Checkpoint dataset.csv
    df.to_csv(os.path.join(output_folder,'dataset.csv'))
    # Step 3. Segmentation with selected model (batch)
    print("Segmenting...")
    if(seg_model in ['nnunet']):
        for view in views:
            if(apply_clahe):
                adapt_images_nnunet(folder_in=os.path.join(paths['cropped_clahe'],view),folder_out=os.path.join(paths['nnunet_in'],view),file_format=file_format)
            else:
                adapt_images_nnunet(folder_in=os.path.join(paths['cropped'],view),folder_out=os.path.join(paths['nnunet_in'],view),file_format=file_format)
            INPUT_FOLDER = os.path.join(paths[f"{seg_model}_in"],view)
            OUTPUT_FOLDER = os.path.join(paths[f"{seg_model}_out"],view)
            if(view=='AP'):
                os.system(f"nnUNet_predict -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -tr nnUNetTrainerV2_50epochs -m 2d -t 136 --disable_tta")
            if(view=='LAT'):
                os.system(f"nnUNet_predict -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -tr nnUNetTrainerV2_50epochs -m 2d -t 135 --disable_tta")
    elif(seg_model in ['medt','gatedaxialunet']):
        for view in views:
            if(apply_clahe):
                adapt_images_medt(folder_in=os.path.join(paths['cropped_clahe'],view),folder_out=os.path.join(paths[f"{seg_model}_in"],view),file_format=file_format,resize_dim=256)
            else:
                adapt_images_medt(folder_in=os.path.join(paths['cropped'],view),folder_out=os.path.join(paths[f"{seg_model}_in"],view),file_format=file_format,resize_dim=256)
            os.chdir(MEDT_DIR)
            MODEL_DIR = paths['medt_models'][view][seg_model]
            INPUT_FOLDER = os.path.join(paths[f"{seg_model}_in"],view)
            OUTPUT_FOLDER = os.path.join(paths[f"{seg_model}_out"],view)
            maybe_make_dir(OUTPUT_FOLDER)
            DIM = 256
            maybe_remove_jupyter_checkpoints(INPUT_FOLDER)
            if(seg_model in ['medt']):
                MODEL_NAME = "MedT"
            else:
                MODEL_NAME = seg_model
            os.system(f"python test.py --loaddirec {MODEL_DIR} --val_dataset {INPUT_FOLDER} --direc {OUTPUT_FOLDER} --batch_size 1 --modelname {MODEL_NAME} --imgsize {DIM} --gray ''no''")
            os.chdir(BASE_DIR)
    # Checkpoint dataset.csv
    df.to_csv(os.path.join(output_folder,'dataset.csv'))
    # Step 4. Schema over the images
    views_bkup = views.copy()
    for index,row in df.iterrows():
        # Info
        print(f"Extracting regions from case {index}: {row['case_id']}")
        # Restore views
        views = views_bkup.copy()
        # Process views
        if('AP' in views):
            if (str(row['img_path_AP']).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or row['img_path_AP']==""):
                views.remove('AP')
        if('LAT' in views):
            if (str(row['img_path_LAT']).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or row['img_path_LAT']==""):
                views.remove('LAT')
        # Process
        if('AP' in views and 'LAT' in views):
            # 4.1. Paths
            ## 4.1.1. Input image paths
            if(seg_model in ['nnunet']):
                ext = '.nii.gz'
                img_path_AP = os.path.join(paths[f"{seg_model}_in"],'AP',row['fname_AP_without_ext']+'_0000'+ext)
                img_path_LAT = os.path.join(paths[f"{seg_model}_in"],'LAT',row['fname_LAT_without_ext']+'_0000'+ext)
            elif(seg_model in ['medt','gatedaxialunet']):
                ext = '.png'
                img_path_AP = os.path.join(paths[f"{seg_model}_in"],'AP','img',row['fname_AP_without_ext']+ext)
                img_path_LAT = os.path.join(paths[f"{seg_model}_in"],'LAT','img',row['fname_LAT_without_ext']+ext)
            ## 4.1.2. Paths for draw & region extraction
            img_path_AP_reg = os.path.join(paths['cropped'],'AP',os.path.basename(row["img_path_AP"]))
            img_path_LAT_reg = os.path.join(paths['cropped'],'LAT',os.path.basename(row["img_path_LAT"]))
            ## 4.1.3. Predicted labels
            lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*'+ext)) if str(row['case_id'])+"-" in f]
            lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*'+ext)) if str(row['case_id'])+"-" in f]
            if(lbl_path_AP and len(lbl_path_AP)==1):
                lbl_path_AP=lbl_path_AP[0]
            else:
                raise Exception(f"AP label for case {row['case_id']} could not be determined. Please check before moving forward.")
            if(lbl_path_LAT and len(lbl_path_LAT)==1):
                lbl_path_LAT=lbl_path_LAT[0]
            else:
                raise Exception(f"LAT label for case {row['case_id']} could not be determined. Please check before moving forward.") 
            # 4.2. Read data
            ## 4.2.1. Read images & labels
            if(seg_model in ['nnunet']):
                img_AP = read_nifti(img_path_AP)
                img_LAT = read_nifti(img_path_LAT)
                lbl_AP = read_nifti(lbl_path_AP)
                lbl_LAT = read_nifti(lbl_path_LAT)
            elif(seg_model in ['medt','gatedaxialunet']):
                img_AP = imread(img_path_AP)
                img_LAT = imread(img_path_LAT)
                lbl_AP = imread(lbl_path_AP)
                lbl_LAT = imread(lbl_path_LAT)
            ## 4.2.2. Read images for regions
            if(file_format_orig in ['jpg','png']):
                img_AP_reg = imread(img_path_AP_reg)
                img_LAT_reg = imread(img_path_LAT_reg)
            elif(file_format_orig in ['nifti']):
                img_AP_reg = read_nifti(img_path_AP_reg)
                img_LAT_reg = read_nifti(img_path_LAT_reg)
            # 4.3. LAT orientation correction
            img_LAT_CNN = preprocess_with_clahe(img_LAT,img_shape=(256,256))
            img_LAT_CNN = np.expand_dims(img_LAT_CNN,axis=0)
            model = tf.keras.models.load_model(paths['lat_cnn_model'])
            orientation = model.predict(img_LAT_CNN).flatten()[0]
            orientation_binary = (orientation>0.5).astype(int)
            print(f"Predicted Orientation: {orientation_binary}")
            df.loc[index,'pred_orientation'] = orientation_binary
            if(orientation_binary==0):
                img_LAT = cv2.flip(img_LAT, 1)
                img_LAT_reg = cv2.flip(img_LAT_reg, 1)
                lbl_LAT = cv2.flip(lbl_LAT, 1)
                if(seg_model=='nnunet'):
                    lbl_LAT = cv2.normalize(lbl_LAT,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                maybe_make_dir(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT'))
                imsave(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT','image_corrected.jpg'),img_LAT)
                imsave(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT','label_corrected.jpg'),lbl_LAT)
            # 4.4. Get regions
            regions_AP, img_AP_rotated_draw, regions_LAT, img_LAT_rotated_draw = get_regions_final(img_AP,lbl_AP,img_LAT,lbl_LAT,img_AP_reg,img_LAT_reg)
            # 4.5. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['case_id'],'AP'))
            maybe_make_dir(os.path.join(paths['regions'],row['case_id'],'LAT'))
            ## AP - Save
            out_path = os.path.join(paths['regions'],row['case_id'],'AP','regions_AP.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_AP, fp, cls=NpEncoder)
            img_AP_rotated_draw = ensure_image_format(img_AP_rotated_draw)
            imsave(os.path.join(paths['regions'],row['case_id'],'AP','img_AP_regions.jpg'),img_AP_rotated_draw)
            ## LAT - Save
            out_path = os.path.join(paths['regions'],row['case_id'],'LAT','regions_LAT.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_LAT, fp, cls=NpEncoder)
            img_LAT_rotated_draw = ensure_image_format(img_LAT_rotated_draw)
            imsave(os.path.join(paths['regions'],row['case_id'],'LAT','img_LAT_regions.jpg'),img_LAT_rotated_draw)
        elif('AP' in views):
            # 4.1. Paths
            ## 4.1.1. Paths for region definition
            if(seg_model=='nnunet'):
                img_path_AP = os.path.join(paths[f"{seg_model}_in"],'AP',row['fname_AP_without_ext']+'_0000.nii.gz')
            elif(seg_model in ['medt','gatedaxialunet']):
                img_path_AP = os.path.join(paths[f"{seg_model}_in"],'AP','img',row['fname_AP_without_ext']+'.png')
            ## 4.1.2. Paths for draw & region extraction
            img_path_AP_reg = os.path.join(paths['cropped'],'AP',os.path.basename(row["img_path_AP"]))
            ## 4.1.3. Predicted labels
            if(seg_model=='nnunet'):
                lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*.nii.gz')) if row['case_id']+"-" in f]
            elif(seg_model in ['medt','gatedaxialunet']):
                lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*.png')) if row['case_id']+"-" in f]
            if(lbl_path_AP and len(lbl_path_AP)==1):
                lbl_path_AP=lbl_path_AP[0]
            else:
                raise Exception(f"AP label for case {row['case_id']} could not be determined. Please check before moving forward.")
            # 4.2. Read images
            ## 4.2.1. Read images
            if(seg_model in ['medt','gatedaxialunet']):
                img_AP = imread(img_path_AP)
            elif(seg_model=='nnunet'):
                img_AP = read_nifti(img_path_AP)
            img_AP_reg = imread(img_path_AP_reg)
            # 4.2.2. Read labels
            if(seg_model=='nnunet'):
                lbl_AP = read_nifti(lbl_path_AP)
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_AP = imread(lbl_path_AP)
            # 4.3. Get regions
            try:
                regions_AP, img_AP_rotated_draw = get_regions_final_only_AP(img_AP,lbl_AP,img_AP_reg)
            except Exception as e:
                print(f"Error in case {row['case_id']}. Skipping. Error: {e}")
                continue
            # 4.4. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['case_id'],'AP'))
            out_path = os.path.join(paths['regions'],row['case_id'],'AP','regions_AP.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_AP, fp, cls=NpEncoder)
            img_AP_rotated_draw = ensure_image_format(img_AP_rotated_draw)
            imsave(os.path.join(paths['regions'],row['case_id'],'AP','img_AP_regions.jpg'),img_AP_rotated_draw)
        elif('LAT' in views):
            # 4.1. Paths
            ## 4.1.1. Paths for region definition
            if(seg_model=='nnunet'):
                img_path_LAT = os.path.join(paths[f"{seg_model}_in"],'LAT',row['fname_LAT_without_ext']+'_0000.nii.gz')
            elif(seg_model in ['medt','gatedaxialunet']):
                img_path_LAT = os.path.join(paths[f"{seg_model}_in"],'LAT','img',row['fname_LAT_without_ext']+'.png')
            ## 4.1.2. Paths for draw & region extraction
            img_path_LAT_reg = os.path.join(paths['cropped'],'LAT',os.path.basename(row["img_path_LAT"]))
            ## 4.1.3. Predicted labels
            if(seg_model=='nnunet'):
                lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*.nii.gz')) if row['case_id']+"-" in f]
            elif(seg_model in ['medt','gatedaxialunet']):
                lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*.png')) if row['case_id']+"-" in f]
            if(lbl_path_LAT and len(lbl_path_LAT)==1):
                lbl_path_LAT=lbl_path_LAT[0]
            else:
                raise Exception(f"LAT label for case {row['case_id']} could not be determined. Please check before moving forward.") 
            # 4.2. Read images
            ## 4.2.1. Read images
            if(seg_model in ['medt','gatedaxialunet']):
                img_LAT = imread(img_path_LAT)
            elif(seg_model=='nnunet'):
                img_LAT = read_nifti(img_path_LAT)
            img_LAT_reg = imread(img_path_LAT_reg)
            # 4.2.2. Read labels
            if(seg_model=='nnunet'):
                lbl_LAT = read_nifti(lbl_path_LAT)
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_LAT = imread(lbl_path_LAT)
            # 4.3. LAT orientation correction
            img_LAT_CNN = preprocess_with_clahe(img_LAT,img_shape=(256,256))
            img_LAT_CNN = np.expand_dims(img_LAT_CNN,axis=0)
            model = tf.keras.models.load_model(paths['lat_cnn_model'])
            orientation = model.predict(img_LAT_CNN).flatten()[0]
            orientation_binary = (orientation>0.5).astype(int)
            print(f"Predicted Orientation: {orientation_binary}")
            df.loc[index,'pred_orientation'] = orientation_binary
            if(orientation_binary==0):
                img_LAT = cv2.flip(img_LAT, 1)
                img_LAT_reg = cv2.flip(img_LAT_reg, 1)
                lbl_LAT = cv2.flip(lbl_LAT, 1)
                if(seg_model=='nnunet'):
                    lbl_LAT = cv2.normalize(lbl_LAT,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                maybe_make_dir(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT'))
                imsave(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT','image_corrected.jpg'),img_LAT)
                imsave(os.path.join(paths['orientation_corrected'],row['case_id'],'LAT','label_corrected.jpg'),lbl_LAT)
            # 4.4. Get regions
            regions_LAT, img_LAT_rotated_draw = get_regions_final_only_LAT(img_LAT,lbl_LAT,img_LAT_reg)
            # 4.5. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['case_id'],'LAT'))
            out_path = os.path.join(paths['regions'],row['case_id'],'LAT','regions_LAT.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_LAT, fp, cls=NpEncoder)
            img_LAT_rotated_draw = ensure_image_format(img_LAT_rotated_draw)
            imsave(os.path.join(paths['regions'],row['case_id'],'LAT','img_LAT_regions.jpg'),img_LAT_rotated_draw)
    # Checkpoint dataset.csv
    df.to_csv(os.path.join(output_folder,'dataset.csv'))
    views_bkup = views.copy()
    # Step 5. Crop patches and save
    for index,row in df.iterrows():
        # Info
        print(f"Cropping regions in case {index}: {row['case_id']}")
        # Restore views
        views = views_bkup.copy()
        # Process views
        if(row['img_path_AP'].endswith(('jpg','png'))):
            if('AP' in views):
                if (str(row['img_path_AP']).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or row['img_path_AP']==""):
                    views.remove('AP')
            if('LAT' in views):
                if (str(row['img_path_LAT']).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or row['img_path_LAT']==""):
                    views.remove('LAT')
        for view in views:
            # Load image to extract regions from
            img_path_reg = os.path.join(paths['cropped'],view,os.path.basename(row[f"img_path_{view}"]))
            if(file_format in ['.jpg','.png']):
                img_reg = imread(img_path_reg)
            elif(file_format in ['.nii.gz']):
                img_reg = read_nifti(img_path_reg)
            # Load JSON with relative coordinates
            json_path = os.path.join(paths['regions'],row['case_id'],view,f"regions_{view}.json")
            if os.path.exists(json_path):
                with open(json_path) as json_file:
                    regions = json.load(json_file)
            else:
                print(f"Regions JSON for case {row['case_id']} and view {view} in path {json_path} not found. Skipping.")
                continue
            # Apply flip to LAT image
            if(view=='LAT' and row['pred_orientation']==0):
                img_reg = cv2.flip(img_reg, 1)
            # Apply rotation to AP image
            if(view in ['AP']):
                img_reg = rotate_image(img_reg, -(regions['image_rotation']))
            # Crop regions
            for reg in regions['rel']:
                maybe_make_dir(os.path.join(paths['regions'],row['case_id'],view,reg))
                if('lung' in reg):
                    for third_idx in regions['rel'][reg]['thirds']:
                        # Crop region
                        coord_x = int(np.round(regions['rel'][reg]['thirds'][third_idx]['x']*img_reg.shape[1]))
                        coord_y = int(np.round(regions['rel'][reg]['thirds'][third_idx]['y']*img_reg.shape[0]))
                        coord_w = int(np.round(regions['rel'][reg]['thirds'][third_idx]['width']*img_reg.shape[1]))
                        coord_h = int(np.round(regions['rel'][reg]['thirds'][third_idx]['height']*img_reg.shape[0]))
                        img = crop_img(img_reg,(coord_x,coord_y,coord_h,coord_w))
                        # Save
                        out_path = os.path.join(paths['regions'],row['case_id'],view,reg,f"third_{int(third_idx)+1}.jpg")
                        imsave(out_path,img)
                # Crop region
                coord_x = int(np.round(regions['rel'][reg]['x']*img_reg.shape[1]))
                coord_y = int(np.round(regions['rel'][reg]['y']*img_reg.shape[0]))
                coord_w = int(np.round(regions['rel'][reg]['width']*img_reg.shape[1]))
                coord_h = int(np.round(regions['rel'][reg]['height']*img_reg.shape[0]))
                img = crop_img(img_reg,(coord_x,coord_y,coord_h,coord_w))
                # Save
                out_path = os.path.join(paths['regions'],row['case_id'],view,reg,f"{reg}.jpg")
                imsave(out_path,img)
    # Checkpoint dataset.csv
    df.to_csv(os.path.join(output_folder,'dataset.csv'))
    # Step 6. Completion message
    print("###################")
    print(f"Finished. Your results have been saved in: {output_folder}")
    print("###################")
    print('\n')

def check_and_adapt_csv(csv_path,file_format):
    AP_OK, LAT_OK = 2*(False,)
    views = []
    if(os.path.isfile(csv_path)):
        try:
            df = pd.read_csv(csv_path)
            df = df.reset_index()  # Reset index in case CSV contains index
        except:
            print("Please check CSV, maybe bad formatted.")
        if(len(df)<1):
            raise Exception("Empty CSV. Please check before moving forward.")
        if('case_id' not in df.keys()):
            raise Exception("'case_id' column not included in the CSV. Please check before moving forward.")
        selected_columns = ['case_id']
        if 'img_path_AP' in df.columns:
            selected_columns.append('img_path_AP')
        if 'img_path_LAT' in df.columns:
            selected_columns.append('img_path_LAT')
        if('img_path_AP' in df.keys() and 'img_path_LAT' in df.keys()):
            df = df[['case_id','img_path_AP','img_path_LAT']]
            AP_OK, LAT_OK = 2*(True,)
        elif('img_path_AP' in df.keys()):
            df = df[['case_id','img_path_AP']]
            AP_OK = True
        elif('img_path_LAT' in df.keys()):
            df = df[['case_id','img_path_LAT']]
            LAT_OK = True
    else:
        raise Exception("CSV file does not exist.")
    # Create views list
    if(AP_OK):
        views.append('AP')
    if(LAT_OK):
        views.append('LAT')
   
    def get_full_path(img_path):
     if pd.notna(img_path):  # Evita errores con valores NaN
        if not os.path.isabs(img_path):  
            return os.path.join(img_path,)  
     return img_path

    # Apply the function to the path columns
    df['img_path_AP'] = df['img_path_AP'].apply(lambda x: get_full_path(x) if pd.notna(x) else np.nan)
    df['img_path_LAT'] = df['img_path_LAT'].apply(lambda x: get_full_path(x) if pd.notna(x) else np.nan)

    # If everything is OK, proceed with more checks.
    for index,row in df.iterrows():
        for view in views:
            f = row[f"img_path_{view}"]
            if pd.notna(f):  # Avoid errors with NaN
               f = get_full_path(f)  # Ensure full path
            if (str(f).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or str(f)==""):
                continue
            if os.path.isdir(os.path.dirname(f)):
                assert os.path.isfile(f), f"File {f} not found! Please check before moving forward."
                
            else:
                raise Exception(f"Folder {os.path.dirname(f)} does not exist. Please check paths")
            # Create another column with fname_without_ext per view. Check format consistency and file existance.
            if file_format in ['.jpg', '.png']:
                ext = os.path.splitext(f)[-1]
                # Create the column if it does not exist
                if f"fname_{view}_without_ext" not in df.columns:
                    df[f"fname_{view}_without_ext"] = ""  # Inicializa con strings vacíos
                # Asegurar que la columna es de tipo 'object' (string)
                df[f"fname_{view}_without_ext"] = df[f"fname_{view}_without_ext"].astype(str)
                # Verificar formato del archivo
                assert ext == file_format, f"File formats not matching. File {f} not matching {file_format}."
                # Asignar el nombre del archivo sin extensión
                df.loc[index, f"fname_{view}_without_ext"] = os.path.splitext(os.path.basename(f))[0]

            elif(file_format in ['.nii.gz']):
                ext = os.path.basename(f)[-7:]
                if f"fname_{view}_without_ext" not in df.columns:
                    df[f"fname_{view}_without_ext"] = ""  # Inicializa con strings vacíos
                
                # Asegurar que la columna es de tipo 'object' (string)
                df[f"fname_{view}_without_ext"] = df[f"fname_{view}_without_ext"].astype(str)

                # Verificar formato del archivo
                assert ext == file_format, f"File formats not matching. File {f} not matching {file_format}."

            
                df.loc[index,f"fname_{view}_without_ext"] = (os.path.splitext(os.path.basename(f))[0][:-4])

    # If everything is OK, proceed with more checks.
    for index,row in df.iterrows():
        for view in views:
            f = row[f"img_path_{view}"]
            if (str(f).endswith(("0", "None", "none", "n/a", "N/a", "nan", "-1")) or str(f)==""):
                continue
            if(file_format in ['.jpg','.png']):
                img = imread(f)
            elif(file_format in ['.nii.gz']):
                img = read_nifti(f)
            
            if len(img.shape)!=2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                #print(f"WARNING: Images should be in 2D! Please consider this as a warning, you could have problems in next steps if not controlled. Dimensions: {img.shape}")
    
    df.replace("", np.nan, inplace=True)
    return df, views

def adapt_images_nnunet(folder_in,folder_out,file_format):
    subfiles = sorted(glob.glob(os.path.join(folder_in,'*'+file_format)))
    for f in subfiles:
        maybe_make_dir(folder_out)
        if(file_format in ['.jpg','.png']):
            output_filename_truncated = os.path.splitext(os.path.basename(f))[0]
            out_path_truncated = os.path.join(folder_out,output_filename_truncated)
            convert_2d_image_to_nifti(f, out_path_truncated)
        elif(file_format == '.nii.gz'):
            out_path = os.path.join(folder_out,os.path.basename(f)[:-7]+'_0000.nii.gz')
            copy(f,out_path)

def adapt_images_medt(folder_in,folder_out,file_format,resize_dim=256):
    subfiles = sorted(glob.glob(os.path.join(folder_in,'*'+file_format)))
    for f in subfiles:
        if(file_format in ['.jpg','.png']):
            img = imread(f)
        elif(file_format == '.nii.gz'):
            img = read_nifti(f)
        img = resize(img,(resize_dim,resize_dim))
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        maybe_make_dir(os.path.join(folder_out,'img'))
        maybe_make_dir(os.path.join(folder_out,'labelcol'))
        if(file_format in ['.jpg','.png']):
            out_path = os.path.join(folder_out,'img',os.path.splitext(os.path.basename(f))[0]+'.png')
            out_path_labelcol = os.path.join(folder_out,'labelcol',os.path.splitext(os.path.basename(f))[0]+'.png')  # Not used by medt implementation, just to comply with code's requirements.
        elif(file_format == '.nii.gz'):
            out_path = os.path.join(folder_out,'img',os.path.splitext(os.path.basename(f))[0][:-4]+'.png')
            out_path_labelcol = os.path.join(folder_out,'labelcol',os.path.splitext(os.path.basename(f))[0][:-4]+'.png')  # Not used by medt implementation, just to fullfil with code's requirements.
        imsave(out_path,img)
        imsave(out_path_labelcol,img)
        
def check_existing_path(path):
    if(os.path.exists(path)):
        print(f"{path} already exists.")
        return True
    
def ensure_image_format(image, target_format='2d'):
    """
    Ensures the image has the specified format (2D grayscale or 3D RGB).
    
    Parameters:
    - image: numpy array representing the image
    - target_format: '2d' for grayscale or '3d' for RGB
    
    Returns:
    - Image converted to the requested format
    """
    # Handle 4D images first
    if image.ndim == 4:
        print(f"⚠️ Warning: 4D image detected {image.shape}, adjusting...")
        
        if image.shape[0] == 1:  
            image = np.squeeze(image, axis=0)  # Remove batch dimension
        elif image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)  # Remove redundant channel dimension
        elif image.shape[-1] == image.shape[-2]:  # Special case
            image = image[..., 0]
            image = ensure_image_format(image, target_format)  # Recursively call with 3D image
            
        if image.ndim == 4:
            raise ValueError(f"Could not reduce 4D image: {image.shape}")
    
    # Convert to target format
    if target_format.lower() == '2d':
        if image.ndim == 3 and image.shape[-1] in [3, 4]:  # RGB/RGBA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:  # Already 2D
            return image
        else:
            raise ValueError(f"Incompatible image format for 2D conversion: {image.shape}")
    
    elif target_format.lower() == '3d':
        if image.ndim == 2:  # Grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[-1] == 3:  # Already RGB
            return image
        elif image.ndim == 3 and image.shape[-1] == 4:  # RGBA to RGB
            return image[..., :3]
        else:
            raise ValueError(f"Incompatible image format for 3D conversion: {image.shape}")
    
    else:
        raise ValueError(f"Invalid target format: {target_format}. Must be '2d' or '3d'")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__=="__main__":
    main()