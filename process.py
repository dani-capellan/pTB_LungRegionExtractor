# -*- coding: utf-8 -*-
"""
@author: Daniel Capellán-Martín <daniel.capellan@upm.es>
"""

from utils import *
from functions_main import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process AP and/or LAT CXR pediatric images. img_AP and img_LAT folders should contain analogous images. If only AP or LAT images are introduced (not both), no matching between views will be done.')
    parser.add_argument("--csv", action='store', required=True, default=None, help="Comma-separated CSV containing information about the images. Possible columns: ['case_id','img_path_AP','img_path_LAT']")
    parser.add_argument("--seg_model", action='store', required=True, default=None, choices=['nnunet','medt','gatedaxialunet'], help="Model used for segmenting the lungs. Choose among: ['nnunet','medt','gatedaxialunet']")
    parser.add_argument("--output_folder", "-o", action='store', default=None, help="Output folder. Required.")
    parser.add_argument("--fformat", "-f", action='store', default=None, choices=['jpg','png','nifti'], help="JPG, PNG, NIFTI (.nii.gz) formats allowed: ['jpg','png','nifti']")
    args = parser.parse_args()

    # Flag to apply clahe?
    # Output regions in folders corresponding to each of the cases processed, not per image.

    csv_path = args.csv
    seg_model = args.seg_model
    output_folder = args.output_folder
    file_format_orig = args.fformat
    BASE_DIR = os.getcwd()

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
        'regions': os.path.join(output_folder, 'regions'), 
        'yolo_models': {
            'AP': os.path.join(BASE_DIR,'yolov5_weights','AP_pTB_yolov5_weights_v12112021.pt'),
            'LAT': os.path.join(BASE_DIR,'yolov5_weights','LAT_pTB_yolov5_weights_v09022022.pt')
        }
    }

    # File format definition
    file_format = file_format_orig
    if file_format_orig=='nifti': file_format='nii.gz'
    file_format = '.'+file_format

    # Run checks
    df, views = check_and_adapt_csv(csv_path,file_format)
    
    # Process
    # Steps 0-1: With a for loop
    for index, row in df.iterrows():
        # Info
        print(f"Processing case {index}: {row['case_id']}")
        # Get paths
        if('AP' in views):
            img_path_AP = row['img_path_AP']
        if('LAT' in views):
            img_path_LAT = row['img_path_LAT']
        # At this point, paths should be properly defined
        # Step 0. Read image(s)
        if(file_format_orig in ['jpg','png']):
            if('AP' in views):
                img_AP = imread(img_path_AP)
            if('LAT' in views):
                img_LAT = imread(img_path_LAT)
        elif(file_format_orig in ['nifti']):
            if('AP' in views):
                img_AP = read_nifti(img_path_AP)
            if('LAT' in views):
                img_LAT = read_nifti(img_path_LAT)
        # Step 1. Preprocess & save
        for view in views:
            if(view=='AP'):
                img = img_AP.copy()
            elif(view=='LAT'):
                img = img_LAT.copy()
            img_clahe = adapt_2d_and_apply_clahe(img)
            img_clahe_resize = resize(img_clahe,(512,512))
            img_clahe = cv2.normalize(img_clahe,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            img_clahe_resize = cv2.normalize(img_clahe_resize,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            maybe_make_dir(os.path.join(paths['preprocessed'],view))
            maybe_make_dir(os.path.join(paths['yolo_in'],view))
            if(view=='AP'):
                imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_AP)),img_clahe)
                imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_AP)),img_clahe_resize)
            elif(view=='LAT'):
                imsave(os.path.join(paths['preprocessed'],view,os.path.basename(img_path_LAT)),img_clahe)
                imsave(os.path.join(paths['yolo_in'],view,os.path.basename(img_path_LAT)),img_clahe_resize)
    # Step 2. Crop with YOLO (batch)
    os.chdir(os.path.join(BASE_DIR,'yolov5'))
    for view in views:
        YOLO_MODEL_PATH = paths['yolo_models'][view]
        DIM = 512
        CONF = 0.5
        IMG_DIR_FOR_YOLO = os.path.join(paths['yolo_in'],view)
        PROJECT_OUT = os.path.join(paths['yolo_out'])
        NAME = view
        MAX_DETECTIONS = 1
        os.system(f"python detect.py --weights {YOLO_MODEL_PATH} --img {DIM} --conf {CONF} --source {IMG_DIR_FOR_YOLO} --project {PROJECT_OUT} --name {NAME} --exist-ok --max-det {MAX_DETECTIONS} --save-txt --save-conf")
    ## 2.1. Restore base path
    os.chdir(BASE_DIR)
    ## Crop images with YOLOv5 predictions
    err_idx = []
    for index, row in df.iterrows():
        # Info
        print(f"Cropping case {index}: {row['case_id']}")
        # Get paths
        if('AP' in views):
            img_path_AP = row['img_path_AP']
        if('LAT' in views):
            img_path_LAT = row['img_path_LAT']
        # At this point, paths should be properly defined
        # Step 2.0. Read image(s)
        if(file_format_orig in ['jpg','png']):
            if('AP' in views):
                img_AP = imread(img_path_AP)
            if('LAT' in views):
                img_LAT = imread(img_path_LAT)
        elif(file_format_orig in ['nifti']):
            if('AP' in views):
                img_AP = read_nifti(img_path_AP)
            if('LAT' in views):
                img_LAT = read_nifti(img_path_LAT)
        # Step 2.1. Crop & save
        for view in views:
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
            maybe_make_dir(os.path.join(paths['cropped'],view))
            maybe_make_dir(os.path.join(paths['cropped_clahe'],view))
            if(file_format_orig in ['jpg','png']):
                imsave(os.path.join(paths['cropped'],view,os.path.basename(row[f"img_path_{view}"])),img_cropped)
                imsave(os.path.join(paths['cropped_clahe'],view,os.path.basename(row[f"img_path_{view}"])),img_cropped_clahe)
            if(file_format_orig in ['nifti']):
                convert_2d_image_array_to_nifti(img_cropped, os.path.join(paths['cropped'],view,os.path.basename(row[f"img_path_{view}"])[:-5]))
                convert_2d_image_array_to_nifti(img_cropped_clahe, os.path.join(paths['cropped_clahe'],view,os.path.basename(row[f"img_path_{view}"])[:-5]))   
        ## Step 2.2. Remove those cases which reported errors
        df = df.drop(err_idx)
    # Step 3. Segmentation with selected model (batch)
    if(seg_model=='nnunet'):
        for view in views:
            maybe_make_dir(os.path.join(paths['nnunet_in'],view))
            adapt_images_nnunet(folder_in=os.path.join(paths['cropped_clahe'],view),folder_out=os.path.join(paths['nnunet_in'],view),file_format=file_format)
            INPUT_FOLDER = os.path.join(paths['nnunet_in'],view)
            OUTPUT_FOLDER = os.path.join(paths['nnunet_out'],view)
            if(view=='AP'):
                os.system(f"nnUNet_predict -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -tr nnUNetTrainerV2_50epochs -m 2d -t 136")
            if(view=='LAT'):
                os.system(f"nnUNet_predict -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -tr nnUNetTrainerV2_50epochs -m 2d -t 135")
    # elif(seg_model=='medt'):
    #         adapt_images_medt(folder_in=os.path.join(paths('yolo_out'),view),folder_out=os.path.join(paths('medt_in'),view),file_format=file_format,resize_dim=256)
    #         INPUT_FOLDER = paths['medt_in']
    #         OUTPUT_FOLDER = paths['medt_out']
    #         os.system(f"python predict_medt-gatedaxialunet_20-40-60.py")
    # elif(seg_model=='gatedaxialunet'):
    #         adapt_images_medt(folder_in=os.path.join(paths('yolo_out'),view),folder_out=os.path.join(paths('gatedaxialunet_in'),view),file_format=file_format,resize_dim=256)
    #         INPUT_FOLDER = paths['gatedaxialunet_in']
    #         OUTPUT_FOLDER = paths['gatedaxialunet_out']
    #         os.system(f"python predict_medt-gatedaxialunet_20-40-60.py")
    # Step 4. Schema over the images
    for index,row in df.iterrows():
        if('AP' in views and 'LAT' in views):
            # 4.1. Paths
            img_path_AP = row['img_path_AP']
            img_path_LAT = row['img_path_LAT']
            if(seg_model=='nnunet'):
                lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*.nii.gz')) if row['case_id'] in f]
                lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*.nii.gz')) if row['case_id'] in f]
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*.png')) if row['case_id'] in f]
                lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*.png')) if row['case_id'] in f]
            if(lbl_path_AP and len(lbl_path_AP)==1):
                lbl_path_AP=lbl_path_AP[0]
            else:
                raise Exception(f"AP label for case {row['case_id']} could not be determined. Please check.")
            if(lbl_path_LAT and len(lbl_path_LAT)==1):
                lbl_path_LAT=lbl_path_LAT[0]
            else:
                raise Exception(f"LAT label for case {row['case_id']} could not be determined. Please check.") 
            # 4.2. Read data
            ## 4.2.1. Read images
            if(file_format_orig in ['jpg','png']):
                img_AP = imread(img_path_AP)
                img_LAT = imread(img_path_LAT)
            elif(file_format_orig in ['nifti']):
                img_AP = read_nifti(img_path_AP)
                img_LAT = read_nifti(img_path_LAT)
            # 4.2.2. Read labels
            if(seg_model=='nnunet'):
                lbl_AP = read_nifti(lbl_path_AP)
                lbl_LAT = read_nifti(lbl_path_LAT)
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_AP = imread(lbl_path_AP)
                lbl_LAT = imread(lbl_path_LAT)
            # 4.3. Get regions
            regions_AP, img_AP_rotated_draw, regions_LAT, img_LAT_rotated_draw = get_regions_final(img_AP,lbl_AP,img_LAT,lbl_LAT)
            # 4.4. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['fname_AP_without_ext']))
            maybe_make_dir(os.path.join(paths['regions'],row['fname_LAT_without_ext']))
            ## AP
            out_path = os.path.join(paths['regions'],row['fname_AP_without_ext'],'regions_AP.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_AP, fp, cls=NpEncoder)
            imsave(os.path.join(paths['regions'],row['fname_AP_without_ext'],'img_AP_regions.jpg'),img_AP_rotated_draw)
            ## LAT
            out_path = os.path.join(paths['regions'],row['fname_LAT_without_ext'],'regions_LAT.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_LAT, fp, cls=NpEncoder)
            imsave(os.path.join(paths['regions'],row['fname_LAT_without_ext'],'img_LAT_regions.jpg'),img_LAT_rotated_draw)
        elif('AP' in views):
            # 4.1. Paths
            img_path_AP = row['img_path_AP']
            lbl_path_AP = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'AP','*.nii.gz')) if row['case_id'] in f]
            if(lbl_path_AP and len(lbl_path_AP)==1):
                lbl_path_AP=lbl_path_AP[0]
            else:
                raise Exception(f"AP label for case {row['case_id']} could not be determined. Please check.")
            # 4.2. Read images
            ## 4.2.1. Read images
            if(file_format_orig in ['jpg','png']):
                img_AP = imread(img_path_AP)
            elif(file_format_orig in ['nifti']):
                img_AP = read_nifti(img_path_AP)
            # 4.2.2. Read labels
            if(seg_model=='nnunet'):
                lbl_AP = read_nifti(lbl_path_AP)
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_AP = imread(lbl_path_AP)
            # 4.3. Get regions
            regions_AP, img_AP_rotated_draw = get_regions_final_only_AP(img_AP,lbl_AP)
            # 4.4. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['fname_AP_without_ext']))
            out_path = os.path.join(paths['regions'],row['fname_AP_without_ext'],'regions_AP.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_AP, fp, cls=NpEncoder)
            imsave(os.path.join(paths['regions'],row['fname_AP_without_ext'],'img_AP_regions.jpg'),img_AP_rotated_draw)
        elif('LAT' in views):
            # 4.1. Paths
            img_path_LAT = row['img_path_LAT']
            lbl_path_LAT = [f for f in glob.glob(os.path.join(paths[f"{seg_model}_out"],'LAT','*.nii.gz')) if row['case_id'] in f]
            if(lbl_path_LAT and len(lbl_path_LAT)==1):
                lbl_path_LAT=lbl_path_LAT[0]
            else:
                raise Exception(f"LAT label for case {row['case_id']} could not be determined. Please check.") 
            # 4.2. Read images
            ## 4.2.1. Read images
            if(file_format_orig in ['jpg','png']):
                img_LAT = imread(img_path_LAT)
            elif(file_format_orig in ['nifti']):
                img_LAT = read_nifti(img_path_LAT)
            # 4.2.2. Read labels
            if(seg_model=='nnunet'):
                lbl_LAT = read_nifti(lbl_path_LAT)
            elif(seg_model=='medt' or seg_model=='gatedaxialunet'):
                lbl_LAT = imread(lbl_path_LAT)
            # 4.3. Get regions
            regions_LAT, img_LAT_rotated_draw = get_regions_final_only_LAT(img_LAT,lbl_LAT)
            # 4.4. Save results
            maybe_make_dir(os.path.join(paths['regions'],row['fname_LAT_without_ext']))
            out_path = os.path.join(paths['regions'],row['fname_LAT_without_ext'],'regions_LAT.json')
            with open(out_path, 'w') as fp:
                json.dump(regions_LAT, fp, cls=NpEncoder)
            imsave(os.path.join(paths['regions'],row['fname_LAT_without_ext'],'img_LAT_regions.jpg'),img_LAT_rotated_draw)
    # Step 5. Crop patches and save
    # (Step 5. In uncropped version)
    ## Save results
    # Read img and label

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
            raise Exception("Empty CSV. Please check.")
        if('case_id' not in df.keys()):
            raise Exception("'case_id' column not included in the CSV. Please check.")
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
    # If everything is OK, proceed with more checks.
    for index,row in df.iterrows():
        for view in views:
            f = row[f"img_path_{view}"]
            assert os.path.isfile(f), f"File {f} not found! Please check."
            # Create another column with fname_without_ext per view. Check format consistency and file existance.
            if(file_format in ['.jpg','.png']):
                ext = os.path.splitext(f)[-1]
                assert ext==file_format, f"File formats not matching. Please verify all files have the same file extension indicated in the command. \n File {f} not matching file extension {file_format}."
                df.loc[index,f"fname_{view}_without_ext"] = os.path.splitext(os.path.basename(f))[0]
            elif(file_format in ['.nii.gz']):
                ext = os.path.basename(f)[-7:]
                assert ext==file_format, f"File formats not matching. Please verify all files have the same file extension indicated in the command. \n File {f} not matching file extension {file_format}."
                df.loc[index,f"fname_{view}_without_ext"] = os.path.splitext(os.path.basename(f))[0][:-4]
        
    return df, views

def adapt_images_nnunet(folder_in,folder_out,file_format):
    subfiles = sorted(glob.glob(os.path.join(folder_in,'*'+file_format)))
    for f in subfiles:
        if(file_format in ['.jpg','.png']):
            output_filename_truncated = os.path.splitext(os.path.basename(f))[0]
            out_path_truncated = os.path.join(folder_out,output_filename_truncated)
            convert_2d_image_to_nifti(f, out_path_truncated)
        elif(file_format == '.nii.gz'):
            out_path = os.path.join(folder_out,os.path.basename(f))
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
        if(file_format in ['.jpg','.png']):
            out_path = os.path.join(folder_out,os.path.splitext(os.path.basename(f))[0]+'.png')
        elif(file_format == '.nii.gz'):
            out_path = os.path.join(folder_out,os.path.splitext(os.path.basename(f))[0][:-4]+'.png')
        imsave(out_path,img)

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