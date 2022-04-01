# -*- coding: utf-8 -*-
"""
@author: Daniel Capellán-Martín <daniel.capellan@upm.es>
"""

from math import *
from operator import itemgetter
import cv2
import numpy as np
from skimage.color import label2rgb,rgb2gray,gray2rgb

# Functions
def findRectangles(contours,img,threshold=0.05):
    '''
    Function for finding rectangles in image with specific conditions
    Input:
        contours: array of contours from cv2.findContours
        image: 2-D (not 3D stack) Image to which cv2.findContours is applied
        threshold: threshold for deleting little contributions
    Output:
        rectangles: Dict with coordinates (x,y,width,height) of the rectangles (keys: 0,1,2,3,...)
        rect_img: image with rectangles detected with black background
        out_img: image with rectangles detected in original image
    '''
    # Initialize variables
    h,w = img.shape
    rect_img = np.uint8(np.zeros(np.shape(img)+(3,)))
    out_img = gray2rgb(img)
    count = 0
    rectangles = {}
    for points in contours:
        if(cv2.contourArea(points)>np.round((threshold*w*h))):  # leave out all little contributions (with area lower than 5% of the whole image area (w*h))
            # Get rectangle and coordinates
            rect = cv2.minAreaRect(points)  # Rectangle and circle
            box = cv2.boxPoints(rect)  # Points of rectangle
            box = np.int0(box)
            # Info to dict
            rectangle = {
                'x': np.int0(np.min([i[0] for i in box])),  # X coordinate
                'y': np.int0(np.min([i[1] for i in box])), # Y coordinate
                'width': np.int0(np.max([i[0] for i in box]) - np.min([i[0] for i in box])),
                'height': np.int0(np.max([i[1] for i in box]) - np.min([i[1] for i in box])),
                'rotation': rect[2]
            }
            ###
            ((center_x,center_y), (width_rect, height_rect), theta) = rect
            # In order to match PCA, take height as highest value and width as lowest
            width = np.min([width_rect,height_rect])
            height = np.max([width_rect,height_rect])
            center = (center_x,center_y)
            rectangle = {
                'center_x': center_x,
                'center_y': center_y,
                'center': center,
                'width': width,
                'height': height
            }
            ## Get rotation angle
            box_contoured = np.expand_dims(box,axis=1)
            rotation_angle = getOrientation(box_contoured,img)
            rectangle['rotation'] = rotation_angle
            ###
            # Draw rectangles in image
#             box2 = np.array([[rectangle['x'],rectangle['y']],[rectangle['x'],rectangle['y']+rectangle['height']],[rectangle['x']+rectangle['width'],rectangle['y']+rectangle['height']],[rectangle['x']+rectangle['width'],rectangle['y']]])
            cv2.drawContours(rect_img,[box],0,(0,255,0),thickness=2)  # Draw rectangle in black image
#             cv2.drawContours(rect_img,[box2],0,(255,0),thickness=2)  # Draw rectangle in black image
            cv2.drawContours(out_img,[box],0,(0,255,0),thickness=2)  # Draw rectangle in original image
#             cv2.drawContours(out_img,[box2],0,(255,0),thickness=2)  # Draw rectangle in black image
            # Store results
            rectangles[count] = rectangle
            count+=1
    return rectangles, rect_img, out_img

def findRectangles_90deg(contours,img,threshold=0.05):
    '''
    Function for finding rectangles in image with specific conditions. Rectangles given are totally vertical (90 degrees with respect to horizontal axis)
    Input:
        contours: array of contours from cv2.findContours
        image: 2-D (not 3D stack) Image to which cv2.findContours is applied
        threshold: threshold for deleting little contributions
    Output:
        rectangles: Dict with coordinates (x,y,width,height) of the rectangles (keys: 0,1,2,3,...)
        rect_img: image with rectangles detected with black background
        out_img: image with rectangles detected in original image
    '''
    # Initialize variables
    h,w = img.shape
    rect_img = np.uint8(np.zeros(np.shape(img)+(3,)))
    out_img = gray2rgb(img)
    count = 0
    rectangles = {}
    # Adapt img to draw
    if(len(img.shape)<3):
        img = gray2rgb(img)
    for points in contours:
        if(cv2.contourArea(points)>np.round((threshold*w*h))):  # leave out all little contributions (with area lower than 5% of the whole image area (w*h))
            # Get vertical rectangle and coordinates
            bounding_rect = cv2.boundingRect(points)
            (x,y,w,h) = bounding_rect
            # Info to dict
            rectangle = {
                'x': x,  # X coordinate
                'y': y, # Y coordinate
                'width': w,
                'height': h,
                'rotation': None
            }
            ###
            # Draw rectangles in image
            rect_img = cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Store results
            rectangles[count] = rectangle
            count+=1
    return rectangles, rect_img, out_img

def draw_rectangle(image, centre, theta, width, height):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    # print(R)
    print(centre[0])
    p1 = [ + width / 2,  + height / 2]
    p2 = [- width / 2,  + height / 2]
    p3 = [ - width / 2, - height / 2]
    p4 = [ + width / 2,  - height / 2]
    p1_new = np.dot(p1, R)+ centre
    p2_new = np.dot(p2, R)+ centre
    p3_new = np.dot(p3, R)+ centre
    p4_new = np.dot(p4, R)+ centre
    print(p1_new)
    img = cv2.line(image, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p2_new[0, 0]), int(p2_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p3_new[0, 0]), int(p3_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p4_new[0, 0]), int(p4_new[0, 1])), (int(p1_new[0, 0]), int(p1_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), (255, 0, 0), 1)

    return img

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    '''source: https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/'''
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
  
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
  
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
  
    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
#     drawAxis(img, cntr, p1, (255, 255, 0), 1)
#     drawAxis(img, cntr, p2, (0, 0, 255), 5)
  
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]
  
    # Label with the rotation angle
#     label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
#     textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
#     cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
  
    return -int(np.rad2deg(angle)) + 90

def get_schema(lbl):
    # Initial variables
    h,w = lbl.shape
    # Step 1. Find contours
    contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Step 2. If 2 rectangles are detected, ok. Else, increment threshold for deleting little contributions and previosuly perform morphological opening with vertical bar (loop x3)
    rectangles, rect_lbl, out_lbl = findRectangles(contours,lbl)
    if(len(rectangles)!=2):
        for threshold, k in zip([0.1,0.25],[1,2]):
            # Try only with higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles(contours,lbl,threshold)
            ## Check
            if(len(rectangles)==2):
                break
            # Morphological operation
            se_h, se_w = int(np.round(2*k*0.1*h)), int(np.round(k*0.1*w))
            SE = cv2.getStructuringElement(cv2.MORPH_RECT,(se_w,se_h))
            lbl = cv2.morphologyEx(lbl, cv2.MORPH_OPEN, SE)
            # Try with opening & initial threshold
            contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Need to find contours another time
            rectangles, rect_lbl, out_lbl = findRectangles(contours,lbl)
            ##  Check
            if(len(rectangles)==2):
                break
            # Try with opening & higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles(contours,lbl,threshold)
            ##  Check
            if(len(rectangles)==2):
                break    
    # Step 3. Check there are two rectangles only
    if(len(rectangles)!=2):
        print("Lungs were not detected! Please check the image!")
    # Step 4. Reorder rectangles so that 0 is right lung and 1 is left lung
    rectangles = {i: sorted(rectangles.values(), key=itemgetter('center_x'))[i] for i in list(rectangles.keys())}
    
    return rectangles, rect_lbl, out_lbl

def get_schema_AP_90deg(lbl):
    # Initial variables
    h,w = lbl.shape
    # Step 1. Find contours
    contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Step 2. If 2 rectangles are detected, ok. Else, increment threshold for deleting little contributions and previosuly perform morphological opening with vertical bar (loop x3)
    rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl)
    if(len(rectangles)!=2):
        for threshold, k in zip([0.1,0.25],[1,2]):
            # Try only with higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl,threshold)
            ## Check
            if(len(rectangles)==2):
                break
            # Morphological operation
            se_h, se_w = int(np.round(2*k*0.1*h)), int(np.round(k*0.1*w))
            SE = cv2.getStructuringElement(cv2.MORPH_RECT,(se_w,se_h))
            lbl = cv2.morphologyEx(lbl, cv2.MORPH_OPEN, SE)
            # Try with opening & initial threshold
            contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Need to find contours another time
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl)
            ##  Check
            if(len(rectangles)==2):
                break
            # Try with opening & higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl,threshold)
            ##  Check
            if(len(rectangles)==2):
                break    
    # Step 3. Check there are two rectangles only
    if(len(rectangles)!=2):
        print("Lungs were not detected! Please check the image!")
    # Step 4. Reorder rectangles so that 0 is right lung and 1 is left lung
    rectangles = {i: sorted(rectangles.values(), key=itemgetter('x'))[i] for i in list(rectangles.keys())}
    
    return rectangles, rect_lbl, out_lbl

def get_schema_LAT_90deg(lbl):
    '''
    Equal to AP, but only 1 region needed
    '''
    # Initial variables
    h,w = lbl.shape
    # Step 1. Find contours
    contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Step 2. If 2 rectangles are detected, ok. Else, increment threshold for deleting little contributions and previosuly perform morphological opening with vertical bar (loop x3)
    rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl)
    if(len(rectangles)!=1):
        for threshold, k in zip([0.1,0.25],[1,2]):
            # Try only with higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl,threshold)
            ## Check
            if(len(rectangles)==1):
                break
            # Morphological operation
            se_h, se_w = int(np.round(2*k*0.1*h)), int(np.round(k*0.1*w))
            SE = cv2.getStructuringElement(cv2.MORPH_RECT,(se_w,se_h))
            lbl = cv2.morphologyEx(lbl, cv2.MORPH_OPEN, SE)
            # Try with opening & initial threshold
            contours, hierarchy = cv2.findContours(lbl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Need to find contours another time
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl)
            ##  Check
            if(len(rectangles)==1):
                break
            # Try with opening & higher threshold
            rectangles, rect_lbl, out_lbl = findRectangles_90deg(contours,lbl,threshold)
            ##  Check
            if(len(rectangles)==1):
                break    
    # Step 3. Check there are two rectangles only
    if(len(rectangles)!=1):
        print("Lungs were not detected! Please check the image!")
    
    return rectangles, rect_lbl, out_lbl

def subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad
    
    # Make width and height integers if they are not
    width = int(np.round(width))
    height = int(np.round(height))

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)

def rotate_image(image, theta):
    theta *= 3.14159 / 180 # convert to rad
    
    # Make width and height integers if they are not
    width = image.shape[1]
    height = image.shape[0]
    
    center = (int(np.round(width/2)),int(np.round(height/2)))

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)

def abs2rel(abs_dict,img):
    '''
    inputs:
        abs_dict: dict - dictionary containing, at least: {'x','y','width','height'}
        img: image - where to extract shape/dimensions
    outputs: relative coordinates:
        rel_dict - dictionary with the form {'x','y','width','height'}
    '''
    h,w = img.shape[0:2]
    rel_dict = {
        'x': abs_dict['x']/w,
        'y': abs_dict['y']/h,
        'width': abs_dict['width']/w,
        'height': abs_dict['height']/h,
    }
    return rel_dict

def get_regions_AP(img_AP_rotated,img_AP_rotated_reg,rectangles_AP_90deg):
    # Adapt input
    regions = {
        'abs': {},
        'rel': {}
    }
    regions['abs']['right_lung'] = rectangles_AP_90deg[0]
    regions['abs']['left_lung'] = rectangles_AP_90deg[1]
    # Img already rotated -> to RGB
    if(len(img_AP_rotated.shape)==2):
        img_AP_rotated_draw = gray2rgb(img_AP_rotated_reg)
    else:
        img_AP_rotated_draw = img_AP_rotated_reg.copy()
    # 0. Same Y coordinate and height for both lungs: correct misalignment between lungs
    min_y_lungs = np.min([regions['abs']['right_lung']['y'],regions['abs']['left_lung']['y']])
    max_y_lungs = np.max([regions['abs']['right_lung']['y']+regions['abs']['right_lung']['height'],regions['abs']['left_lung']['y']+regions['abs']['left_lung']['height']])
    max_height_lungs = max_y_lungs - min_y_lungs
    regions['abs']['right_lung']['y'] = min_y_lungs  # Align top coordinate for both lungs
    regions['abs']['left_lung']['y'] = min_y_lungs  # Align top coordinate for both lungs
    regions['abs']['right_lung']['height'] = max_height_lungs
    regions['abs']['left_lung']['height'] = max_height_lungs
    ## Relative coordinates with respect to image shape
    regions['rel']['right_lung'] = abs2rel(regions['abs']['right_lung'], img_AP_rotated)
    regions['rel']['left_lung'] = abs2rel(regions['abs']['left_lung'], img_AP_rotated)
    # 1. Upper patch
    upper_patch_x = regions['abs']['right_lung']['x'] + int(np.round(0.5*regions['abs']['right_lung']['width']))
    upper_patch_y = np.min([regions['abs']['right_lung']['y'],regions['abs']['left_lung']['y']])
    upper_patch_width = (regions['abs']['left_lung']['x'] + int(np.round(0.5*regions['abs']['left_lung']['width']))) - upper_patch_x
    upper_patch_heigth = int(np.round(0.5*max_height_lungs))
    regions['abs']['upper_patch'] = {
        'x': upper_patch_x,
        'y': upper_patch_y,
        'width': upper_patch_width,
        'height': upper_patch_heigth
    }
    ## Relative coordinates with respect to image shape
    regions['rel']['upper_patch'] = abs2rel(regions['abs']['upper_patch'], img_AP_rotated)
    ## Draw on image
    # cv2.rectangle(img_AP_rotated_draw, (regions['abs']['upper_patch']['x'], regions['abs']['upper_patch']['y']), (regions['abs']['upper_patch']['x'] + regions['abs']['upper_patch']['width'], regions['abs']['upper_patch']['y'] + regions['abs']['upper_patch']['height']), (255, 153, 51), 6)  # Absolute coordinates
    p0 = (int(np.round(regions['rel']['upper_patch']['x']*img_AP_rotated_draw.shape[1])), int(np.round(regions['rel']['upper_patch']['y']*img_AP_rotated_draw.shape[0])))
    p1 = (int(np.round((regions['rel']['upper_patch']['x'] + regions['rel']['upper_patch']['width'])*img_AP_rotated_draw.shape[1])), int(np.round((regions['rel']['upper_patch']['y'] + regions['rel']['upper_patch']['height'])*img_AP_rotated_draw.shape[0])))
    cv2.rectangle(img_AP_rotated_draw, p0, p1, (255, 153, 51), 6)  # Relative coordinates
    # 2. Middle patch
    max_height_lungs = np.max([regions['abs']['right_lung']['height'],regions['abs']['left_lung']['height']])
    middle_patch_x = regions['abs']['right_lung']['x'] + int(np.round(0.4*regions['abs']['right_lung']['width']))
    middle_patch_y = np.min([regions['abs']['right_lung']['y'],regions['abs']['left_lung']['y']]) + int(np.round(0.2*max_height_lungs))
    middle_patch_width = (regions['abs']['left_lung']['x'] + int(np.round(0.6*regions['abs']['left_lung']['width']))) - middle_patch_x
    middle_patch_heigth = int(np.round(0.5*max_height_lungs))
    regions['abs']['middle_patch'] = {
        'x': middle_patch_x,
        'y': middle_patch_y,
        'width': middle_patch_width,
        'height': middle_patch_heigth
    }
    ## Relative coordinates with respect to image shape
    regions['rel']['middle_patch'] = abs2rel(regions['abs']['middle_patch'], img_AP_rotated)
    ## Draw on image
    # cv2.rectangle(img_AP_rotated_draw, (regions['abs']['middle_patch']['x'], regions['abs']['middle_patch']['y']), (regions['abs']['middle_patch']['x'] + regions['abs']['middle_patch']['width'], regions['abs']['middle_patch']['y'] + regions['abs']['middle_patch']['height']), (0, 153, 255), 6)  # Absoulte coordinates
    p0 = (int(np.round(regions['rel']['middle_patch']['x']*img_AP_rotated_draw.shape[1])), int(np.round(regions['rel']['middle_patch']['y']*img_AP_rotated_draw.shape[0])))
    p1 = (int(np.round((regions['rel']['middle_patch']['x'] + regions['rel']['middle_patch']['width'])*img_AP_rotated_draw.shape[1])), int(np.round((regions['rel']['middle_patch']['y'] + regions['rel']['middle_patch']['height'])*img_AP_rotated_draw.shape[0])))
    cv2.rectangle(img_AP_rotated_draw, p0, p1, (0, 153, 255), 6)  # Relative coordinates
    # 3. Thirds
    for side in ['right','left']:
        regions['abs'][f"{side}_lung"]['thirds'] = {}
        regions['rel'][f"{side}_lung"]['thirds'] = {}
        for i in [0,1,2]:
            height_third = int(np.round((1/3)*regions['abs'][f"{side}_lung"]['height']))
            regions['abs'][f"{side}_lung"]['thirds'][i] = {
                'x': regions['abs'][f"{side}_lung"]['x'],
                'y': regions['abs'][f"{side}_lung"]['y']+height_third*i,
                'width': regions['abs'][f"{side}_lung"]['width'],
                'height': height_third
            }
            ## Relative coordinates with respect to image shape
            regions['rel'][f"{side}_lung"]['thirds'][i] = abs2rel(regions['abs'][f"{side}_lung"]['thirds'][i], img_AP_rotated)
            ## Draw on image
            # cv2.rectangle(img_AP_rotated_draw, (regions['abs'][f"{side}_lung"]['thirds'][i]['x'], regions['abs'][f"{side}_lung"]['thirds'][i]['y']), (regions['abs'][f"{side}_lung"]['thirds'][i]['x'] + regions['abs'][f"{side}_lung"]['thirds'][i]['width'], regions['abs'][f"{side}_lung"]['thirds'][i]['y'] + regions['abs'][f"{side}_lung"]['thirds'][i]['height']), (204, 0, 204), 6)  # Absoulte coordinates
            p0 = (int(np.round(regions['rel'][f"{side}_lung"]['thirds'][i]['x']*img_AP_rotated_draw.shape[1])), int(np.round(regions['rel'][f"{side}_lung"]['thirds'][i]['y']*img_AP_rotated_draw.shape[0])))
            p1 = (int(np.round((regions['rel'][f"{side}_lung"]['thirds'][i]['x'] + regions['rel'][f"{side}_lung"]['thirds'][i]['width'])*img_AP_rotated_draw.shape[1])), int(np.round((regions['rel'][f"{side}_lung"]['thirds'][i]['y'] + regions['rel'][f"{side}_lung"]['thirds'][i]['height'])*img_AP_rotated_draw.shape[0])))
            cv2.rectangle(img_AP_rotated_draw, p0, p1, (204, 0, 204), 6)  # Relative coordinates

    return regions, img_AP_rotated_draw

def get_regions_LAT(img_LAT_rotated,img_LAT_rotated_reg,rectangles_LAT_90deg,regions_AP):
    '''
    Originally, LAT images don't need to be rotated
    '''
    # Adapt input
    regions = {
        'abs': {},
        'rel': {}
    }
    regions['abs']['lungs'] = rectangles_LAT_90deg[0]
    ## Relative coordinates with respect to image shape
    regions['rel']['lungs'] = abs2rel(rectangles_LAT_90deg[0], img_LAT_rotated)
    # Img already rotated -> to RGB
    if(len(img_LAT_rotated.shape)==2):
        img_LAT_rotated_draw = gray2rgb(img_LAT_rotated_reg)
    else:
        img_LAT_rotated_draw = img_LAT_rotated_reg.copy()
    # 1. Get abs charactersitics from AP Lungs
    abs_y_lungs_AP = np.min([regions_AP['abs']['right_lung']['y'],regions_AP['abs']['left_lung']['y']])
    abs_height_lungs_AP = np.max([regions_AP['abs']['right_lung']['height'],regions_AP['abs']['left_lung']['height']])
    relative_vertical_coordinates_middle_patch_AP = {
        'rel_y': (regions_AP['abs']['middle_patch']['y']-abs_y_lungs_AP)/abs_height_lungs_AP,
        'rel_height': regions_AP['abs']['middle_patch']['height']/abs_height_lungs_AP
    }
    # 2. Middle patch
    regions['abs']['middle_patch'] = {
        'x': rectangles_LAT_90deg[0]['x']+int(np.round(rectangles_LAT_90deg[0]['width']*((1/4)+0.05))),  # 5% added - more posterior
        'y': rectangles_LAT_90deg[0]['y']+int(np.round(relative_vertical_coordinates_middle_patch_AP['rel_y']*rectangles_LAT_90deg[0]['height'])),
        'width': int(np.round(rectangles_LAT_90deg[0]['width']*(2/4))),
        'height': int(np.round(rectangles_LAT_90deg[0]['height']*relative_vertical_coordinates_middle_patch_AP['rel_height'])),
    }
    ## Relative coordinates with respect to image shape
    regions['rel']['middle_patch'] = abs2rel(regions['abs']['middle_patch'], img_LAT_rotated)
    ## Draw on image
    # cv2.rectangle(img_LAT_rotated_draw, (regions['abs']['middle_patch']['x'], regions['abs']['middle_patch']['y']), (regions['abs']['middle_patch']['x'] + regions['abs']['middle_patch']['width'], regions['abs']['middle_patch']['y'] + regions['abs']['middle_patch']['height']), (0, 153, 255), 6)  # Absolute coordinates
    p0 = (int(np.round(regions['rel']['middle_patch']['x']*img_LAT_rotated_draw.shape[1])), int(np.round(regions['rel']['middle_patch']['y']*img_LAT_rotated_draw.shape[0])))
    p1 = (int(np.round((regions['rel']['middle_patch']['x'] + regions['rel']['middle_patch']['width'])*img_LAT_rotated_draw.shape[1])), int(np.round((regions['rel']['middle_patch']['y'] + regions['rel']['middle_patch']['height'])*img_LAT_rotated_draw.shape[0])))
    cv2.rectangle(img_LAT_rotated_draw, p0, p1, (0, 153, 255), 6)  # Relative coordinates
    # 3. Thirds
    regions['abs']['lungs']['thirds'] = {}
    regions['rel']['lungs']['thirds'] = {}
    for i in [0,1,2]:
        height_third = int(np.round((1/3)*regions['abs']['lungs']['height']))
        regions['abs']['lungs']['thirds'][i] = {
            'x': regions['abs']['lungs']['x'],
            'y': regions['abs']['lungs']['y']+height_third*i,
            'width': regions['abs']['lungs']['width'],
            'height': height_third
        }
        ## Relative coordinates with respect to image shape
        regions['rel']['lungs']['thirds'][i] = abs2rel(regions['abs']['lungs']['thirds'][i], img_LAT_rotated)
        ## Draw on image
        # cv2.rectangle(img_LAT_rotated_draw, (regions['abs']['lungs']['thirds'][i]['x'], regions['abs']['lungs']['thirds'][i]['y']), (regions['abs']['lungs']['thirds'][i]['x'] + regions['abs']['lungs']['thirds'][i]['width'], regions['abs']['lungs']['thirds'][i]['y'] + regions['abs']['lungs']['thirds'][i]['height']), (204, 0, 204), 6)  # Absolute coordinates
        p0 = (int(np.round(regions['rel']['lungs']['thirds'][i]['x']*img_LAT_rotated_draw.shape[1])), int(np.round(regions['rel']['lungs']['thirds'][i]['y']*img_LAT_rotated_draw.shape[0])))
        p1 = (int(np.round((regions['rel']['lungs']['thirds'][i]['x'] + regions['rel']['lungs']['thirds'][i]['width'])*img_LAT_rotated_draw.shape[1])), int(np.round((regions['rel']['lungs']['thirds'][i]['y'] + regions['rel']['lungs']['thirds'][i]['height'])*img_LAT_rotated_draw.shape[0])))
        cv2.rectangle(img_LAT_rotated_draw, p0, p1, (204, 0, 204), 6)  # Relative coordinates
    
    return regions, img_LAT_rotated_draw

def get_regions_LAT_without_AP(img_LAT_rotated,img_LAT_rotated_reg,rectangles_LAT_90deg):
    '''
    Originally, LAT images don't need to be rotated
    '''
    # Adapt input
    regions = {
        'abs': {},
        'rel': {}
    }
    regions['abs']['lungs'] = rectangles_LAT_90deg[0]
    # Img already rotated -> to RGB
    if(len(img_LAT_rotated.shape)==2):
        img_LAT_rotated_draw = gray2rgb(img_LAT_rotated_reg)
    else:
        img_LAT_rotated_draw = img_LAT_rotated_reg.copy()
    # 1. Get absolute charactersitics from AP Lungs
    relative_vertical_coordinates_middle_patch_AP = {
        'rel_y': 0.2,  # Manual - no correspondence between AP and LAT views
        'rel_height': 0.5  # Manual - no correspondence between AP and LAT views
    }
    # 2. Middle patch
    regions['abs']['middle_patch'] = {
        'x': rectangles_LAT_90deg[0]['x']+int(np.round(rectangles_LAT_90deg[0]['width']*(1/4))),
        'y': rectangles_LAT_90deg[0]['y']+int(np.round(relative_vertical_coordinates_middle_patch_AP['rel_y']*rectangles_LAT_90deg[0]['height'])),
        'width': int(np.round(rectangles_LAT_90deg[0]['width']*(2/4))),
        'height': int(np.round(rectangles_LAT_90deg[0]['height']*relative_vertical_coordinates_middle_patch_AP['rel_height'])),
    }
    ## Relative coordinates with respect to image shape
    regions['rel']['middle_patch'] = abs2rel(regions['abs']['middle_patch'], img_LAT_rotated)
    ## Draw on image
    # cv2.rectangle(img_LAT_rotated_draw, (regions['abs']['middle_patch']['x'], regions['abs']['middle_patch']['y']), (regions['abs']['middle_patch']['x'] + regions['abs']['middle_patch']['width'], regions['abs']['middle_patch']['y'] + regions['abs']['middle_patch']['height']), (0, 153, 255), 6)  # Absolute coordinates
    p0 = (int(np.round(regions['rel']['middle_patch']['x']*img_LAT_rotated_draw.shape[1])), int(np.round(regions['rel']['middle_patch']['y']*img_LAT_rotated_draw.shape[0])))
    p1 = (int(np.round((regions['rel']['middle_patch']['x'] + regions['rel']['middle_patch']['width'])*img_LAT_rotated_draw.shape[1])), int(np.round((regions['rel']['middle_patch']['y'] + regions['rel']['middle_patch']['height'])*img_LAT_rotated_draw.shape[0])))
    cv2.rectangle(img_LAT_rotated_draw, p0, p1, (0, 153, 255), 6)  # Relative coordinates
    # 3. Thirds
    regions['abs']['lungs']['thirds'] = {}
    regions['rel']['lungs']['thirds'] = {}
    for i in [0,1,2]:
        height_third = int(np.round((1/3)*regions['abs']['lungs']['height']))
        regions['abs']['lungs']['thirds'][i] = {
            'x': regions['abs']['lungs']['x'],
            'y': regions['abs']['lungs']['y']+height_third*i,
            'width': regions['abs']['lungs']['width'],
            'height': height_third
        }
        ## Relative coordinates with respect to image shape
        regions['rel']['lungs']['thirds'][i] = abs2rel(regions['abs']['lungs']['thirds'][i], img_LAT_rotated)
        ## Draw on image
        # cv2.rectangle(img_LAT_rotated_draw, (regions['abs']['lungs']['thirds'][i]['x'], regions['abs']['lungs']['thirds'][i]['y']), (regions['abs']['lungs']['thirds'][i]['x'] + regions['abs']['lungs']['thirds'][i]['width'], regions['abs']['lungs']['thirds'][i]['y'] + regions['abs']['lungs']['thirds'][i]['height']), (204, 0, 204), 6)  # Absolute coordinates
        p0 = (int(np.round(regions['rel']['lungs']['thirds'][i]['x']*img_LAT_rotated_draw.shape[1])), int(np.round(regions['rel']['lungs']['thirds'][i]['y']*img_LAT_rotated_draw.shape[0])))
        p1 = (int(np.round((regions['rel']['lungs']['thirds'][i]['x'] + regions['rel']['lungs']['thirds'][i]['width'])*img_LAT_rotated_draw.shape[1])), int(np.round((regions['rel']['lungs']['thirds'][i]['y'] + regions['rel']['lungs']['thirds'][i]['height'])*img_LAT_rotated_draw.shape[0])))
        cv2.rectangle(img_LAT_rotated_draw, p0, p1, (204, 0, 204), 6)  # Relative coordinates

    return regions['abs'], img_LAT_rotated_draw

def get_regions_final(img_AP,lbl_AP,img_LAT,lbl_LAT,img_AP_reg,img_LAT_reg):
    '''Correspondence between AP and LAT views'''
    # 1. AP
    ## 1.1. Get initial schema
    rectangles_AP, rect_lbl_AP, out_lbl_AP = get_schema(lbl_AP)
    ## 1.2. Rotate image and label
    theta_rotation_image_AP = np.mean([rectangles_AP[key]['rotation'] for key in rectangles_AP]) 
    img_AP_rotated = rotate_image(img_AP, -(theta_rotation_image_AP))
    img_AP_rotated_reg = rotate_image(img_AP_reg, -(theta_rotation_image_AP))
    lbl_AP_rotated = rotate_image(lbl_AP, -(theta_rotation_image_AP))
    ## 1.3. Get vertical rectangles from rotated label
    rectangles_AP_90deg, rect_lbl_AP_90deg, out_lbl_AP_90deg = get_schema_AP_90deg(lbl_AP_rotated)
    ## 1.4. Get regions from AP image
    regions_AP, img_AP_rotated_draw = get_regions_AP(img_AP_rotated,img_AP_rotated_reg,rectangles_AP_90deg)
    regions_AP['image_rotation'] = theta_rotation_image_AP  # Rotation in degrees
    # 2. LAT
    ## 2.1. Get vertical rectangles from LAT label (not rotation needed)
    rectangles_LAT_90deg, rect_lbl_LAT_90deg, out_lbl_LAT_90deg = get_schema_LAT_90deg(lbl_LAT)
    ## 2.2. Get regions from LAT image
    regions_LAT, img_LAT_rotated_draw = get_regions_LAT(img_LAT,img_LAT_reg,rectangles_LAT_90deg,regions_AP)
    return regions_AP, img_AP_rotated_draw, regions_LAT, img_LAT_rotated_draw

def get_regions_final_only_AP(img_AP,lbl_AP,img_AP_reg):
    '''Warning: If only AP or LAT, results will have no correspondence between AP and LAT views'''
    # 1. AP
    ## 1.1. Get initial schema
    rectangles_AP, rect_lbl_AP, out_lbl_AP = get_schema(lbl_AP)
    ## 1.2. Rotate image and label
    theta_rotation_image_AP = np.mean([rectangles_AP[key]['rotation'] for key in rectangles_AP]) 
    img_AP_rotated = rotate_image(img_AP, -(theta_rotation_image_AP))
    img_AP_rotated_reg = rotate_image(img_AP_reg, -(theta_rotation_image_AP))
    lbl_AP_rotated = rotate_image(lbl_AP, -(theta_rotation_image_AP))
    ## 1.3. Get vertical rectangles from rotated label
    rectangles_AP_90deg, rect_lbl_AP_90deg, out_lbl_AP_90deg = get_schema_AP_90deg(lbl_AP_rotated)
    ## 1.4. Get regions from AP image
    regions_AP, img_AP_rotated_draw = get_regions_AP(img_AP_rotated,img_AP_rotated_reg,rectangles_AP_90deg)
    regions_AP['image_rotation'] = theta_rotation_image_AP  # Rotation in degrees
    return regions_AP, img_AP_rotated_draw

def get_regions_final_only_LAT(img_LAT,lbl_LAT,img_LAT_reg):
    '''Warning: If only AP or LAT, results will have no correspondence between AP and LAT views'''
    # 2. LAT
    ## 2.1. Get vertical rectangles from LAT label (not rotation needed)
    rectangles_LAT_90deg, rect_lbl_LAT_90deg, out_lbl_LAT_90deg = get_schema_LAT_90deg(lbl_LAT)
    ## 2.2. Get regions from LAT image
    regions_LAT, img_LAT_rotated_draw = get_regions_LAT_without_AP(img_LAT,img_LAT_reg,rectangles_LAT_90deg)
    return regions_LAT, img_LAT_rotated_draw