import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

############possible alternative to the merging technique: define a 
             #circle with radius = h/w + threshold and center being center of image
             #then check if the extremities of the other box are in this radius.
             #If so, merge. If not, don't
             #have a special threshold for the dots.
def merge_near(boxes, hor_threshold=10, ver_threshold=10, merge_dots = False):
    """
    Takes a list of bounding boxes and merges those that are either fully 
    contained within other, within a certain radius of each other, 
    
    Parameters
    ----------
    boxes : list
        Original list of bounding boxes.

    hor_threshold : int, optional
        The horziontal threshold with which to consider two boxes near each 
        other. Basically, defines the horizontal radius. The default is 10.
    ver_threshold : int, optional
        The vertical threshold with which to consider two boxes near each 
        other. Basically, defined the vertical radius. The default is 10.
    merge_dots : bool, optional
        Flag to signal if we're merging dots usual shapes. This is done
        because dots need a different vertical threshold than the normal 
        shapes. The default is False.

    Returns
    -------
    list
        Updated list with merged boxes.

    """

    #Final list that is returned
    prev = boxes.copy()      
    
    changed = True
    merged_boxes = set({})
    
    #As long as changes are being made, keep on looping through the list
    while changed == True:              
        
        changed = False
        updated = []

        #If we are checking dots, filter the list for dots only
        if merge_dots:
            dots = [(x,y,w,h) for x,y,w,h in prev if w<140 and h<140]
            combinations = list(itertools.product(prev, dots))
        else:
            combinations = list(itertools.product(prev, prev))
            
        combinations = [e for e in combinations if e[0] != e[1]]
        
        #Loop through the boxes in the list -that includes the merged boxes-
        for pos, ((x1, y1, w1, h1), (x2, y2, w2, h2)) in enumerate(combinations):   
            
            #If a change was made, break out of the loop to do another iteration
            if changed == False:
                #Calculate different relative positions
                #You have 3 cases:  Fully contained within each other
                #                   Starts inside, ends outside / starts outside, ends inside
                #                   Starts outside, ends outside                                                
                #Note that we don't have the starting outside and ending inside because it is the same but different perspective, which we account for                                      
                
                #We want to merge if:   Fully contained 
                #                       Starts within ends outside, provided other dimension's difference is within threshold
                #                       Starts outside ends outside, provided both dimensions' differences are within threshold
                #Note that, while the difference might catch the starts inside, it is bounded by the threshold. So if the second rect starts within but is much larger, the difference in extremities will be > threshold. However, we still want to merge
                
                is_start_contained_hor = (x2 < x1 + w1 and x2 > x1)
                is_start_contained_ver = (y2 < y1 + h1 and y2 > y1)
                is_end_contained_hor = (x2 + w2 < x1 + w1 and x2 + w2 > x1)
                is_end_contained_ver = (y2 + h2 < y1 + h1 and y2 + h2> y1)
                is_contained = is_start_contained_hor and is_start_contained_ver and is_end_contained_hor and is_end_contained_ver
                are_extremities_threshold_hor = np.abs(x2 + w2 - x1) < hor_threshold or np.abs(x1 + w1 - x2) < hor_threshold or np.abs(x2-x1) < hor_threshold
                are_extremities_threshold_ver = np.abs(y2 + h2 - y1) < ver_threshold or np.abs(y1 + h1 - y2) < ver_threshold or np.abs(y2-y1) < ver_threshold
                
                if (is_contained or (
                        is_start_contained_hor and are_extremities_threshold_ver) or (
                            is_start_contained_ver and are_extremities_threshold_hor) or (
                                are_extremities_threshold_ver and are_extremities_threshold_hor)):
                    
                
                    
                    
                    #Note down the updated coordinates of the merged boxes
                    updated_x = min(x1,x2)
                    updated_y = min(y1,y2)
                    #Update the widths and height by getting final point - start point
                    updated_w = max(x1+w1, x2+w2) - updated_x
                    updated_h = max(y1+h1, y2+h2) - updated_y
                    
                    updated.append((updated_x, updated_y, updated_w, updated_h))
                    
                    merged_boxes.add((x1,y1,w1,h1))
                    merged_boxes.add((x2,y2,w2,h2))
                    
                    changed = True
                    
                    #Add all other boxes to updated for rerun
                    for box in boxes:
                        if box not in merged_boxes:
                            updated.append(box)
                    prev = updated
        
            else:
                break

    return prev if prev else boxes     

def get_bounding_box(input_dir, fname, output_directory="", merge=True):
    """
    Extracts the letter from an image by getting bounding boxes and resizes 
    the photo to 40x30x3 (RGB color stays)

    Parameters
    ----------
    input_dir : string
        Directory that contains the file to be analyzed (excluding the name of the file).
    fname : string
        The name of the file to be analyzed.
    output_directory : string, optional
        Directory where the resulting file should be saved. If not present,
        no file is saved. The default is "".
    merge : bool, optional
        Flag to decide whether to run the merge algorithm or not. The default 
        is True.

    Returns
    -------
    ndarray
        Original image with bounding boxes applied on it.
    letter_imgs : list
        List of cropped images from the bounding boxes.

    """
        
        
    image = cv2.imread(f"{input_dir}/{fname}")
    original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    bounding_rects = [cv2.boundingRect(c) for c in cnts]
    
    
    #Run the merge algorithm twice: once to catch the dots, and once for the others
    if merge:
        bounding_rects = merge_near(bounding_rects, hor_threshold=5, ver_threshold=100, merge_dots=True)    
        bounding_rects = merge_near(bounding_rects, hor_threshold=5, ver_threshold=10)
    
    
    letter_imgs = []
    
    for index, (x,y,w,h) in enumerate(bounding_rects):    
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
        
        
        t, imbw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ROI = original[y:y+h, x:x+w]
        imbw = imbw[y:y+h, x:x+w]
        ROI = cv2.resize(ROI, (28, 28), interpolation = cv2.INTER_CUBIC)[:,:,0]
        imbw = cv2.resize(imbw, (28, 28), interpolation = cv2.INTER_CUBIC)
        
        letter_imgs.append(imbw)

        if output_directory:
            ext_pos = fname.rfind(".")
            file_name = fname[:ext_pos] + f"_{index}" + fname[ext_pos:]
            cv2.imwrite(f'{output_directory}/{file_name}', imbw)
            
    
    return (image[:,:,::-1], letter_imgs)


def process_dataset(input_dir, output_dir):
    for i, file in enumerate(os.listdir(input_dir)):
        _, cropped = get_bounding_box(input_dir, file)
        cropped = cropped[0]
        cv2.imwrite(f'{output_dir}/{file}', cropped)
        
    
        
#process_dataset(r"C:\Users\Anton\Desktop\project 2\input\images", r"C:\Users\Anton\Desktop\project 2\input\updated_2")

"""
base_str = "img045-"
for i in range(35,47):
    full = base_str + f"0{i}.png"
    bounding, cropped = get_bounding_box("C:/Users/Anton/Desktop/project 2/input/images",full)
    plt.imshow(bounding)
    plt.show()
"""
#bounding, cropped = get_bounding_box(r"C:\Users\Anton\Desktop\project 2\src\test_images","test_11.png")
#plt.imshow(bounding)
