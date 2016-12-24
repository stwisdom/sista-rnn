import os
import cv2
import numpy as np

def ReadAndNormalizeImage(img_file, dim, color='y'):
    print("Loading", img_file)
    im = resize_and_crop_image(img_file, dim).astype(np.float32)
    #im[:,:,0] -= 103.939
    #im[:,:,1] -= 116.779
    #im[:,:,2] -= 123.68
    if (color=='y'):
        im=0.299*im[:,:,0]+0.587*im[:,:,1]+0.114*im[:,:,2]
    elif (color=='rgb'):
        im = im.transpose((2,0,1))
    im = im/255.0
    return im

def load_data(dim, dirname, color='y'):
    '''
    - dimension is a tuple like (224,224) corresponding to the height and the width
    - dirname is the main directory like "101_ObjectCategories"
    - color (optional) is either 'y' for intensity from YCrCb or 'rgb'
    '''
    X_data = [] # for keeping the image
    imgfiles = [] # for keeping the image file names
    y_data = [] # for keeping the image label id
    label_name = [] # for keeping the label dictionary
    label_cpt = 0 # Label id star at index 0
    for class_directory in os.listdir(dirname): # For each class directory
        if os.path.isdir(os.path.join(dirname, class_directory)): # if it's a directory
            for filename in os.listdir(os.path.join(dirname,class_directory)): # for echo file
                img_path = os.path.join(dirname, class_directory, filename) # full path of the file
                if img_path.endswith(".jpg"): # if it's an image -- you can add you own image extension
                    normalize_image = ReadAndNormalizeImage(img_path, dim)
                    X_data.append(normalize_image)
                    imgfiles.append(img_path)
                    y_data.append(label_cpt)
            label_name.append(class_directory)
            label_cpt += 1 # Label id incrementation
            # if label_cpt >= 10: break; # Uncomment to limit to 10 classes
    y_data = np.array(y_data)
    X_data = np.array(X_data, dtype=np.float32)
    label_name = np.array(label_name)
    return X_data, imgfiles, y_data, label_name

def resize_and_crop_image(img_file, dim):
    '''Takes an image path, crop in the center square if require and resize'''
    img = cv2.imread(img_file) # Read the image with open-cv
    height, width, depth = img.shape # Get the image information
    new_height = height
    new_width = width
    if height > width: # If the image is Landscape
        new_height = width
        height_middle = height / 2
        low_offset = height_middle - int(width/2)
        high_offset = height_middle + int(width/2)
        cropped_img = img[low_offset:high_offset , 0:width] # remove the horizontal sides
        resized_img = cv2.resize(cropped_img,(dim),interpolation=cv2.INTER_CUBIC) # and resize it
        return resized_img
    elif width > height: # If the image is portrait
        new_width = height
        width_middle = width / 2
        low_offset = width_middle - int(height/2)
        high_offset = width_middle + int(height/2)
        cropped_img = img[0:height , low_offset:high_offset] # remove the vertical sides
        resized_img = cv2.resize(cropped_img,(dim),interpolation=cv2.INTER_CUBIC) # resize it
        return resized_img
    else: # It's already a square, no need to crop
        resized_img = cv2.resize(img,(dim),interpolation=cv2.INTER_CUBIC) # just resize it!
        return resized_img

