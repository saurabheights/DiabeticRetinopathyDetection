# Partially adopted from https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py

import cv2, glob
import numpy as np
from os.path import join, exists
from os import listdir, makedirs


# function to scale the given image based on a scale value for the radius

def scaleRadius(img,scale):
    x=img[img.shape[0]//2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)


# function to preprocess images from a give folder and save them
#   (save_path_same) determines whether to use the same path for the given folder or save in the current relative path
#   (target_size) determines whether to resize the images after preprocessing to exact dimensions or not

def preprocess(folder='sample', scales=[300], save_path_same=True, target_size=(0,0)):
        
    for scale in scales:
        if (save_path_same):
            write_folder = folder+'_'+str(scale)
        else:
            write_folder = 'processed_'+str(scale)
        
        if not exists(write_folder):
            makedirs(write_folder)
            
        for f in listdir(folder):
            try:
                read_path = join(folder, f)
                a = cv2.imread(read_path)
                a = scaleRadius(a,scale)
                b = np.zeros(a.shape)
                cv2.circle(b,(a.shape[1]//2,a.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0)
                aa = cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
                
                if (target_size != (0,0)):
                    aa = cv2.resize(aa, target_size)
                boo = cv2.imwrite(join(write_folder, f), aa)

                print("Processed Image: ", f)
                print("Save Location: ", join(write_folder, f))
                print("Success: ", boo)
                print("New Dimensions: ", aa.shape[0], " X ", aa.shape[1])
                print("______________________________________________\n")
                
            except:
                print("Could not process file: ", f)
                
