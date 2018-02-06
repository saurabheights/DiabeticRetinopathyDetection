# Example Usage:
# >> python preprocess.py SOURCE_PATH
# target path is SOURCE_PATH+'_300'


# Partially adopted from https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py

import cv2
import numpy as np
from os.path import join, exists, basename
from os import listdir, makedirs
import sys
import logging
from datetime import datetime


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
    
    file = basename(folder)
    handlers = [logging.FileHandler(datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S-{file}.log")),
                logging.StreamHandler()]
    
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO, handlers=handlers)
    
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

                logging.info("Processed Image: "+ str(f))
                logging.info("Save Location: "+ str(join(write_folder, f)))
                logging.info("Success: "+str(boo))
                logging.info("New Dimensions: "+str(aa.shape[0])+" X "+str(aa.shape[1]))
                logging.info("______________________________________________\n")
                
            except:
                logging.info("Could not process file: "+str(f))
                

if __name__ == "__main__":
    source_path = sys.argv[1]
    same_path = eval(sys.argv[2])
    print("Reading Images from Directory: "+str(source_path))
    preprocess(source_path, save_path_same = same_path)
    print("DONE.")
          
    
        
