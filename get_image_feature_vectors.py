#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For saving 'feature vectors' into a txt file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path
import os

from PIL import Image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import ImageDataGenerator


'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator


import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
#from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
'''


#################################################



#################################################
# This function:
# Loads the JPEG image at the given path
# Decodes the JPEG image to a uint8 W X H X 3 tensor
# Resizes the image to 224 x 224 x 3 tensor
# Returns the pre processed image as 224 x 224 x 3 tensor
#################################################
def load_image(imagePath):
    #image = load_img(imagePath, color_mode = "grayscale")
    #image_array = img_to_array(image)
    #file_name = imagePath.split(os.path.sep)[-1].split(".")[0]
    #file_name = imagePath.split(os.path.sep)[-1]
    #file_name = os.path.basename(imagePath).split('.')[0]
    #file_name = os.path.basename(imagePath)

    #head, file_name = os.path.split(imagePath)
    #print("file name: ")
    #print(file_name)
    #new_path = file_name + "_grayscale.jpeg"
    #save_img(new_path, image_array)
    #image_pil = array_to_img(image_array)
    
    # Reads the image file and returns data type of string
    #img = tf.io.read_file(new_path)
    img = tf.io.read_file(imagePath)
    # Convert image to grayscale

    # Decodes the image to W x H x 3 shape tensor with type of uint8
    #img = tf.io.decode_jpeg(img, channels=3)
    img = tf.io.decode_image(img, channels=3)
    

    # Resize the image to 224 x 244 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)
    #img = tf.image.rgb_to_grayscale(img).numpy()
  
    # Converts the data type of uint8 to float32 by adding a new axis
    # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
    # This is required for the mobilenet model we are using
    img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    return img

#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_feature_vectors():

  i = 0

  start_time = time.time()

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
  print("---------------------------------")

  # Definition of module with using tfhub.dev handle
  module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
  
  # Load the module
  module = hub.load(module_handle)

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors -  Started at %s" %time.ctime())
 

  # Loops through all images in a local folder
  for filename in glob.glob('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa/*.jpeg'):
    i = i + 1

    print("-----------------------------------------------------------------------------------------")
    print("Image count                     :%s" %i)
    print("Image in process is             :%s" %filename)

    # Loads and pre-process the image
    img = load_img(filename)

    # Calculate the image feature vector of the img
    features = module(img)   
  
    # Remove single-dimensional entries from the 'features' array
    feature_set = np.squeeze(features)  

    # Saves the image feature vectors into a file for later use

    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    #out_path = os.path.join('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/feature-vectors', outfile_name)
    out_path = os.path.join('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/feature-vectors-grayscale', outfile_name)

    # Saves the 'feature_set' to a text file
    np.savetxt(out_path, feature_set, delimiter=',')

    print("Image feature vector saved to   :%s" %out_path)
  
  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors - Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  print("--- %s images processed ---------" %i)
  

def get_feature_vector(imagePath):
    # Definition of module with using tfhub.dev handle
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
  
    # Load the module
    module = hub.load(module_handle)

    #filename = imagePath.split(os.path.sep)[-1].split(".")[0]
    #filename = imagePath.split(os.path.sep)[-1].split(".")[0]
    # Loads and pre-process the image
    img = load_image(imagePath)

    # Calculate the image feature vector of the img
    features = module(img)   
  
    # Remove single-dimensional entries from the 'features' array
    feature_set = np.squeeze(features)  
    return feature_set
    # Saves the image feature vectors into a file for later use
    
    '''
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path = os.path.join('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/feature-vectors', outfile_name)

    # Saves the 'feature_set' to a text file
    np.savetxt(out_path, feature_set, delimiter=',')
    '''
    
#get_feature_vectors()
