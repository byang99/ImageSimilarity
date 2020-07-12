# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:27:55 2020

@author: Brian
"""

#from cluster_image_feature_vectors import PredictSofaPrices
#from get_image_feature_vectors import load_img, get_feature_vectors

import cluster_image_feature_vectors as c
import get_image_feature_vectors as g

# Glob for reading file names in a folder
import glob
import os.path

import time
from decimal import Decimal


def getAccuracy():
    start_time = time.time()
    model = c.PredictSofaPrices()
    model.loadTree('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/trees.ann')
    total = 0;
    count = 0;
    cc = pd.read_csv('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa.csv')
    price = list(cc['price'])
        
    for imagePath in glob.glob('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa/*.jpeg'):
        if count == 100:
            break;
        count += 1
        print("Count: ", count)
        predicted_price = model.predict_price(imagePath)
        a_price = price[int(imagePath.split(os.path.sep)[-1].split(".")[0]) - 1]
        actual_price = Decimal(a_price.replace('$','').replace(',',''))
        print("Actual Price: ", actual_price)
        if abs((predicted_price - actual_price) / actual_price) < 0.10:
            total += 1
        
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

    
    accuracy = total / count
    return accuracy
    
        
    
def main():
    model = c.PredictSofaPrices()
    model.loadTree('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/trees.ann')
    #model.cluster()
    print("Tree loading complete")
    print("Predicting price...")
    path = 'C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofaTest/3.jpg'
    model.predict_price(path)
    
    #print("Calculating model % accuracy - price error is +/- 10%")
    #print("Accuracy: ", getAccuracy())
        
if __name__ == "__main__":
    main()