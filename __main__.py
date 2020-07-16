# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:27:55 2020

@author: Brian
"""

#for batch processing
from threading import Thread
from batch_processor import BatchProcessor


#from cluster_image_feature_vectors import PredictSofaPrices
#from get_image_feature_vectors import load_img, get_feature_vectors

import cluster_image_feature_vectors as c
import get_image_feature_vectors as g

# Glob for reading file names in a folder
import glob
import os.path



import time
from decimal import Decimal

import pandas as pd


def encode(text):
    if type(text) is list or type(text) is tuple:
        return [x.encode('utf-8') for x in text]
    elif type(text) is not int:
        return text.encode('utf-8')
    else:
        return text

       
    
def getAccuracy():
    start_time = time.time()
    model = c.PredictSofaPrices()
    model.loadTree('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/trees.ann')
    total = 0;
    count = 0;
    cc = pd.read_csv('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa.csv')
    price = list(cc['price'])
    
    #counter variables
    index = 89
    counter = 2836
    batch_size = 32
    

    path = 'C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa/*.jpeg'
    files = glob.glob(path)
    total_files = len(files)
    with open('predictedPrices.csv', 'a') as file:
        #file.write("%s,%s,%s\n"%("Actual", "Predicted", "Error"))

        count = 2836
        while counter < total_files:
            print("PROCESSING BATCH ___________________", str(index + 1))
            
            #Set the range limit
            if total_files > counter + batch_size:
                range_limit = counter + batch_size
                
            else:
                range_limit = total_files - counter
            
            predictedPrices = []
            totalError = 0;
            actualPrices = []
            for x in range(counter, range_limit):
                try:
                    count += 1
                    imagePath = files[x]
                    predicted_price = model.predict_price(imagePath)
                    predictedPrices.append(predicted_price)
                    print("Count: ", count)
                    a_price = price[int(imagePath.split(os.path.sep)[-1].split(".")[0]) - 1]
                    actual_price = Decimal(a_price.replace('$','').replace(',',''))
                    actualPrices.append(actual_price)
                    print("Actual Price: ", actual_price)
                    
                    error = abs((predicted_price - actual_price) / actual_price)
                    if error < 0.10:
                        total += 1
                    
                    totalError += error
                    file.write("%s,%s,%s\n"%(actual_price,predicted_price,error))
                            
                    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
                except IndexError:
                    break
            
            counter += batch_size
            index += 1
    '''
    totalError = 0;
    predictedPrices = []
    actualPrices = []
    for imagePath in glob.glob('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa/*.jpeg'):
        count += 1
        print("Count: ", count)
        a_price = price[int(imagePath.split(os.path.sep)[-1].split(".")[0]) - 1]
        actual_price = Decimal(a_price.replace('$','').replace(',',''))
        actualPrices.append(actual_price)
        print("Actual Price: ", actual_price)
        
        error = abs((predicted_price - actual_price) / actual_price)
        if error < 0.10:
            total += 1
        
        totalError += error
        
        
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
    '''
    '''
    with open('predictedPrices.csv', 'w+') as file:
        file.write("%s,%s,%s\n"%("Actual", "Predicted", "Error"))
        for (actual, predicted) in zip(actualPrices, predictedPrices):
            error = abs((predicted - actual) / actual)
            file.write("%s,%s,%s\n"%(actual,predicted,error))
    '''
    avgError = totalError / 5779
    print("Average error: ", avgError)
    accuracy = total / count
    return accuracy
    
        
    
def main():
    model = c.PredictSofaPrices()
    model.loadTree('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/trees.ann')
    #model.cluster()
    print("Tree loading complete")
    print("Predicting price...")
    #path = 'C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofaTest/3.jpg'
    #model.predict_price(path)
    
    print("Calculating model % accuracy - price error is +/- 10%")
    print("Accuracy: ", getAccuracy())
        
if __name__ == "__main__":
    main()