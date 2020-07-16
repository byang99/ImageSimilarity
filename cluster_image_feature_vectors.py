# -*- coding: utf-8 -*-
"""
Created on Thurs Jul  9  5:12 PM  2020

@author: Brian
"""


#################################################
# This script reads image feature vectors from a folder
# and saves the image similarity scores in json file
#################################################

#################################################
# Imports and function definitions
#################################################

#get feature vector of input image
import get_image_feature_vectors as g

#decimal to convert dollar amount to decimal number
from decimal import Decimal

#opencv for reading image file
import cv2

# Numpy for loading image feature vectors from file
import numpy as np

#pandas for extracting price info from csv file
import pandas as pd

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

# json for storing data in json file
import json

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial

#write to csv file
import csv

#statistics - median, mean
import statistics

#################################################
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imutils import paths
import imutils
import os
'''

class PredictSofaPrices(object):
    
    def __init__(self):
        # Configuring annoy parameters
        self.dims = 1792
        self.n_nearest_neighbors = 500
        self.trees = 10000
        
        self.t = AnnoyIndex(self.dims, metric = 'angular')
        
        # Defining data structures as empty dict
        self.file_index_to_file_name = {}
        self.file_index_to_file_vector = {}
        self.file_index_to_product_id = {}
        self.createDictionary("sofaPriceDictionary.csv")
        
    #################################################
    # This function reads from the sofa.csv file
    # to get the prices of each sofa in the training
    # dataset
    #################################################
    def match_price(self, filename):
        cc = pd.read_csv('C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/sofa.csv')
        price = list(cc['price'])
        
        return price[int(filename.split(".")[0]) - 1]
    #################################################
    
    #################################################
    #               CLUSTER()
    # Reads all image feature vectors stored in /feature-vectors/*.npz
    # Adds them all in Annoy Index
    # Builds ANNOY index
    #################################################
    
    
    def cluster(self):
        start_time = time.time()
        print("---------------------------------")
        print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
        print("---------------------------------")
    
        
        # Reads all file names which stores feature vectors 
        feature_vector_path = 'C:/Users/Brian/Desktop/ImageSimilarityDetection/ImageSimilarityDetection/feature-vectors/*.npz'  
        allfiles = glob.glob(feature_vector_path)
    
        for file_index, i in enumerate(allfiles):
            # Reads feature vectors and assigns them into the file_vector 
            file_vector = np.loadtxt(i)
            
            # Assigns file_name, feature_vectors and corresponding product_id
            file_name = os.path.basename(i).split('.')[0]
            self.file_index_to_file_name[file_index] = file_name
            self.file_index_to_file_vector[file_index] = file_vector
            self.file_index_to_product_id[file_index] = self.match_price(file_name)
            
            # Adds image feature vectors into annoy index   
            self.t.add_item(file_index, file_vector)
        
            print("---------------------------------")
            print("Annoy index     : %s" %file_index)
            print("Image file name : %s" %file_name)
            print("Product id      : %s" %self.file_index_to_product_id[file_index])
            print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
            
        # Builds annoy index
        #self.t.build(self.trees)
        
        #saves tree in a file for quick access later on
        #self.t.save('trees.ann')
        
        # opening the csv file in 'w+' mode          
        # writing the dictionary into the file 
        with open('sofaPrices.csv', 'w+') as file:
            for key in self.file_index_to_product_id.keys():
                price = Decimal(self.file_index_to_product_id[key].replace('$','').replace(',',''))
                file.write("%s,%s\n"%(key,price))
            
        '''
        print ("Step.1 - ANNOY index generation - Finished")
        print ("Step.2 - Similarity score calculation - Started ") 
        
        named_nearest_neighbors = []
    
        # Loops through all indexed items
        for i in file_index_to_file_name.keys():
            # Assigns master file_name, image feature vectors and product id values
            master_file_name = file_index_to_file_name[i]
            master_vector = file_index_to_file_vector[i]
            master_product_id = file_index_to_product_id[i]
        
            # Calculates the nearest neighbors of the master item
            nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
        
            # Loops through the nearest neighbors of the master item
            for j in nearest_neighbors:
                print(j)
                # Assigns file_name, image feature vectors and product id values of the similar item
                neighbor_file_name = file_index_to_file_name[j]
                neighbor_file_vector = file_index_to_file_vector[j]
                neighbor_product_id = file_index_to_product_id[j]
        
                # Calculates the similarity score of the similar item
                similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
                rounded_similarity = int((similarity * 10000)) / 10000.0
        
                # Appends master product id with the similarity score 
                # and the product id of the similar items
                named_nearest_neighbors.append({
                  'similarity': rounded_similarity,
                  'master_pi': master_product_id,
                  'similar_pi': neighbor_product_id})
            
            print("---------------------------------") 
            print("Similarity index       : %s" %i)
            print("Master Image file name : %s" %file_index_to_file_name[i]) 
            print("Nearest Neighbors.     : %s" %nearest_neighbors) 
            print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
        
      
        print ("Step.2 - Similarity score calculation - Finished ") 
    
        # Writes the 'named_nearest_neighbors' to a json file
        with open('nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)
    
        print ("Step.3 - Data stored in 'nearest_neighbors.json' file ") 
        print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))
        '''
        
    def loadTree(self, filename):
        self.t.load(filename);
        
    def createDictionary(self, filename):
        reader = csv.reader(open(filename))

        self.file_index_to_product_id = {}
        for row in reader:
            key = int(row[0])
            if key in self.file_index_to_product_id:
                pass
            self.file_index_to_product_id[key] = str(row[1])

    def predict_price(self, imagePath):
        #start_time = time.time()

        '''
        image = cv2.imread(imagePath)
        file_name = os.path.basename(i).split('.')[0]
        	label = price[int(imagePath.split(os.path.sep)[-1].split(".")[0]) - 1]
        '''
        '''
        #print ("Step.1 - Similarity score calculation - Started ") 
        
        named_nearest_neighbors = []
        
        # Assigns master file_name, image feature vectors and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]
        master_product_id = file_index_to_product_id[i]
        '''
        
        # Calculates the nearest neighbors of the master item
        #nearest_neighbors = self.t.get_nns_by_item(i, self.n_nearest_neighbors)
        
        v = g.get_feature_vector(imagePath)
        nearest_neighbors = self.t.get_nns_by_vector(v, self.n_nearest_neighbors)
        
        #Loops through the nearest neighbors of the master item to
        #get an average price
        
        total = 0;
        count = self.n_nearest_neighbors
        prices = []
        
        for j in nearest_neighbors:
            #Assigns file_name, image feature vectors and product id values of the similar item
            #neighbor_file_name = file_index_to_file_name[j]
            #neighbor_file_vector = file_index_to_file_vector[j]
            neighbor_product_id = str(self.file_index_to_product_id[j])
            #print("${:,.2f}".format(float(neighbor_product_id)))
            neighbor_price = Decimal(neighbor_product_id.replace('$','').replace(',',''))
        
            total += neighbor_price
            
            prices.append(neighbor_price)
            #Calculates the similarity score of the similar item
            #similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            #rounded_similarity = int((similarity * 10000)) / 10000.0
    
            #Appends master product id with the similarity score 
            #and the product id of the similar items
            #named_nearest_neighbors.append({
            #  'similarity': rounded_similarity,
            #  'master_pi': master_product_id,
            #  'similar_pi': neighbor_product_id})
        '''
        total = 0
        count = self.n_nearest_neighbors

        for price in prices:
            if 0 <= price < 500:
                total += float(price) * 0.148987714
            elif 500 <= price < 1000:
                total += float(price) * 0.377227894
            elif 1000 <= price < 1500:
                total += float(price) * 0.216646479
            elif 1500 <= price < 2000:
                total += float(price) * 0.106592836
            elif 2000 <= price < 2500:
                total += float(price) * 0.057103305
            elif 2500 <= price < 3000:
                total += float(price) * 0.03789583
            elif 3000 <= price < 3500:
                total += float(price) * 0.022841322
            elif 3500 <= price < 4000:
                total += float(price) * 0.015054508
            elif 4000 <= price < 4500:
                total += float(price) * 0.008652016
            elif 4500 <= price < 5000:
                total += float(price) * 0.003979927
            elif 5000 <= price < 5500:
                total += float(price) * 0.002249524
            elif 5500 <= price < 6000:
                total += float(price) * 0.001038242
            elif 6000 <= price < 6500:
                total += float(price) * 0.000692161
            elif price >= 6500:
                total += float(price) * 0.001038242
        '''
        avg = float(total / count)
        predicted_price = float(statistics.median(prices))

        #predicted_price = total / count
        print("Predicted Price: ${:,.2f}".format(predicted_price))
        #print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
        return Decimal(predicted_price)
    
    
        '''
        print("---------------------------------") 
        print("Similarity index       : %s" %i)
        print("Master Image file name : %s" %file_index_to_file_name[i]) 
        print("Nearest Neighbors.     : %s" %nearest_neighbors) 
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
        '''
        
        
