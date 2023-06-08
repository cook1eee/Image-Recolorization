import numpy as np
from numpy import asarray
import random
import re
import matplotlib.pyplot as plt
from math import sqrt
from numpy import genfromtxt
import cv2
import csv
import math
import glob
#from basicColoringAgent import pixel
def convertImg(data):
    # img = cv2.imread(imgx)
    # data = np.array(img)
    #print(data)
    toCSV(data)
    #return data
    
#Saves 3D array (arr) to a csv file file in the format of [r,b,g]
def toCSV(arr):
    with open("rbg.csv","w") as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in arr:
            for n in i:
                writer.writerow([n[0],n[1],n[2]])

#Opens rbg.csv and saves it into a 2D array [[r,b,g],[r,b,g]....]             
# def getCSV():
#     rbg = []
#     with open("rbg.csv","r") as csvfile:
#         reader = csv.reader(csvfile,delimiter=',')
#         for row in reader:
#             improved.append(row)
#     return improved
# #toCSV(trainImage)

def loadDataSet():
    dataSet = np.loadtxt("rbg.csv")
    return dataSet
        
def euclideanDistance(row1,row2):
    distance = 0.0
    for i in range(len(row1)):
        distance +=(row1[i]-row2[i])**2
    return sqrt(distance)

def initialCentroid():
    #randomly pick 5 data points from csv.
    #centroid = convertImg()
    dataSet = list(loadDataSet())
    return random.sample(dataSet, 5)
    #return random.choice(dataSet)


def dataToClusters(dataSet, centroid, grid):
    rowLength =  grid[0].size 
    clusters = dict()
    k = len(centroid)
    index = -1
    for counter, item in enumerate(dataSet):
        vector1 = item
        minDis = math.inf
        for i in range(k):
            vector2 = centroid[i]
            distance = euclideanDistance(vector1, vector2)
            if distance<minDis:
                minDis = distance
                index = i
                grid[row][column-1] = index
        if index not in clusters.keys():
            clusters.setdefault(index, [])
        clusters[index].append(item)
    return clusters

def recalculateCentroids(clusters):
    newList = []
    for i in clusters.keys():
        centroid = np.mean(clusters[i], axis=0)
        newList.append(centroid)
    return newList

def getVariance(centroid,clusters):
    sum = 0.0
    for i in clusters.keys():
        vector1 = centroid[i]
        distance = 0.0
        for item in clusters[i]:
            vector2 = item
            distance += euclideanDistance(vector1,vector2)
        sum += distance
    return sum

def showCluster(centroidList, clusterDict):
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow'] 
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']
    print(len(centroidList[0]))
    print(len(clusterDict))
    # for key in clusterDict.keys():
    #     print(centroidList[key][0])
    #     print(centroidList[key][1])
    #     print(centroidList[key][2])
    #     print()
    
    #     #plt.plot(centroidList[key][0], centroidList[key][1], centroidList[key][2], centroidMark[key], markersize=1) #质心点
    #     # for item in clusterDict[key]:
    #     #     print("Son of beach!!!!!!!!")
    #     #     plt.plot(item[0], item[1], colorMark[key])
    # plt.show()
    print(clusterDict)
    print(centroidList)
    return [clusterDict,centroidList]

def test_k_means(imgx):
    pixelClusterGrid = np.zeros_like(imgx[:,:,0])
    convertImg(imgx)
    dataSet = loadDataSet()
    centroid = initialCentroid()
    #print(centroid)
    clusters = dataToClusters(dataSet,centroid, pixelClusterGrid)
    #print(clusters)
    newVar = getVariance(centroid,clusters)
    oldVar = 1
    while abs(newVar-oldVar)>=0.0000001:
        centroid = recalculateCentroids(clusters)
        clusters = dataToClusters(dataSet,centroid, pixelClusterGrid)
        oldVar = newVar
        newVar = getVariance(centroid,clusters)
    # print(centroid)
    for i in range(0, pixelClusterGrid.shape[0] , 1):
        for j in range(0, pixelClusterGrid.shape[1] , 1):
            # print('testImage', testImage[i][j])
            pixel = pixelClusterGrid[i][j]
    
    return pixelClusterGrid, centroid


def knnDriver(imgx):
    return test_k_means(imgx)


# pixelClusterGrid = [[Row1], [Row2], [Row n]] n = total number of pixels
# [Row1]= [0, 2, 3, 1 ]
# clusterBGRValues = [[b,g,r], [b,g,r],...] size 5
#                     |       |
#                 cluster0    Cluster 1