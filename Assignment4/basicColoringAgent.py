import cv2
import os, glob
from matplotlib import pyplot as plt
import numpy as np
import csv


class node:
    def __init__(self,data):
        self.value = data
        self.X = None
        self.Y = None
        self.xi = None
        self.close6 = []

#Do changes to this class as necessary
class pixel:
    def __init__(self,data):
        self.value = data
        self.X = None
        self.Y = None
        self.xi = None
        self.close6 = []




def plot(img, title = 'Plot', scale=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4*scale,6*scale))
    plt.title(title)
    plt.imshow(img)

def recolorTraining(pixelClusterGrid, avgValues, testImage):
    cluster0Val = avgValues[0]
    cluster1Val = avgValues[1]
    cluster2Val = avgValues[2]
    cluster3Val = avgValues[3]
    cluster4Val = avgValues[4]

    bRight = testImage[:,:,0]
    gRight = testImage[:,:,1]
    rRight = testImage[:,:,2]
    for i, pixel in enumerate(pixelClusterGrid):
        if pixel == 0:
            rRight[i] = cluster0Val[0]
            gRight[i] = cluster0Val[1]
            bRight[i] = cluster0Val[2]
        elif pixel == 1:
            rRight[i] = cluster1Val[0]
            gRight[i] = cluster1Val[1]
            bRight[i] = cluster1Val[2]
        elif pixel == 2:
            rRight[i] = cluster2Val[0]
            gRight[i] = cluster2Val[1]
            bRight[i] = cluster2Val[2]
        elif pixel == 3:
            rRight[i] = cluster3Val[0]
            gRight[i] = cluster3Val[1]
            bRight[i] = cluster3Val[2]
        elif pixel == 4:
            rRight[i] = cluster4Val[0]
            gRight[i] = cluster4Val[1]
            bRight[i] = cluster4Val[2]
    
    outImage = np.zeros_like(testImage)
    outImage[:,:,0] = bRight
    outImage[:,:,1] = gRight
    outImage[:,:,2] = rRight

    return outImage

def heapify(arr, n, i): 
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i].value < arr[l].value: 
        largest = l 
  
    if r < n and arr[largest].value < arr[r].value: 
        largest = r 

    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i] # swap 

        heapify(arr, n, largest) 

def heapSort(arr): 
    n = len(arr) 
 
    for i in range(int(n/2 - 1), -1, -1): 
        heapify(arr, n, i) 

    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i] # swap 
        heapify(arr, i, 0)

#Saves 3D array (arr) to a csv file file in the format of [r,b,g]
def toCSV(arr):
    with open("rbg.csv","w") as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in arr:
            for n in i:
                writer.writerow([n[0],n[1],n[2]])
#Opens rbg.csv and saves it into a 2D array [[r,b,g],[r,b,g]....]             
def getCSV():
    rbg = []
    with open("rbg.csv","r") as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            rbg.append(row)
    return rbg
#toCSV(trainImage)

def getAll3x3(arr):
    result = []
    sizeX = len(arr)
    sizeY = len(arr[0])
    x = 1
    while(x < sizeX - 1):
        y = 1
        while(y < sizeY - 1):
            #print(y)
            n = node(0)
            # (x,y) -> coordinate of the middle 
            n.X = x
            n.Y = y
            n.value += (arr[x][y-1])
            n.value += (arr[x][y])
            n.value += (arr[x][y+1])

            n.value += (arr[x-1][y-1])
            n.value += (arr[x-1][y])
            n.value += (arr[x-1][y+1])
            
            n.value += (arr[x+1][y-1])
            n.value += (arr[x+1][y])
            n.value += (arr[x+1][y+1])

            result.append(n)
            y+=3
        x+=3
    return result
    

def binarySetter(arr,target):
    i = 0
    n = len(arr)-1
    mid = int((n+i)/2)
    while(i < n):
        if(arr[mid].value == target.value):
            break
        else:
            if arr[mid].value < target.value:
                i = mid+1
                mid = int((n+i)/2)
            else:
                n = mid-1                
                mid = int((n+i)/2)
    #print(mid)
    if(mid+3 < len(arr) and mid-3 >= 0):
        target.close6 = arr[mid-3:mid+3]
    elif(mid+3 < len(arr)):
        target.close6 = arr[:6]
    else:
        target.close6 = arr[len(arr)-6:]

def setAll6(left,right):
    i = 0
    while i < len(right):
        binarySetter(left,right[i])
        i+=1

#matrix is a node object representing a 3x3 matrix on the right
#Check if there is a majority value for each cell. Returns the index for that submatrix. 
#Access the majority value -> node.value/9
def isMajority(left,matrix):
    i = 0
    for leftMatrix in matrix.close6:
        x = leftMatrix.X
        y = leftMatrix.Y
        value = int(leftMatrix.value/9)
        if(left[x][y-1] == value and left[x][y] == value and left[x][y+1] == value and 
        left[x-1][y-1] == value and left[x-1][y] == value and left[x+1][y+1] == value and 
        left[x+1][y-1] == value and left[x+1][y] == value and left[x+1][y+1] == value):
            return i
        i+=1
    return -1

#Finds the average value of the left that is closer to the average of the right
#returns [index of node object (in node.close6), average value of that submatrix]
def getAvgValueLeft(left,matrix):
    i = 0
    indexResult = 0
    avgRight = matrix.value/9
    avgLeft = -1
    tempAvg = 0
    #print("right:",avgRight)
    for leftMatrix in matrix.close6:
        tempAvg = leftMatrix.value/9
        #print("left",tempAvg)
        if(abs(avgRight - tempAvg) > avgLeft):
            avgLeft = abs(avgRight - tempAvg)
            indexResult = i
        i+=1
    return [matrix.close6[indexResult].value/9,indexResult]



# image = glob.glob('*.jpg')
# image = image[0]

# print('Found: ' + image)

# imData = cv2.imread(image)

# # plot(r, scale=2)
# # plot(g, scale=2)

# resX = imData.shape[1]
# resY = imData.shape[0]
# print(imData.shape)

# halfPoint = int(resX/2)
# print(halfPoint)
# trainImage = imData[:,:halfPoint,:]
# testImage = imData[:,halfPoint:,:]

# right = getAll3x3(trainImage)
# left = getAll3x3(testImage)
# heapSort(right)
# heapSort(left)

# setAll6(left,right)
# print(right[30].value)
# print(isMajority(testImage,right[30]))
# print(getAvgValueLeft(testImage, right[30]))


# for i in right[30].close6:
#     print(i.value)

# print(right[30].close6[4].X, right[30].close6[4].Y)

# print()
# print(1)

# for grid in right:
#     res = isMajority(grayTrain,grid)
#     if res != -1:
#         value = grid.close6[res].value/9
#         pixel = grid.close6[res]
#     else:
#         res = getAvgValueLeft(grayTrain,grid)
#         value = res[0]
#         pixel = grid.close6[res[1]]
    
#     x = pixel.X
#     y = pixel.Y
#     for i,_ in enumerate(grid):
#         grid[i] = pixelClusterGrid[x][y]
