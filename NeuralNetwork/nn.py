import cv2
import os, glob
from matplotlib import pyplot as plt
import numpy as np
import csv

image = glob.glob('*.jpg')
image = image[0]
imData = cv2.imread(image)


print(imData)