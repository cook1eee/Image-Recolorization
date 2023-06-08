def greyScale(imData):
    greyData = cv2.cvtColor(imData, cv2.COLOR_BGR2GRAY)
    for i in greyData:
        for j in i:
            greyData[i][j]=j/255
    return greyData


inputBlue = prepareLayerFilter(padBlue,dimF,oriImgDim,oriImgDim)
inputGreen = prepareLayerFilter(padGreen,dimF,oriImgDim,oriImgDim)
inputRed = prepareLayerFilter(padRed,dimF,oriImgDim,oriImgDim)
inputGray = prepareLayerFilter(padGray,dimF,oriImgDim,oriImgDim)

#----First Hidden Layer, assuming 30 nodes----
layer1 = calculateFilter(inputBlue,oriImgDim,30)
reluLayer1 = relu(layer1)

#----Second Hidden Layer, assuming 25 nodes----
layer2 = calculateFilter(reluLayer1,oriImgDim,25)
reluLayer2 = relu(layer2)

#----Third Hidden Layer, assuming 15 nodes----
layer3 = calculateFilter(reluLayer2,oriImgDim,15)
reluLayer3 = relu(layer3)

#----Fourth Hidden Layer, assuming 5 nodes----
layer4 = calculateFilter(reluLayer3,oriImgDim,5)
reluLayer4 = relu(layer4)

#----Fifth Hidden Layer, assuming 1 nodes----
layer5 = calculateFilter(reluLayer4,oriImgDim,1)
reluLayer5 = relu(layer5)
