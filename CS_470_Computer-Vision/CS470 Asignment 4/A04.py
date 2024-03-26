import numpy as np
import cv2

def getLBPImage(image):
    values = np.zeros(8)
    flips = 0

    padImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    height = np.shape(padImage)[0]
    width = np.shape(padImage)[1]

    labelImage = np.copy(image)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_value = padImage[i, j]
            values[0] = padImage[i-1,j]
            values[1] = padImage[i-1,j+1]
            values[2] = padImage[i,j+1]
            values[3] = padImage[i+1,j+1]
            values[4] = padImage[i+1,j]
            values[5] = padImage[i+1,j-1]
            values[6] = padImage[i,j-1]
            values[7] = padImage[i-1,j-1]

            for k in range(len(values)):
                values[k] = np.where(values[k] > center_value, 1, 0)

            flips = 0
            for n in range(1,len(values)):
                if values[n-1] != values[n]:
                    flips = flips+1

            if flips <= 2:
                labelImage[i-1,j-1] = np.count_nonzero(values)
            else:
                labelImage[i-1,j-1] = 9

    return labelImage

def getOneRegionLBPFeatures(subImage):
    pixel_cnt = np.shape(subImage)[0] * np.shape(subImage)[1]


    return subImage

def getLBPFeatures(featuresImage, regionSideCnt):
    return featuresImage