from turtle import window_height
import cv2
import sys
import numpy as np
from pathlib import Path

def applyFilter(image, kernel):
    image = image.astype("float64")
    kernel = kernel.astype("float64")
    kernel = cv2.flip(kernel, -1)
    kheight, kwidth = kernel.shape
    image_height, image_width = image.shape
    kh = kheight // 2
    kw = kwidth // 2

    padimage = cv2.copyMakeBorder(image, kh, kh, kw, kw, borderType=cv2.BORDER_CONSTANT, value=0)
    output = np.copy(image)
    for r in range(image_height):
        for c in range(image_width):
            sub = padimage[r:(r+kheight), c:(c+kwidth)]
            multresult = kernel*sub
            sumval = np.sum(multresult)

            output[r,c] = sumval

    return output




def main():
    if len(sys.argv) < 7:
        print("Not enough arguments")
        exit()
    elif len(sys.argv) < 7 + (int(sys.argv[3]) * int(sys.argv[4])):
        print("Not enough arguments")
        exit()

    path = sys.argv[1]
    output_path = sys.argv[2]
    rows = int(sys.argv[3])
    cols = int(sys.argv[4])
    alpha = float(sys.argv[5])
    beta = float(sys.argv[6])
    kernel = np.zeros((rows, cols))

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("There is no image to be loaded")
        exit()

    inc = 7
    for r in range(0, rows):
        for c in range(0, cols):
            kernel[r, c] = float(sys.argv[inc])
            inc += 1
    image = applyFilter(image, kernel)
    image = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
    cv2.imshow("test", image)
    cv2.waitKey(-1)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    main()