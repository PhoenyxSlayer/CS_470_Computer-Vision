import cv2
import sys
import numpy as np
from pathlib import Path

def slice_image(image, lower_slice, upper_slice):
    image = np.copy(image)
    image = np.where(image < lower_slice, 0, image)
    image = np.where(image > upper_slice, 0, image)

    return image


def main():
    if len(sys.argv) < 5:
        print("Not enough arguments")
        exit()
    
    file_path = sys.argv[1]
    lower_slice = int(sys.argv[2])
    upper_slice = int(sys.argv[3])
    output_folder = sys.argv[4]
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    if image is None:
        print("There is no image to be loaded")
        exit()

    output = slice_image(image, lower_slice, upper_slice)
    out_filename = "OUT_" + Path(file_path).stem + "_" + str(lower_slice) + "_" + str(upper_slice) + ".png"
    output_directory = output_folder+"/"+out_filename

    cv2.imshow("IMAGE", output)
    cv2.waitKey(-1)
    cv2.imwrite(output_directory, output)

if __name__ == "__main__":
    main()