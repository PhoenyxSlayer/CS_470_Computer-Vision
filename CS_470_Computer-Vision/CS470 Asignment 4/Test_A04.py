import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch
import A04

base_dir = "assign04"
image_dir = base_dir + "/" + "images"
ground_dir = base_dir + "/" + "ground"
out_dir = base_dir + "/" + "output"

RTOL=1e-07 
ATOL=1e-07

inputImageFilenames = [
    "Image0.png",
    "Image1.png",
    "Image2.png",
    "Image3.png",
    "Image4.png",
    "Image5.png",
    "Image6.png"
]

groundImageFilenames = [
    "LBP_Image0.png",
    "LBP_Image1.png",
    "LBP_Image2.png",
    "LBP_Image3.png",
    "LBP_Image4.png",
    "LBP_Image5.png",
    "LBP_Image6.png"
]

groundFeatureFilenames = [
    "HISTOGRAMS_1.csv",
    "HISTOGRAMS_2.csv",
    "HISTOGRAMS_3.csv",
    "HISTOGRAMS_4.csv"
]

class Test_A04(unittest.TestCase):   
    
    @classmethod
    def setUpClass(cls):        
        # Load up original and ground truth data first...
        cls.inputImages = []
        cls.groundImages = []
        cls.groundFeatures = []

        for i in range(len(inputImageFilenames)):
            image = cv2.imread(os.path.join(image_dir, inputImageFilenames[i]), cv2.IMREAD_GRAYSCALE)
            cls.inputImages.append(image)

        for i in range(len(groundImageFilenames)):
            gimage = cv2.imread(os.path.join(ground_dir, groundImageFilenames[i]), cv2.IMREAD_GRAYSCALE)
            cls.groundImages.append(gimage)
            
        for i in range(len(groundFeatureFilenames)):
            gdata = pd.read_csv(os.path.join(ground_dir, groundFeatureFilenames[i]))
            cls.groundFeatures.append(gdata)
        
    # Test each getLBPImage    
    def do_test_one_getLBPImage(self, index):
        # Load up original and ground truth images
        image = self.inputImages[index] 
        gimage = self.groundImages[index] 
        # Compute LBP image
        lbp = A04.getLBPImage(image)
        # Scale ground truth image
        ground = (np.copy(gimage)/28).astype("uint8")
        # Is it correct?
        np.testing.assert_allclose(lbp, ground, rtol=RTOL, atol=ATOL)

    # Test ALL getLBPImage
    def test_getLBPImage(self):
        for image_index in range(len(inputImageFilenames)):
            with self.subTest(image_index=image_index):
                self.do_test_one_getLBPImage(image_index)

        
    # Test each getOneRegionLBPFeatures    
    def do_test_getOneRegionLBPFeatures(self, index):
        # Grab ground truth data for ONE big region
        gor = self.groundFeatures[0] 
        
        # Scale ground truth image
        groundLBP = (np.copy(self.groundImages[index])/28).astype("uint8")
            
        # Compute feature vector with image as one big region
        features = A04.getOneRegionLBPFeatures(groundLBP)

        # Is this the right length?
        self.assertEquals(len(features), 10)
        
        # Grab the ground truth data for ONE big region for this particular image        
        oneGround = gor.loc[gor['Filename'] == inputImageFilenames[index]]            
        
        # Drop filename column
        oneGround = oneGround.drop(columns=["Filename"])            
        
        # Convert to numpy array
        oneGround = oneGround.to_numpy()[0]            
        
        # Actually do test  
        np.testing.assert_allclose(oneGround, features, rtol=RTOL, atol=ATOL)    
                
    # Test ALL getOneRegionLBPFeatures
    def test_getOneRegionLBPFeatures(self):
        for image_index in range(len(inputImageFilenames)):
            with self.subTest(image_index=image_index):
                self.do_test_getOneRegionLBPFeatures(image_index)

    # Test each getLBPFeatures    
    def do_test_getLBPFeatures(self, index):
        # Get number of regions on each side
        regionSideCnt = index+1

        # Get appropriate ground truth data
        gor = self.groundFeatures[index]
        
        for image_index in range(len(self.groundImages)):
            with self.subTest(image_index=image_index):
            
                # Scale ground truth image
                groundLBP = (np.copy(self.groundImages[image_index])/28).astype("uint8")     
        
                # Compute full feature vector...
                features = A04.getLBPFeatures(groundLBP, regionSideCnt)

                # Is this the right length?
                self.assertEquals(len(features), 10*regionSideCnt*regionSideCnt)
                
                # Grab the ground truth data for this particular image
                oneGround = gor.loc[gor['Filename'] == inputImageFilenames[image_index]]            
                
                # Drop filename column
                oneGround = oneGround.drop(columns=["Filename"])            
                
                # Convert to numpy array
                oneGround = oneGround.to_numpy()[0]            
                
                # Actually do test  
                np.testing.assert_allclose(oneGround, features, rtol=RTOL, atol=ATOL)      
                        
    # Test ALL getLBPFeatures
    def test_getLBPFeatures(self):
        for rcnt in range(len(self.groundFeatures)):
            with self.subTest(rcnt=rcnt):
                self.do_test_getLBPFeatures(rcnt)

def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A04))

if __name__ == '__main__':    
    main()
