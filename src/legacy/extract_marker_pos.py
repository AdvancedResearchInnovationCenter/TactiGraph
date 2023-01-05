import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops_table, regionprops

center = [155,123]

def cropFrames(img, circle_center=center, circle_rad=100, im_height=260, im_width=346, im_channels=1):
    
    mask = np.zeros((im_height, im_width), dtype=np.float32)            
    cv2.circle(mask, circle_center, circle_rad, [1]*im_channels, -1, 8, 0)
    cropped_image = np.multiply(mask, img)

    return cropped_image


tactile1 = cv2.imread('/home/hussain/Pictures/tactile1.png', cv2.IMREAD_GRAYSCALE)
detector = cv2.SimpleBlobDetector_create()
ret,thresh1 = cv2.threshold(tactile1,20,255,cv2.THRESH_BINARY)
thresh1 = cropFrames(thresh1)
thresh1 = cv2.erode(thresh1, np.ones((5, 5), np.uint8))
# Detect blobs.
#keypoints = detector.detect(thresh1)
#blobs = np.array([kp.pt for kp in keypoints])
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
"""
im_with_keypoints = cv2.drawKeypoints(
    tactile1, 
    keypoints, 
    np.array([]), 
    (0,0,255), 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
"""
print(regionprops(label(thresh1)))
#plt.scatter(*blobs.T, c='r')
plt.imshow((1-thresh1) * tactile1)
#plt.scatter(center[0], center[1])
#plt.scatter(*blobs.T)
plt.show()