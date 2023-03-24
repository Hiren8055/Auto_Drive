import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.filters import sobel
import sys
import cv2
def getBoundary(myimage):
    myimage = np.array(myimage,dtype=np.uint8)
    hist = np.histogram(myimage, bins= np.arange(0,256))

    selected_pixles = []
    for i in hist[1][:-1]:
        if hist[0][i] >= 700:
            selected_pixles.append(i)

    image = sobel(myimage)

    markers = np.zeros_like(myimage)
    markers[myimage < min(selected_pixles)] = 1
    markers[myimage > max(selected_pixles)] = 2
    a = np.float32([(112.43,135),(298.1,135),(-230.9,416),(637.8,416)])
    b = np.float32([[0,0],
                [416,0],
                [0,416],
                [416,416]])
    M = cv2.getPerspectiveTransform(a,b)
    markers = cv2.warpPerspective(markers, M,(416,416))
    markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ret,thresh_binary = cv2.threshold(markers,0,255,cv2.THRESH_BINARY)
    thresh_binary = cv2.cvtColor(thresh_binary, cv2.COLOR_BGR2GRAY)
    thresh_binary = np.expand_dims(thresh_binary,axis=-1)
    contours, hierarchy = cv2.findContours(image =thresh_binary , mode = cv2.RETR_EXTERNAL,method = cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((416,416),dtype=np.uint8)
    markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    j = 0
    mx = -sys.maxsize-1
    mxi = 0
    for i,cnt in enumerate(contours):
        if  cv2.contourArea(cnt) > mx:
                # print("ran2")
                cv2.drawContours(mask,[cnt], 0, (255), -1)
                mx = cv2.contourArea(cnt)
                mxi = i
    c = contours[mxi]
    # print("ran3")
 
    img = np.zeros((416,416),np.uint8)
# calculate points for each contour
    # for i in range(len(contours)):
        # creating convex hull object for each contour
    hull = cv2.convexHull(c, False)
    
    # for i in range(len(contours)):
    color_contours = (255, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(img, contours, mxi, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(img, hull, 0, color, 1, 8)

    img = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    x,y = np.where(img == [255])
    x = x.tolist()
    y = y.tolist()
    boundary = list(zip(y,x))
    a = np.float32([(112.43,135),(298.1,135),(-230.9,416),(637.8,416)])
    b = np.float32([[0,0],
                   [416,0],
                   [0,416],
                   [416,416]])
    M = cv2.getPerspectiveTransform(a,b)

    top_view = []
    for i in range(len(boundary)):
        a = np.array([[(boundary[i][0]),(boundary[i][1])]],dtype='float32')
        a = np.array([a])
        pointsOut = cv2.perspectiveTransform(a, M)
        box = int(pointsOut[0][0][0]), int(pointsOut[0][0][1])
        top_view.append(box)
    xvalues = []
    yvalues = []
    xvalues,yvalues = zip(*top_view)
    xvalues,yvalues = list(xvalues),list(yvalues)

    
    return xvalues,yvalues,boundary