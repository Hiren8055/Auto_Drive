import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.filters import sobel
import sys
import cv2
def getBoundary(myimage):
    # print("ran")
    # cv2.imshow("result",myimage)
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

    # cv2.imshow("markers_top",markers)
    # cv2.waitKey(0)
    ret,thresh_binary = cv2.threshold(markers,0,255,cv2.THRESH_BINARY)
    thresh_binary = cv2.cvtColor(thresh_binary, cv2.COLOR_BGR2GRAY)
    thresh_binary = np.expand_dims(thresh_binary,axis=-1)
    # thresh_binary.sha
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    # dilated = cv2.dilate(image, kernel)
    contours, hierarchy = cv2.findContours(image =thresh_binary , mode = cv2.RETR_EXTERNAL,method = cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((416,416),dtype=np.uint8)
    markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("markers",markers)
    j = 0
    mx = -sys.maxsize-1
    mxi = 0
    for i,cnt in enumerate(contours):
        # print("ran1")
            # if hierarchy[0][i][2] == -1 :
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

# draw contours and hull points
    # color_contours = (0, 255ue - color for convex hull
    # draw ith contour
    # draw ith convex hull object
    # print("length: ",len(c))
    # # for cnt in contours:
    # print("Points: ",points)
    # cv2.drawContours(img, c, 1, (255,0,0), 1, 8, hierarchy)
    # img = cv2.drawContours(img, hull, 1, (255,0,0), 1, 8)
    # img = cv2.polylines(img,c,True,(255),2)
    #     # (x,y)=cnt[0,0]
    #     # if len(approx) >= 5:
    # img = cv2.drawContours(img, c, -1, (255,255,1), 3)
    img = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
        #     cv2.putText(img, 'Polygon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    # print("contours: ",contours)       
    # plt.contour(c[0],c[1],c[2])
    x,y = np.where(img == [255])
    x = x.tolist()
    y = y.tolist()
    boundary = list(zip(y,x))
    # all_pixels = []      
    # for i in range(0, len(c)):
    #     for j in range(0,len(c[i])):
    #         all_pixels.append(c[i][j])  
    # image=np.zeros((416,416),np.uint8)
    # boundary = []
    # for z in all_pixels:
    #     z = z.tolist()
    #     z = tuple(z)
    #     boundary.append(z)
    #     cv2.circle(image,z,2,(255,0,0),-1)
    # print(boundary)
    # xvalues,yvalues = zip(*boundary)
    # xvalues,yvalues = list(xvalues),list(yvalues)
    # for i in range(417):
    #         xvalues.append(i)
    #         yvalues.append(416)
    # bound = list(zip(xvalues,yvalues))
    # bound.sort()
    # print("bound",bound)
    # print("first_point",)
    # temp_x = bound[0][0]
    # temp_y = bound[0][1]
    # while (temp_x,temp_y) not in bound:
    #     xvalues.append(temp_x)
    #     yvalues.append(temp_y)
    #     temp_y -= 1

    # temp_x = bound[-2][0]
    # temp_y = bound[-2][1]
    
    # while (temp_x,temp_y) not in bound:
    #     # print("run...")
    #     xvalues.append(temp_x)
    #     yvalues.append(temp_y)
    #     temp_y -= 1
    # boundary = list(zip(xvalues,yvalues))
    # print("boundary: ",boundary)
    # print("boundary: ",boundary)
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
    # print(top_view)
    # print(f"boundary:{len(boundary)},top_view:{len(top_view)}")
    xvalues = []
    yvalues = []
    # for i in range(len(top_view)):
    #     xvalues.append(top_view[i][0])
    #     yvalues.append(top_view[i][1])
    xvalues,yvalues = zip(*top_view)
    xvalues,yvalues = list(xvalues),list(yvalues)
    # print(list(top_view))
    # top_view = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    # top_view = cv2.normalize(top_view, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # indices = np.where(top_view == [255])
    # curve, = plt.plot(indices[1],indices[0])
    # xvalues = curve.get_xdata()
    # yvalues = curve.get_ydata()
    # xvalues = xvalues.tolist()
    # yvalues = yvalues.tolist()
    # xvalues = indices[1]
    # yvalues = indices[0]
    # topview = np.zeros((416,416),dtype=np.uint8)
    # for i in range(len(top_view)):
    #     cv2.circle(topview,(int(top_view[i][0]),int(top_view[i][1])),2,(255,0,0),-1)
    # topview = cv2.normalize(topview, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mn = sys.maxsize
    mni = 0
    for i  in range(len(xvalues)):
        if abs(208-xvalues[i]) < mn:
            mn = abs(208-xvalues[i])
            mni = i
    cx = xvalues[mni]
    cy = yvalues[mni]
    # print("xval: ",xvalues)
    # print("yval: ",yvalues)
    # print("cx: ",cx)
    # print("cy: ",cy)
    # markers = cv2.normalize(markers, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # top_markers = cv2.perspectiveTransform(markers,M)
    # cv2.circle(top_markers,(208,416),20,(0,0,255),-1)
    # cv2.circle(top_markers,(gx,gy),20,(0,0,255))
    # print(f"gx:{gx},gy:{gy}")
    return xvalues,yvalues,image,mx,(cx,cy),boundary