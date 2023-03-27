import cv2
from tensorflow import keras
import imageio
from matplotlib import pyplot as plt
import numpy as np
import time 
import os
import sys
import findBoundary
from a_star import AStarPlanner
import plotting
# np.set_printoptions(threshold=sys.maxsize)

curr_path = os.path.dirname(__file__)
sx = 210
sy = 416
def start():
    ############################### ##############################
    # model = keras.models.load_model(curr_path+"/roadseg_epoch30_dataset867.h5",compile=False)
    model = keras.models.load_model(curr_path+"/roadseg_epoch30_dataset867.h5",compile=False)


    ############################################################
    # cap = cv2.VideoCapture(curr_path+"/testing_video.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\lane-segmentation\lane segmentation\raw2.mp4")
    # cap  = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Image Segmentation\data\data_road\testing\image_2\um_00000.png")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Solecthon\video.mp4")
    cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Downloads\endurance9.mp4")
    ########################################################
    i = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        test_img = cv2.resize(frame,(416,416))
        
        
        result,prediction_image = getSegmentedArea(test_img,model)
        xvalues,yvalues,goal,topview = getBoundary(result)
        gx = goal[0]
        gy = goal[1]
        # x,y = np.where(topview==255)
        # top_coordinates = 
        # print(top_coor)
        
        
        # print (indices)
        # coord = list(zip(indices[1], indices[0]))
        # print (coord)

        # cv2.circle
        # plt.title("Curve plotted using the given points")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()
        # a = np.array([[(208), (416)]], dtype='float32')
        # # a = np.array([[(2,1)]], dtype='float32')
        
        # a = np.array([a])
        # pointsOut = cv2.perspectiveTransform(a, M)
        # s = int(pointsOut[0][0][0]), int(pointsOut[0][0][1])
        # cv2.circle(topview,(s[0],s[1]),10,(255,0,0),-1)
        # print(f"sx:{s[0]},sy:{s[1]}")
        
        # n = len(xvalues)
        # gx = xvalues[n-1]
        # gy = yvalues[n-1]
        # for i in range(417):
        #     xvalues.append(i)
        #     yvalues.append(416)
        # bound = list(zip(xvalues,yvalues))
        # temp_x = 0
        # temp_y = 414
        # while (temp_x,temp_y) not in bound:
        #     print("running....")
        #     xvalues.append(temp_x)
        #     yvalues.append(temp_y)
        #     temp_y -= 1

        # temp_x = 415
        # temp_y = 414
        # # print(f"({xvalues},{yvalues}),",end="")
        # while (temp_x,temp_y) not in bound:
        #     print("run...")
        #     xvalues.append(temp_x)
        #     yvalues.append(temp_y)
        #     temp_y -= 1
        # # print(bound)
        # n = len(xvalues)
        # gx = xvalues[n//2]
        # gy = yvalues[n//2]
       
        
        rx,ry = getpath(xvalues,yvalues,gx,gy)
        # topview = plotting.plot_cv2(rx,ry,topview,goal)
        cv2.imshow("frame",test_img)
        cv2.imshow("prediction",prediction_image)
        cv2.imshow("topview",topview)
        
        end = time.time()
        d = end - start
        fps = 1/d
        print(f"fps: {fps}")
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

def getBoundary(result):
    xvalues,yvalues,goal,topview= findBoundary.getBoundary(result)
    return xvalues,yvalues,goal,topview

def getpath(xvalues,yvalues,gx,gy):
    a_star = AStarPlanner(xvalues, yvalues, 15, 1)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    return rx,ry

def getSegmentedArea(test_img,model):
    frame = np.expand_dims(test_img, axis = 0)
    prediction = model.predict(frame)
    prediction_image = prediction.reshape((416,416))
    result = cv2.normalize(prediction_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return result,prediction_image






if "__main__" == __name__ : 
    start()
    


