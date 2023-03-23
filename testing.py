import cv2
# from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import time 
import os
import sys
import findBoundary
from a_star import AStarPlanner
import plotting
sys.path.append(r'D:\THINGS\Yolo_v4\darknet\build\darknet\x64')
import darknet2 as darknet
import torch
import math
# import tensorflow as tf
mask = np.zeros((416,416,3), dtype=np.uint8)
from localization import get_corrected_top_view
import control
import dataLog
import custom_test
import serial
ARDUINO_CONNECTED = True
# np.set_printoptions(threshold=sys.maxsize)
prev_time_my = time.time()
def connect_arduino():
    #
    # if(ARDUINO_CONNECTED):
    arduino = None
    try:
        arduino = serial.Serial('/dev/ttyACM0', 115200,timeout=0.1)
        # arduino = serial.Serial('COM5', BAUD_RATE)# ,timeout=.1)
        arduino.flushInput()
        print("Connecting to: /dev/ttyACM0")
    except:
        try:
            arduino = serial.Serial('COM6', 115200)
            # arduino.flushInput()
            # arduino.flushOutput()
            print("com7")
        except:
            print("failed to connect to COM6")
    time.sleep(2)
    return arduino

def send_data(arduino, data):
    print("st_angle_send",data)
    # arduino.flushInput()
    # arduino.flushOutput()
    arduino.write(str(data).encode())

def stop_car(arduino):   #bluetooth 
    arduino.flushOutput()
    time.sleep(1)
    arduino.write(str('b').encode())
    # time.sleep(1)
    #arduino.close()
    print("Stop command")
    
def select_cam_port():
    port = 10
    camera = cv2.VideoCapture(port)
    # print("Searching Camera port")
    
    while camera.isOpened() is not True:
        
        camera.release()
        port -= 1
        camera = cv2.VideoCapture(port)
        
        if port == -1:
            print("CAMERA PORT NOT FOUND")
            return 0

    camera.release()
    print("Camera running on port:", port)
    return port

curr_path = os.path.dirname(__file__)
sx = 1000
sy = 0
MS = 1/2
# ratio = 
def set_saved_video(input_video, output_video, cap):
    # fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # print("output_video",output_video.shape)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    video = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    return video

def start():
    ############################### ##############################
    # model = keras.models.load_model(curr_path+"/roadseg_epoch30_dataset867.h5",compile=False)
    # model = keras.models.load_model(curr_path+"/roadseg_epoch20_256_dataset867.h5",compile=False)
    # model = keras.models.load_model(curr_path+"/seg_867_25_L.pt",compile=False)
    # '''model = create_model()
    # model = model.load_weights("seg_867_25_L.pt")'''
    # model = tf.keras.models.load_model("seg_867_25_L.pt")
    ############################################################
    # cap = cv2.VideoCapture(curr_path+"/testing_video.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\lane-segmentation\lane segmentation\raw2.mp4")
    # cap  = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Image Segmentation\data\data_road\testing\image_2\um_00000.png")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Solecthon\video.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\WIN_20230319_16_19_35_Pro.mp4")
    # cap =  cv2.VideoCapture(2)
    global prev_time_my
    arduino = None
    model = torch.load("seg_2142_20_pc.pt",map_location=torch.device('cuda'))
    model.eval()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture(r"D:\Autonomous\logs\23-3-2023\test3.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\WIN_20230319_16_14_39_Pro.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Desktop\Segmentation\test23 (1).mp4")
    # cap = cv2.VideoCapture(r"C:\Users\Aman Sheikh\Downloads\endurance9.mp4")
    ########################################################
    i = 0
    width = 416
    height = 416
    config_file = r"models\yolov4-3l-v4.cfg"
    data_file = r"models\yolov4-3l-v4.data"
    weights = r"models\yolov4-3l-v4.weights"
    network, class_names, class_colors   = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=1
        )
    j = 0
    f, log_file_name = dataLog.give_file()
    video = set_saved_video(cap, log_file_name+".avi", cap)
    count = 0
    if ARDUINO_CONNECTED:
        arduino = connect_arduino()
    # if ARDUINO_CONNECTED :
    #     print("Starting throttle command")
    #     # arduino.flushOutput()   
    #     arduino.write(str('a').encode())
        # time.sleep(1)
    similar_range = 25
    while cap.read():
        start = time.time()
        ret, frame = cap.read()
        if count == 1:
            
            if ARDUINO_CONNECTED :
                print("Starting throttle command")
                # arduino.flushOutput() 
                time.sleep(1)  
                arduino.write(str('a').encode())
                time.sleep(5)
        count+=1
        cv2.imwrite(r"D:\Autonomous\Segmentation_Pytorch (2)\logs\23-3-2023\output"+str(count)+".jpeg",frame)
        # video.write(frame)
        
        test_img = cv2.resize(frame,(416,416))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = 416
        height = 416
        
        frame_resized_darknet = cv2.resize(frame_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        frame_resized = cv2.bitwise_or(frame_resized,mask)
        # cv2.imshow("frame",frame)
        
        result = custom_test.predict(model,frame_resized)
        result = cv2.bitwise_not(result)
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # result = getSegmentedArea(test_img,model)
        # a = np.float32([(112.43,135),(298.1,135),(-230.9,416),(637.8,416)])
        # b = np.float32([[0,0],
        #             [416,0],
        #             [0,416],
        #             [416,416]])
        # M = cv2.getPerspectiveTransform(a,b)
        # result = cv2.warpPerspective(result, M,(416,416))
        img_for_detect = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized_darknet.tobytes())
        detection = darknet.detect_image(network, class_names, img_for_detect, thresh=.50)
        print("detections: ",detection)
        mybox = []
        for i  in range(len(detection)):
            x, y, w, h = detection[i][2][0],\
            detection[i][2][1],\
            detection[i][2][2],\
            detection[i][2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            # print(type(detection[0]))
            #person.append( ( (xmin+xmax)//2,(ymax) ) )
            p = np.array([[((xmax+xmin)//2), (ymax//1)]], dtype='float32')
            # a = np.array([[(2,1)]], dtype='float32')
            p = np.array([p])
            a = np.float32([(112.43,135),(298.1,135),(-230.9,416),(637.8,416)])
            b = np.float32([[0,0],
                   [416,0],
                   [0,416],
                   [416,416]])
            M = cv2.getPerspectiveTransform(a,b)
            pointsOut = cv2.perspectiveTransform(p, M)
            box = int(pointsOut[0][0][0]), int(pointsOut[0][0][1])
            mybox.append(box)
        # result = cv2.normalize(result, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        xvalues,yvalues,boundary,maxArea,center,boundAry = getBoundary(result)
        # cv2.imshow("boundAry",boundAry)
        cx,cy = center
        y = 415
        x = 208
        leftarea = []
        rightarea = []
        line_x = []
        line_y = []
        # while y >= cy:
        #     line_x.append(x)
        #     line_y.append(y)
        #     leftarea.append((x,y))
        #     rightarea.append((x,y))
        #     y -= 1
            
        end = xvalues[0]
        x = 208
        y = 416
        
        # selected_point_1 = (0,0)
        # selected_point_2 = (0,0)
        # j = 0
        # for i in range(len(bound)):
        #     if bound[i][1] == 415:
        #         j += 1
        #         selected_point_1 = bound[i]
        #         if j == 2:
        #             selected_point_2 = bound[i]
        #             break
        # print(selected_point_2)
        # while x >= selected_point_1[0]:
        #     leftarea.append((x,y))
        #     x -=1
        # while x <= selected_point_2[0]:
        #     rightarea.append((x,y))
        #     x +=1
        
        # for i in range(len(xvalues)):
        #     if xvalues[i] < 208:
        #         leftarea.append((xvalues[i],yvalues[i]))
        #     else:
        #         rightarea.append((xvalues[i],yvalues[i]))
        # x,y = zip(*leftarea)
        # x,y = list(x),list(y)
        # x1,y1 = zip(*rightarea)
        # x1,y1 = list(x1),list(y1)
        # leftArea = polygonArea(x,y,len(x))
        # rightArea = polygonArea(x1,y1,len(x1))
        # area_diff = leftArea-rightArea
        # print("leftArea: ",leftArea)
        # print("rightArea: ",rightArea)
        # print("area_diff: ",area_diff)
        # if area_diff > 0:
        #     goal = area_diff* 
        
        
            
            # xvalues.append(box[0])
            # yvalues.append(box[1])
        
        # bound = list(zip(xvalues,yvalues))
        # line = list(zip(line_x,line_y))
        # line = get_corrected_top_view(line)
        # bound = get_corrected_top_view(bound)
        # leftarea = get_corrected_top_view(leftarea)
        # rightarea = get_corrected_top_view(rightarea)
        
        boundAry = get_corrected_top_view(boundAry)
        xvalues,yvalues = zip(*boundAry)
        xvalues,yvalues = list(xvalues),list(yvalues)
        top_view = np.zeros((1100, 1300, 3),dtype=np.uint8)
        
        # for i in range(len(leftarea)):
        #     cv2.circle(top_view,(int(leftarea[i][0]-400),int(1100-leftarea[i][1])),10,(255,0,255),-1)
        # for i in range(len(rightarea)):
        #     cv2.circle(top_view,(int(rightarea[i][0]-400),int(1100-rightarea[i][1])),10,(255,200,0),-1)
        for b in boundAry:
            cv2.circle(top_view,(int(b[0]-400),int(1100-b[1])),5,(255,0,0),-1)
        # print(bound)

        sorted(boundAry,key=lambda x: x[1])
        similar_Y = []
        for i in range(len(boundAry)-1):
            if boundAry[i][1] == boundAry[i+1][1]:
                similar_Y.append(boundAry[i])
        n = len(similar_Y)
        if n > 1:
            print("ran_aman")
            gx = similar_Y[n//2][0]
            gy = similar_Y[n//2][1]
        # print("boundary",boundAry)
        # for i  in range(len(boundAry)):
        #     if boundAry[i][1] < mx:
        #         mx = bound[i][0]
        #         mxi = i 
                
        
        # print()
        # line_x,line_y = zip(*line)
        # line_x,line_y = list(line_x),list(line_y)
        # n = len(line)
        # cx = line_x[n//2]
        # cy = line_y[n//2]
        gx = boundAry[0][0]
        gy = boundAry[0][1]
        # left_gx = []
        # j = 1
        # for i in range(15):
        #     left_gx.append((bound[mni-j][0],bound[mni-j][1]))
        #     j += 1
        # right_gx = []
        # j = 1
        # for i in range(15):
        #     right_gx.append((bound[mni+j][0],bound[mni+j][1]))
        #     j += 1 
        # print(right_gx)
        # print(left_gx)
        # for i in range(15):
        #     cv2.circle(top_view,(int(right_gx[i][0]-400),int(1100-right_gx[i][1])),5,(0,0,255),-1)
        #     cv2.circle(top_view,(int(left_gx[i][0]-400),int(1100-left_gx[i][1])),5,(0,255,0),-1)
        # rx,ry,g_x,g_y = getpath(xvalues,yvalues,gx,gy)
        print("gx gy",gx,gy)
        cv2.circle(top_view,(int(gx-400),int(1100-gy)),20,(0,0,255),-1)
        mybox = get_corrected_top_view(mybox)
        for box in mybox:
            boundAry.append((box[0],box[1]))
            xvalues.append(box[0])
            yvalues.append(box[1])
        for b in mybox:
            cv2.circle(top_view,(int(b[0]-400),int(1100-b[1])),20,(100,100,0),-1)
        rx,ry,gx,gy = getpath(xvalues,yvalues,gx,gy)
        lines = list(zip(rx,ry))
        lines = lines[::-1]
        carx = 1000
        cary = 0
        i = 0
        radius = 180
        allPointsInside = False
        angle = 0
        print("path: ",lines)
        try:
            if type(lines) is tuple:
                lines = [lines]
            dist = math.sqrt((cary-lines[i][1])**2+(carx-lines[i][0])**2)
            if len(lines) > 1:
                while dist < radius:
                    dist = math.sqrt((cary-lines[i][1])**2+(carx-lines[i][0])**2)
                    if dist > radius:    
                        break
                    i += 1
                    if i > len(lines):
                        allPointsInside = True
                        break
            if allPointsInside:
                i = -1
            angle = control.car_angle((carx,cary),lines[i],True)
            print("outside",angle)
            angle = math.floor(angle)
        except:
            angle = 0
            i = -1
            print("problem in pp logic angle")
        selected_point = lines[i]
        if angle > 50:
            angle = 50
        elif angle < -50:
            angle = -50 
        print(f"angle: {angle},dist: {dist}")
        top_view = plotting.plot_cv2(rx,ry,top_view,(gx,gy),(sx,sy),radius,selected_point)
        top_view = cv2.resize(top_view, (608, 608),interpolation=cv2.INTER_LINEAR) 
        cv2.imshow("frame",test_img)
        # cv2.imshow("boundary",boundary)
        cv2.imshow("prediction",result)
        cv2.imshow("topview",top_view)
        end = time.time()
        d = end - start
        fps = 1/d
        print(f"fps: {fps}")
        if time.time() - prev_time_my >= MS:
            prev_time_my = time.time()
            if(ARDUINO_CONNECTED):

                serial_data = angle
                serial_writen_now = True
                
                send_data(arduino, serial_data)
        if cv2.waitKey(1) == ord("q"):
            break
    if ARDUINO_CONNECTED:
        stop_car(arduino)
    video.release()
    cap.release()
    cv2.destroyAllWindows()

def getBoundary(result):
    xvalues,yvalues,boundary,maxArea,center,boundAry= findBoundary.getBoundary(result)
    return xvalues,yvalues,boundary,maxArea,center,boundAry

def getpath(xvalues,yvalues,gx,gy):
    # li = list(zip(xvalues,yvalues))
    # print(f"fi:{li}")
    a_star = AStarPlanner(xvalues, yvalues,27,50)
    grid = a_star.get_grid(sx, sy, gx, gy,xvalues,yvalues)
    # print(grid)
    mn = sys.maxsize
    g_x = 0
    g_y = 0
    for i in range(len(grid)):
        dist = math.sqrt((grid[i][1]-gy)**2+(grid[i][0]-gx)**2)
        if dist < mn:
            mn = dist
            g_x = grid[i][0]
            g_y = grid[i][1]
    print(f"g_x: {g_x},g_y: {g_y}")
    rx, ry= a_star.planning(sx, sy, g_x, g_y,xvalues,yvalues)
    return rx,ry,g_x,g_y

def getSegmentedArea(test_img,model):
    frame = np.expand_dims(test_img, axis = 0)
    prediction = model.predict(frame)
    prediction_image = prediction.reshape((416,416))
    result = cv2.normalize(prediction_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return result

def polygonArea(X, Y, n):
 
    # Initialize area
    area = 0.0
 
    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0,n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i   # j is previous vertex to i
     
 
    # Return absolute value
    return int(abs(area/2.0))

def convertBack(x, y, w, h):
    """
    Converts detections output into x-y coordinates
    :x, y: position of bounding box
    :w, h: height and width of bounding box
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
 







if "__main__" == __name__ : 
    start()
    


# print("maxArea: ",maxArea)
        # for i in range(len(leftarea)):
        #     cv2.circle(topview,leftarea[i],5,(255,0,0),-1)
        # for i in range(len(rightarea)):
        #     cv2.circle(topview,rightarea[i],20,(255,0,0),-1)
        # x = 208
        # y = 416
        # while(x > )
        # print(f"len: ",len(xvalues))
        # for i  in range(len(xvalues)):
        #     if abs(208-xvalues[i]) < mn:
        #         mn = abs(208-xvalues[i])
        #         mni = i
                # print(mni,mn)



                # for i in range(417):
        #     xvalues.append(i)
        #     yvalues.append(416)
        # bound = list(zip(xvalues,yvalues))
        # temp_x = 0
        # temp_y = 414
        # while (temp_x,temp_y) not in bound:
        #     # print("running....")
        #     xvalues.append(temp_x)
        #     yvalues.append(temp_y)
        #     temp_y -= 1

        # temp_x = 415
        # temp_y = 414
        
        # while (temp_x,temp_y) not in bound:
        #     # print("run...")
        #     xvalues.append(temp_x)
        #     yvalues.append(temp_y)
        #     temp_y -= 1
        # print(f"{list(zip(xvalues,yvalues))}",end="")


'''min_dis = sys.maxsize
        for i  in range(len(xvalues)):
            min_dis = min(min_dis, abs(208-xvalues[i]))
            print('min_dis: ', min_dis)'''