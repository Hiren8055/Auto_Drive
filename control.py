import sys
import math
import numpy as np

###########---Pure Pursuit---##############
L = 200

def slope(x1, y1, x2, y2):
    if(x2 - x1 != 0):
      return (float)(y2-y1)/(x2-x1)
    return sys.maxsize


def lineLineIntersection(A, B, C, D):
	# Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])

    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])

    determinant = a1*b2 - a2*b1

    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return (10*9, 10*9)
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return (x, y)

def findAngle(M1, M2):
    PI = 3.14159265
     
    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
 
    # Calculate tan inverse of the angle
    ret = math.atan(angle)
 
    # Convert the angle from
    # radian to degree
    # val = (ret * 180) / PI
 
    # Print the result
    return ret

def angleForCar(path,prev_angle): 
    carx = 1000
    cary = 0
    i = 0
    print(path)
    # if len(path) == 1:
    #     return 0,i
    allPointsInside = False
    if type(path) is tuple:
        path = [path]
    dist = math.sqrt((cary-path[i][1])**2+(carx-path[i][0])**2)
    if len(path) > 1:
        while dist < L:
            dist = math.sqrt((cary-path[i][1])**2+(carx-path[i][0])**2)
            if dist > L: 
                break
            i += 1
            if i >= len(path):
                allPointsInside = True
                break
    if allPointsInside:
        i = -1
    if i > 0 :
        point = rightlinecircleintersection(path[i-1],path[i],[carx,cary],L)
    else:
        point = rightlinecircleintersection([carx,cary],path[i],[carx,cary],L)
    #print("points ",point[0],point[1])
    # if nextPoint >= 1:
    #     point = self.rightlinecircleintersection(pathpoints[nextPoint-1],pathpoints[nextPoint],[carx,cary],lookAhead)
    # else:
    #     point = self.rightlinecircleintersection([carx,cary],pathpoints[nextPoint],[carx,cary],lookAhead)
    # print("points ",point[0],point[1])
    
    # cv2.circle(image,(int(point[1][0]),int(point[1][1])),10,(0,0,255),-1)
    
    # print("*******")
    # print(path[i][0])
    # print(carx)
    # print("*******")
    d = math.sqrt(62*2 + 130*2)/2.0
    d1 = math.sqrt(20*2 + 130*2)/2.0
    alpha = math.radians(prev_angle) # rotation angle in radians
    beta = math.atan2(130, 62)
    beta1 = math.atan2(130, 20)
    P2 = (int(carx - d * math.cos(beta + alpha)), int(cary - d * math.sin(beta+alpha)))
    P1 = (int(carx + d * math.cos(beta - alpha)), int(cary - d * math.sin(beta-alpha))) 
    P0 = (int(carx + d * math.cos(beta + alpha)), int(cary + d * math.sin(beta+alpha))) 
    P3 = (int(carx - d * math.cos(beta - alpha)), int(cary + d * math.sin(beta-alpha))) 
    x = d1
    
    midPoint = [(P1[0]+P2[0])//2,(P1[1]+P2[1])//2]
    
    # cv2.circle(image,(int(midPoint[0]),int(midPoint[1])),10,(0,0,0),-1)
    
    print("################################")

    print(midPoint)
    print(P1[0],P2[0])
    print("################################")
    # cv2.line(image, (int(carx),int(cary)), (int(point[1][0]),int(point[1][1])), (0,255,0), 10)
    # cv2.line(image, (int(carx),int(cary)), (int(midPoint[0]),int(midPoint[1])), (0,0,255), 10)
    intersection = lineLineIntersection(P1,P2,point[1],(carx,cary))
    start = (int(carx - x * math.cos(beta1 + alpha)), int(cary - x * math.sin(beta1+alpha)))
    end   = (int(carx + x * math.cos(beta1 - alpha)), int(cary - x * math.sin(beta1-alpha))) 
    if int(intersection[0]) in range(start[0],end[0]):
        return 0,i

    m1 = slope(carx,cary,point[1][0],point[1][1])
    m2 = slope(carx,cary,midPoint[0],midPoint[1])
    
    # gamma = findAngle(m1,m2)
    gamma = math.atan2(path[i][1],path[i][0])
    # gamma = math.radians(gamma)
    delta = math.atan((2*190*math.sin(gamma))/L)
    # sign = (point[1][0]-midPoint[0])*(path[i][1]-midPoint[1])-(point[1][1]-midPoint[1])*(path[i][0]-midPoint[0])
    # print("sign: ",sign)
    sign = (2*(point[1][0]-midPoint[0]))/(L**2)
    # delta = round(delta)
    if sign < 0:
        return -delta,i
    return delta,i
    # return delta,i
        
    
def rightlinecircleintersection(pointA,pointB,center,radius):
    '''
    This function finds intersection of line drawn from point inside the circle and point just outside the circle. 
    Initially for the first point car is considered as inside point.
    '''
    baX = pointB[0] - pointA[0]
    baY = pointB[1] - pointA[1]
    caX = center[0] - pointA[0]
    caY = center[1] - pointA[1]
    a = baX * baX + baY * baY
    bBy2 = baX * caX + baY * caY #
    c = caX * caX + caY * caY - radius * radius #quadC
    pBy2 = bBy2 / a
    q = c / a
    disc = pBy2 * pBy2 - q
    if disc < 0: 
        return []
    # // if disc == 0 ... dealt with later
    tmpSqrt = math.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt
    p1 = [pointA[0] - baX * abScalingFactor1, pointA[1]
            - baY * abScalingFactor1]
    if disc == 0: # abScalingFactor1 == abScalingFactor2
        return [p1]
    p2 = [pointA[0] - baX * abScalingFactor2, pointA[1]
            - baY * abScalingFactor2]
    return [p1,p2]

###########---Pure Pursuit---##############
a = range(-75, -26)
b = range(-26, -19)  # 7
c = range(-19, -13)  # 6
d = range(-13, -7)  # 6
e = range(-7, 0)  # 8
f = range(0, 7)
g = range(7, 13)
h = range(13, 19)
i = range(19, 26)
j = range(26, 75)

m = range(-90, -26)
n = range(-26, -12)
o = range(-12, 0)
p = range(0, 12)
q = range(12, 26)
r = range(26, 90)


def steer(angle):
    """
    Maps angle range to integer for sending to Arduino
    :angle:   steering angle
    :returns: mapped integer
    """
    if(angle in a):
        return '0'
    elif(angle in b):
        return '1'
    elif(angle in c):
        return '2'
    elif(angle in d):
        return '3'
    elif(angle in e):
        return '4'
    elif(angle in f):
        return '4'
    elif(angle in g):
        return '5'
    elif(angle in h):
        return '6'
    elif(angle in i):
        return '7'
    elif(angle in j):
        return '8'


def car_angle(p1, p2, is_ground):
    """
    Computes angle w.r.t., car
    :returns: angle w.r.t, car
    """
    
    x, y = p1
    print(p2)
    if p2:
        p, q = p2
    print("p1:", p1)
    print("p2:", p2)
    try:
        slope = (q - y)/(p - x)
    except:
        print("entered in angle except")
        #slope = 99999
        slope = sys.maxsize
    angle = np.arctan(slope)*180/math.pi
    
    if is_ground is False:
        if(angle > 0):
            return -1*(90 - angle)
        return (90 + angle)
    else:
        if(angle > 0):
            return (90 - angle)
        return -1*(90 + angle)


