import cv2




def plot_cv2(rx,ry,topview,goal,start,radius,selected_point,boundary,grid):
    gx = goal[0]
    gy = goal[1]
    sx = start[0]
    sy = start[1]
    for b in boundary:
        cv2.circle(topview,(int(b[0]-400),int(1100-b[1])),5,(255,0,0),-1)
    for g in grid:
        cv2.circle(topview,(int(g[0]-400),int(1100-g[1])),2,(255,0,0),-1)
    for i in range(len(rx)):
        cv2.circle(topview,(rx[i]-400,1100-ry[i]),5,(255,255,0),-1)
    cv2.circle(topview,(int(gx-400),int(1100-gy)),20,(0,0,255),-1)
    cv2.circle(topview,(int(sx-400),int(1100-sy)),20,(255,0,0),-1)
    cv2.circle(topview,(int(sx-400),int(1100-sy)),radius,(255,255,0),2)
    cv2.circle(topview,(int(selected_point[0]-400),int(1100-selected_point[1])),10,(0,255,255),2)
    return topview