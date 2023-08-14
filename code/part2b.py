from ast import Lambda
import copy
import cv2 as cv
import numpy as np


# compute projection matrix from camera matrix and homography
def projMatrix(K,H):
    
    # calculate K_inv
    K_inv = np.linalg.inv(K)
    # calculate B tilda
    B_til = np.matmul(K_inv, H)
    if np.linalg.det(B_til)<0:
        B_til = B_til*(-1)
    Lambda = 2/(np.linalg.norm(np.matmul(K_inv, H[:,0]))  + np.linalg.norm(np.matmul(K_inv, H[:,1])) )
    r1 = Lambda * B_til[:,0]
    r2 = Lambda * B_til[:,1]
    r3 = np.cross(r1, r2, axis=0)
    t = Lambda * B_til[:,2]
    B = np.column_stack((r1, r2, r3, t))
    P = np.matmul(K,B)
    
    return P


# function to find corner of the cube
def findCorner(P):
    
    p1 = np.array([[0],[0],[-1],[1]])
    p2 = np.array([[0],[1],[-1],[1]])
    p3 = np.array([[1],[1],[-1],[1]])
    p4 = np.array([[1],[0],[-1],[1]])
    
    cp1 = np.matmul(P,p1)
    cp1 = cp1/cp1[-1]
    cp2 = np.matmul(P,p2)
    cp2 = cp2/cp2[-1]
    cp3 = np.matmul(P,p3)
    cp3 = cp3/cp3[-1]
    cp4 = np.matmul(P,p4)
    cp4 = cp4/cp4[-1]
    
    cp1 = cp1[:-1]
    cp2 = cp2[:-1]
    cp3 = cp3[:-1]
    cp4 = cp4[:-1]
    
    cp1 = np.asarray(cp1)
    cp2 = np.asarray(cp2)
    cp3 = np.asarray(cp3)
    cp4 = np.asarray(cp4)
    
    cube_point = np.float32([cp1, cp2, cp3, cp4])
    
    return cube_point


# function to draw the cube on the video frame
def drawCube(image, tag_corner, cube_corner):
    
    final_image = copy.deepcopy(image)
    cube_corner = cube_corner.astype(int)
    
    for i in range(4):
        p1 = (cube_corner[i][0][0], cube_corner[i][1][0])
        p2 = (tag_corner[i][0], tag_corner[i][1])
        final_image = cv.line(final_image, p2, p1, (0, 0, 255), thickness=4, lineType=8)
        
    for i in range(3):
        p1 = (cube_corner[i][0][0], cube_corner[i][1][0])
        p2 = (cube_corner[i+1][0][0], cube_corner[i+1][1][0])
        final_image = cv.line(final_image, p1, p2, (0, 0, 255), thickness=4, lineType=8)
        
    p1 = (cube_corner[0][0][0], cube_corner[0][1][0])
    p2 = (cube_corner[3][0][0], cube_corner[3][1][0])
    final_image = cv.line(final_image, p1, p2, (0, 0, 255), thickness=4, lineType=8)
    
    return final_image


# function to rescale the image frame
def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dim = (width,height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

# function to compute homography matrix
def computeHMatrix(corner):
    
    x_min, y_min = np.argmin(corner, axis=0)
    x_max, y_max = np.argmax(corner, axis=0)

    B1 = corner[x_min]
    B2 = corner[y_max]
    B3 = corner[x_max]
    B4 = corner[y_min]
    A1 = [0, 0]
    A2 = [0, 1]
    A3 = [1, 1]
    A4 = [1, 0]
    
    A = np.array([
                  [A1[0], A1[1], 1, 0, 0, 0, -B1[0]*A1[0], -B1[0]*A1[1], -B1[0]],
                  [0, 0, 0, A1[0], A1[1], 1, -B1[1]*A1[0], -B1[1]*A1[1], -B1[1]],
                  [A2[0], A2[1], 1, 0, 0, 0, -B2[0]*A2[0], -B2[0]*A2[1], -B2[0]],
                  [0, 0, 0, A2[0], A2[1], 1, -B2[1]*A2[0], -B2[1]*A2[1], -B2[1]],
                  [A3[0], A3[1], 1, 0, 0, 0, -B3[0]*A3[0], -B3[0]*A3[1], -B3[0]],
                  [0, 0, 0, A3[0], A3[1], 1, -B3[1]*A3[0], -B3[1]*A3[1], -B3[1]],
                  [A4[0], A4[1], 1, 0, 0, 0, -B4[0]*A4[0], -B4[0]*A4[1], -B4[0]],
                  [0, 0, 0, A4[0], A4[1], 1, -B4[1]*A4[0], -B4[1]*A4[1], -B4[1]],
                ])
    
    U, S, V = np.linalg.svd(A)

    h = V[-1,:]
    h = h/h[-1]
    h = np.array(h)
    H =  (np.reshape(h,(3,3)))
    return H

# camera matrix adjusted as per rescale image
K = np.array([[1346.10059534175*0.6, 0, 932.163397529403*0.6],[0, 1355.93313621175*0.6, 654.898679624155*0.6],[0,0,1]])

# read video frame
capture = cv.VideoCapture('1tagvideo.mp4')
video  = cv.VideoWriter('cube.avi', cv.VideoWriter_fourcc(*'XVID'),27,(1152,648))


while True:
    
    try:
        
        isTrue,frame = capture.read()
        
        if not isTrue:
            break
        
        # read video frame
        resizedframe = rescaleFrame(frame, 0.6)
        
        # apply preprocessing to remove noise
        blur = cv.GaussianBlur(resizedframe, (155,155), cv.BORDER_DEFAULT)
        gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
        gblur = cv.GaussianBlur(thresh, (99,99), cv.BORDER_DEFAULT)
        
        # detect corner using shi-tomashi corner detector
        corner = cv.goodFeaturesToTrack(gblur, 40, 0.06, 10)
        corner = np.int0(corner)
        corner = corner.reshape(corner.shape[0],2)
        
        # find the white page corner and remove it from list
        x_min, y_min = np.argmin(corner, axis=0)
        x_max, y_max = np.argmax(corner, axis=0)
        corner = np.delete(corner,[x_min, y_min, x_max, y_max],0)

        # find the Ar tag corner and stored it into variable name tag_corner as per pojection point sequance
        x_min, y_min = np.argmin(corner, axis=0)
        x_max, y_max = np.argmax(corner, axis=0)

        B1 = corner[x_min]
        B2 = corner[y_max]
        B3 = corner[x_max]
        B4 = corner[y_min]
        
        tag_corner = []
        
        tag_corner.append(B1)
        tag_corner.append(B2)
        tag_corner.append(B3)
        tag_corner.append(B4)
        
        # compute homography between april tag and base of the cube(projection of cube on arpil tag)
        H = computeHMatrix(corner)

        # compute projection matrix
        P = projMatrix(K,H)

        # find the pixel location of corner using projection matrix
        cube_corners = findCorner(P)
        
        # draw the cube on each video frame
        # the output will be the each video frame containing cube on it
        final_image = drawCube(resizedframe, tag_corner, cube_corners)
        
        video.write(final_image)

        cv.imshow('tag_video', final_image)
        
        if cv.waitKey(20) & 0xFF==ord('d'):
            break                
              
    except:
        isTrue,frame = capture.read()
        
video.release()