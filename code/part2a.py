import copy
import cv2 as cv
import numpy as np

# function to rescale frame
def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dim = (width,height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

# function to compute homograhy matrix
# output will be homography matrix
def computeHMatrix(corner,blank):
    
    
    x_min, y_min = np.argmin(corner, axis=0)
    x_max, y_max = np.argmax(corner, axis=0)
    corner = np.delete(corner,[x_min, y_min, x_max, y_max],0)


    x_min, y_min = np.argmin(corner, axis=0)
    x_max, y_max = np.argmax(corner, axis=0)

    B1 = corner[y_min]
    B2 = corner[x_max]
    B3 = corner[y_max]
    B4 = corner[x_min]

    A1 = [0,0]
    A2 = [blank.shape[1], 0]
    A4 = [0, blank.shape[0]]
    A3 = [blank.shape[1], blank.shape[0]]
    
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

# function to find the final homographic image
def computehomography(H, resizedframe, blank):
    resizedframe = cv.cvtColor(resizedframe,cv.COLOR_BGR2GRAY)
    for i in range(blank.shape[0]):
        for j in range(blank.shape[1]):
            B = np.array([[j],[i],[1]])
            A = np.dot(H,B)
            A = A/A[-1]
            A = A.astype(int)
            x = A[0].item()
            y = A[1].item()
            blank[i][j] = resizedframe[y][x]
                 
    return blank   


# function to compute homopgraphy to paste testudo image in video frame
def computeTestudoHomography(H, resizedframe, blank):
    final_image = copy.deepcopy(resizedframe)
    
    resizedframe = cv.cvtColor(resizedframe,cv.COLOR_BGR2GRAY)
    
    for i in range(blank.shape[0]):
        for j in range(blank.shape[1]):
            B = np.array([[j],[i],[1]])
            A = np.dot(H,B)
            A = A/A[-1]
            A = A.astype(int)
            x = A[0].item()
            y = A[1].item()
            final_image[y][x][:] = blank[i][j][:]
                 
    return final_image


# function to find the correct tag orientation
def tagOrientation(corner, resizedframe, blank):
    # compute homography
    H = computeHMatrix(corner,blank)
    # find image of tag imposed on blank image
    blank = computehomography(H, resizedframe, blank)
    # apply threshold to blank image
    istrue, tag = cv.threshold(blank, 210, 255, cv.THRESH_BINARY)

    # split image vertically into 8 equal part
    bin = np.array_split(tag,8,axis=0)

    # split bin horizontally into 8 equal part
    # result will be array of blocks of size (50,50) pixel
    blocks = [np.array_split(bin_slice, 8 , axis=1) for bin_slice in bin ]

    # once the image is devided into blocks, apply the image processing to make sure that each block is completely black or white
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            median = np.median(blocks[i][j], axis=1)
            median = np.median(median)
            blocks[i][j][:] = median

    rotation = 0
    # find the rotation and stored it into rotation variable
    while np.median(blocks[5][5]) != 255:
        tag = np.rot90(tag,3)
        bin = np.array_split(tag,8,axis=0)
        blocks = [np.array_split(bin_slice, 8 , axis=1) for bin_slice in bin ]
        rotation = rotation + 1
        if rotation > 3:
            break
        
    return rotation
    
# read video
capture = cv.VideoCapture('1tagvideo.mp4')

# read testudo image
testudo_main = cv.imread('testudo.png')

# start video writer
video  = cv.VideoWriter('Testudo_imposed.avi', cv.VideoWriter_fourcc(*'XVID'),27,(1152,648))

while True:
    try:
        isTrue,frame = capture.read()
        if not isTrue:
            break
        testudo = copy.deepcopy(testudo_main)
        
        # process the image frame to remove noise. it will necessary to detect corner
        resizedframe = rescaleFrame(frame, 0.6)
        blur = cv.GaussianBlur(resizedframe, (155,155), cv.BORDER_DEFAULT)
        gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
        gblur = cv.GaussianBlur(thresh, (99,99), cv.BORDER_DEFAULT)
        
        # detect corner using shi-tomashi filter
        corner = cv.goodFeaturesToTrack(gblur, 40, 0.06, 10)
        corner = np.int0(corner)
        corner = corner.reshape(corner.shape[0],2)

        # blank image to compute rotation
        blank = np.zeros((400,400), dtype = 'uint8')

        # find the rotation of tag in each frame
        rotation = tagOrientation(corner, resizedframe, blank)

        # rotate the testudo as per tag rotation
        while rotation != 0:
            testudo = cv.rotate(testudo, cv.ROTATE_90_COUNTERCLOCKWISE)
            rotation = rotation - 1

        # compute homography matrix between video frame and testude
        H = computeHMatrix(corner, testudo)
        
        # perform the final homography to imposed testudo on video frame
        final_image = computeTestudoHomography(H, resizedframe, testudo)
        
        cv.imshow('tag_video', final_image)
        video.write(final_image)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break                
              
    except:
        isTrue,frame = capture.read()
        
video.release()