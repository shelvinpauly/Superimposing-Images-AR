import cv2 as cv
import numpy as np


# function to rescale frame
def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dim = (width,height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)


#function to compute homography matrix
def computeHMatrix(A1, A2, A3, A4, B1, B2, B3, B4):
    
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

# function to compute homography
# output of the function will be final image after homography
def computehomography(H, resizedframe, blank):
    resizedframe = cv.cvtColor(resizedframe,cv.COLOR_BGR2GRAY)
    for i in range(blank.shape[0]):
        for j in range(blank.shape[1]):
            B = np.array([[j],[i],[1]])
            A = np.dot(H,B)  # multiplication with homography matrix
            A = A/A[-1]
            A = A.astype(int)
            x = A[0].item()
            y = A[1].item()
            blank[i][j] = resizedframe[y][x]
                 
    return blank

def decodeTag(tag):

    # split image vertically into 8 equal part
    bin = np.array_split(tag,8,axis=0)

    # split bin horizontally into 8 equal part
    # result will be array of blocks of size (50,50) pixel
    blocks = [np.array_split(bin_slice, 8 , axis=1) for bin_slice in bin ]
    
    # print(len(blocks))
    
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            median = np.median(blocks[i][j], axis=1)
            median = np.median(median)
            blocks[i][j][:] = median
    
    # rotate tag till white blocks reach to bottom right side of the tag
    while(int(np.median(blocks[5][5]))) != 255 :
        tag = np.rot90(tag,3)
        bin = np.array_split(tag,8,axis=0)
        blocks = [np.array_split(bin_slice, 8 , axis=1) for bin_slice in bin ]
         
    # check the inner 4 blocks of the tag
    # if the blocks is white, its corrosponding value will be 1 and it will be stored in value array. For black, the corrosponding value will be 1
    value = []
    
    coordinate = [(3,3),(3,4),(4,4),(4,3)]
    for i in coordinate:
        if np.median(blocks[i[0]][i[1]]) == 255:
            value.append(1)
        else:
            value.append(0)
    
    # decode the binary array stored in V into decimal value
    strs = [str(digit) for digit in value]
    strings = "".join(strs)
    a = int(strings, 2)
    return a
    
capture = cv.VideoCapture('1tagvideo.mp4')

# take single frame from video
isTrue,frame = capture.read()

# image processing operation
resizedframe = rescaleFrame(frame, 0.6)
blur = cv.GaussianBlur(resizedframe, (155,155), cv.BORDER_DEFAULT)
gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
blur = cv.GaussianBlur(thresh, (99,99), cv.BORDER_DEFAULT)

# corner detection using shi-tomasho detector
corner = cv.goodFeaturesToTrack(blur, 40, 0.06, 10)
corner = np.int0(corner)
corner = corner.reshape(corner.shape[0],2)

# find the white page corner and remove it from corner list
x_min, y_min = np.argmin(corner, axis=0)
x_max, y_max = np.argmax(corner, axis=0)
corner = np.delete(corner,[x_min, y_min, x_max, y_max],0)

# once the white page corner is removed, corner corrosponds to x_min, y_min, x_max and y_max are april tag corner
x_min, y_min = np.argmin(corner, axis=0)
x_max, y_max = np.argmax(corner, axis=0)

B1 = corner[y_min]
B2 = corner[x_max]
B3 = corner[y_max]
B4 = corner[x_min]

# blank image to compute homography
blank = np.zeros((400,400), dtype = 'uint8')

#find the corner of blank image
A1 = [0,0]
A2 = [blank.shape[1], 0]
A4 = [0, blank.shape[0]]
A3 = [blank.shape[1], blank.shape[0]]

# compute homograhy matrix
H = computeHMatrix(A1, A2, A3, A4, B1, B2, B3, B4)

# find homographic image
# the result will be april tag imposed on blank image
blank = computehomography(H, resizedframe, blank)

# apply threshold 
istrue, blank = cv.threshold(blank, 210, 255, cv.THRESH_BINARY)

cv.imshow('tag after homography', blank)

# pass the Ar tag to detect fiducial ID
tag_value = decodeTag(blank)

print("ID of the tag is"+ " " + str(tag_value))

cv.waitKey(0)
