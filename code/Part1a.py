import cv2 as cv
import numpy as np
from scipy.fftpack import fft2, ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt


# function to rescale the frame
def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dim = (width,height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)


# function for mask
def maskArea(image, r):
    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x-center[0])**2 + (y-center[1])**2 <= r*r
    mask[mask_area] = 0
    
    return mask

# video reading
capture = cv.VideoCapture('1tagvideo.mp4')

isTrue, frame = capture.read()

# rescaling image to fit into frame
image = rescaleFrame(frame, 1)
# convert to gray
image_gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# mask for filter
mask = maskArea(image_gray, 50)
# fourier transform
image_fft = fft2(image_gray)
# shifting origin to middle of image
image_fftshift = fftshift(image_fft)
# mangitude sprectume for plot
magnitude_sprectum = np.log(np.abs(image_fftshift))
# masking of the fourier transform
image_mask_fft = image_fftshift*mask
magnitude_sprectum_mask = np.log(np.abs(image_mask_fft))
# shifting masked image back and take inverse transform
image_ifft = np.abs(ifft2(ifftshift(image_mask_fft)))
# thresholding resultant image
threshold, thresh = cv.threshold(image_ifft, 50, 255, cv.THRESH_BINARY)


# plot the images
plt.subplot(2,2,1)
plt.imshow(image_gray, cmap='gray')
plt.title('Input image')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(magnitude_sprectum, cmap='gray')
plt.title('FFT of input image')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(magnitude_sprectum_mask, cmap='gray')
plt.title('Masking of FFT of input image with high pass filter')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(thresh, cmap='gray')
plt.title('Output image')
plt.xticks([])
plt.yticks([])
plt.show()

cv.waitKey(0)

print(type(image_ifft))

