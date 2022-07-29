import os

import cv2
import numpy as np

k = 0.3

def canny(img):

    # Setting All parameters
    t_lower = 255  # Lower Threshold
    t_upper = 255  # Upper threshold
    aperture_size = 3  # Aperture size
    
    # Applying the Canny Edge filter
    # with Custom Aperture Size
    edge = cv2.Canny(img, t_lower, t_upper, 
                    apertureSize=aperture_size)
    return edge
        

def projective(image,k):
    num_rows, num_cols = image.shape[:2]
    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1], [num_cols-1,num_rows-1]])
    dst_points = np.float32([[0,0], [num_cols-1,int(k*num_rows)], [0,num_rows-1], [num_cols-1,int((1-k)*num_rows)]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_protran = cv2.warpPerspective(image, projective_matrix, (num_cols,num_rows))
    return img_protran

def resize(image,size):
    dim = (size,size)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def turn(image, angle):
    rows,cols,channels= image.shape 
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle*-1,1) 
    rotate_30 = cv2.warpAffine(image,M,(cols,rows))
    return rotate_30 


def gaus_blur(img):
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)
    return img

def hard_blur(img):
    ksize = (7, 7)
    dst = cv2.blur(img, ksize)
    return dst

def smooth_blur(img):
    ksize = (3, 3)
    dst = cv2.blur(img, ksize)
    return dst


def reflection(img):
    rows, cols, dim = img.shape
    M = np.float32([[-1, 0, cols],
                     [ 0, 1, 0   ],
                     [ 0, 0, 1   ]])

    img = cv2.warpPerspective(img, M, (int(cols), int(rows)))
    return img








            

