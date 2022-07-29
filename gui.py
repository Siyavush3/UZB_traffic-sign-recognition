import numpy as np
import cv2

from PIL import Image, ImageEnhance
#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
# IMPORT THE TRANNIED MODEL
from keras.models import load_model
model = load_model('traffic_sigm.h5')


def detect_image(img):
    
    image = img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))
    cv2.imwrite('hsv.jpg', hsv)
    mask = cv2.inRange(hsv, (89, 105, 73), (255, 255, 255))

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)


    contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected_images = []
    i = 0
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        save_image = image[y:y + h, x:x + w]
        Height, Width, _ = save_image.shape
        if not ((Width > Height + 20) or (Height > Width + 20)) and (
                ((Width > 30) and (Height > 30)) and ((Width < 70) and (Height < 70))):
            cv2.imwrite(str(i)+'.png' , save_image)
            detected_images.append(save_image)
        i += 1
    return detected_images

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    return img


def equalize(img):
    img = cv2.equalizeHist(img)


    return img


def bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(im_pil)
    factor = 3.0
    im_output = enhancer.enhance(factor) 
    open_cv_image = np.array(im_output) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image 


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


def preprocessing(img):
    img = bright(img)
    img = canny(img)
    img = grayscale(img)
    

    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Attention Crosswalk'
    elif classNo == 1:
        return 'Speed limit 50'
    elif classNo == 2:
        return 'No drive'
    elif classNo == 3:
        return 'No road'
    elif classNo == 4:
        return 'One way road'
    elif classNo == 5:
        return 'dangerous'
    elif classNo == 6:
        return 'Photo radar'
    elif classNo == 7:
        return 'Attention Circular motion'
    elif classNo == 8:
        return 'Stop'
    elif classNo == 9:
        return 'raffic light'
    elif classNo == 10:
        return 'Videio radar'
    elif classNo == 11:
        return 'Crosswalk'
    elif classNo == 12:
        return 'End of one way road'
    elif classNo == 13:
        return 'End of speed limit'
    elif classNo == 14:
        return 'Removal of restrictions'
    elif classNo == 15:
        return 'give way'
    elif classNo == 16:
        return 'children'
    elif classNo == 17:
        return 'Driving straight or left'

video_capture = cv2.VideoCapture("report.mp4")
ret = True
while ret:
    ret,frame = video_capture.read()
    dImages = detect_image(frame)
    print(len(dImages))
    if not(len(dImages)==0): 
        for imgOrignal in dImages:
            print(imgOrignal )

        # PROCESS IMAGE
            img = np.asarray(imgOrignal)
            print(img )
            img = cv2.resize(img, (64, 64),interpolation = cv2.INTER_AREA)
            print(img )
            img = preprocessing(img)



        # PREDICT IMAGE
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=1)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
            # print(getCalssName(classIndex))

                cv2.imwrite(str(getCalssName(classIndex))+'_'+str(probabilityValue) + '.png', imgOrignal)

