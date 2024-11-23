import cv2
import pytesseract
import numpy as np
import random

# Set the path for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#maybe | instead of blank
"""
##https://stackoverflow.com/questions/71097585/python-tesseract-ocr-isnt-properly-detecting-both-numbers
bgr_image = cv2.imread("battle.png")
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, np.array([0, 180, 218]), np.array([60, 255, 255]))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dilate = cv2.dilate(mask, kernel, iterations=1)
thresh = 255 - cv2.bitwise_and(dilate, mask)
text = pytesseract.image_to_string(thresh, config="--psm 6 digits")
print(text)
"""
"""
#https://stackoverflow.com/questions/65739986/text-reading-with-tesseract-in-a-noisy-image
img = cv2.imread("battle.png")
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thr = cv2.threshold(gry, 220, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
txt = pytesseract.image_to_string(thr,config='--psm 6')
print(txt)
cv2.imshow("thr", thr)
cv2.waitKey(0)
"""

def read_screen(screen):
    img = None
    #while type(img) == type(None): # lib read or lib DAT: CRC error
    img = cv2.imread(screen)
    if type(img) == type(None):
        raise Exception("Read Image Failed")
    return img

def name_read(img, top_down, left_right):
    """
    tested with red background and 2 and 1 digit, red background
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.3), int((top_down[1]-top_down[0])*1.3))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([100, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)

    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    #https://stackoverflow.com/questions/2363490/limit-characters-tesseract-is-looking-for
    txt = pytesseract.image_to_string(cropped,config=' tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6')
    

    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

# Read text using the blue color as mask
# ?
def blue_read(img, top_down, left_right):
    """
    tested for blue
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.55), int((top_down[1]-top_down[0])*1.55))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([100, 255, 100]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='--psm 6')
    
    
    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

# Read text using a color mask defined by specific ranges
# ?
def color_read(img, top_down, left_right):
    """
    tested for cream white
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.55), int((top_down[1]-top_down[0])*1.55))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([100, 100, 100]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)
    
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=0123456789 --psm 6')
    
    
    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

# Read text using the red color as mask
# ?
def red_read(img, top_down, left_right):
    """
    tested with red background and 2 and 1 digit, before the inrange
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.3), int((top_down[1]-top_down[0])*1.3))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    cropped = cv2.bitwise_not(cropped)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='--psm 6')
    

    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

# Read text using a specific color mask
# ?
def odd_read(img, top_down, left_right):
    """
    tested with red background and 2 and 1 digit, red background
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.3), int((top_down[1]-top_down[0])*1.3))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([150, 255, 255]))
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel = np.zeros((1,2),dtype=np.uint8)#was not here before
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)

    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=0123456789/ --psm 6')
    

    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

# Read text using the white color as mask
# ?
def white_read(img, top_down, left_right):
    """
    tested with blue background 2 digits
    """
    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*1.55), int((top_down[1]-top_down[0])*1.55))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([255, 160, 255]))#was [255, 150, 255]
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel = np.zeros((1,2),dtype=np.uint8)#was not here before
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6')
    
    
    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

def random_white_read(img, top_down, left_right):
    """
    tested with blue background 2 digits. White read but with random for dim and kernel.
    """

    scale = random.uniform(1,2)
    kernel_types = [cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), np.zeros((round(random.uniform(1,2)),2),dtype=np.uint8)]

    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*scale), int((top_down[1]-top_down[0])*scale))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([255, 100, 255]))#was [255, 150, 255]
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel = random.choice(kernel_types)#was not here before
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=0123456789/ --psm 6')
    
    
    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

def random_odd_read(img, top_down, left_right):
    """
    tested with red background and 2 and 1 digit, red background
    """

    scale = random.uniform(1,2)
    kernel_types = [cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), np.zeros((round(random.uniform(1,2)),2),dtype=np.uint8)]

    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*scale), int((top_down[1]-top_down[0])*scale))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([150, 255, 255]))
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel = random.choice(kernel_types)
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)

    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=0123456789/ --psm 6')
    

    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt

def random_color_read(img, top_down, left_right):
    """
    tested for cream white. color_read but with random for dim and kernel
    """

    scale = random.uniform(1,2)
    kernel_types = [cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), np.zeros((round(random.uniform(1,2)),2),dtype=np.uint8)]

    #https://www.w3schools.com/python/ref_func_slice.asp#:~:text=The%20slice()%20function%20returns,slice%20only%20every%20other%20item.
    cropped = img[slice(top_down[0],top_down[1]),slice(left_right[0],left_right[1])]#https://learnopencv.com/cropping-an-image-using-opencv/
    dim = (int((left_right[1]-left_right[0])*scale), int((top_down[1]-top_down[0])*scale))#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    cropped = cv2.resize(cropped, dim,cv2.INTER_AREA)
    #cropped = bitwise_not(cropped)
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #seems to push the border instead of character
    mask = cv2.inRange(cropped, np.array([0, 0, 0]), np.array([100, 100, 100]))
    kernel = random.choice(kernel_types)
    dilate = cv2.dilate(mask, kernel, iterations=1)
    cropped = cv2.bitwise_and(dilate, mask)
    
    #kernel = np.zeros((2,2),dtype=np.uint8)
    #cropped = cv2.dilate(cropped, kernel, iterations=10)
    #_, cropped = cv2.threshold(cropped, 150,250,cv2.THRESH_BINARY)
    #kernel = np.zeros((2,1),dtype=np.uint8)
    #cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel,iterations=1)
    #cropped = cv2.dilate(cropped, kernel, iterations=5)
    #kernel = np.zeros((1,2),dtype=np.uint8)
    #cropped = cv2.erode(cropped, kernel, iterations=1)
    txt = pytesseract.image_to_string(cropped,config='tessedit_char_whitelist=0123456789 --psm 6')
    
    
    #debug
    #print(txt)
    #cv2.imshow("img", cropped)
    #cv2.waitKey(0)
    return txt