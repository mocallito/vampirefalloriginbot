#https://stackoverflow.com/questions/63559110/send-input-to-bluestacks-app-running-in-background/65330005#65330005
#https://www.youtube.com/watch?v=J3fatZ2OVIU
#https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
#https://docs.python.org/3/tutorial/classes.html
#Z,X,C,V,=,],#,Alt+1,I,F1
#import pytesseract as tess
#tess.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
from PIL import Image, ImageEnhance, ImageOps

from pynput.keyboard import Key, Controller
import time
import random
import win32api, win32con, win32gui, win32ui
#import ppadb
from ctypes import windll
import logging

# Set up logging configuration
# A logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(__name__+'.log','a')
#fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

# Define constants for key codes
SHIFT = win32con.VK_SHIFT
ESC = win32con.VK_ESCAPE

"""
The following code block defines some helper functions and an example of using those functions to interact with a window.
"""

# Example main function that interacts with a specific window (e.g., BlueStacks)
"""
def main():
    # Uncomment the line below to list all window names
    #list_window_names()
    # Specify the window name (e.g., BlueStacks or ...)
    #window_name = "*Untitled - Notepad"
    window_name = "BlueStacks"
    hwnd = win32gui.FindWindow(None, window_name)
    # Find the inner windows and target the specific one (e.g., EDIT control)
    for i in get_inner_windows(hwnd):
        if 'EDIT' in i:
            hwnd = get_inner_windows(hwnd)[i]
            break
    #hwnd = get_inner_windows(hwnd)['Edit']#meant for Notepad
    win = win32ui.CreateWindowFromHandle(hwnd)
    win.SendMessage(win32con.WM_CHAR, ord('i'), 0)

# Function to list all visible window names
def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), '"' + win32gui.GetWindowText(hwnd) + '"')
    win32gui.EnumWindows(winEnumHandler, None)

# Function to get inner windows of a specific window handle
def get_inner_windows(whndl):
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            hwnds[win32gui.GetClassName(hwnd)] = hwnd
        return True
    hwnds = {}
    win32gui.EnumChildWindows(whndl, callback, hwnds)
    return hwnds

# Function to simulate mouse click at specific coordinates
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
"""

#leaving the window unmoved
#screen shot shortcut ctrl+shift+s
#standard screenshot has dimension 1280x720 
#standard name is com.earlymorningstudio.vampire_Screenshot_2023.01.09_10.15.36
#maybe python open file on the newest https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder

#always the preparing for battle with the black back ground
#make function for each field and have them different input
# Function to process an image and extract text using OCR
"""
img = Image.open('battle.png')
allowed_nr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/']
# Clear the alpha channel if present
if(img.mode == 'RGBA'):
    img.load()
    r, g, b, a = img.split()
    img = Image.merge('RGB', (r, g, b))

# Optional: Invert all colors
# img = ImageOps.invert(img)

# Crop the image to focus on the players bar
left = 0
top = 50
right = 430
bottom = 100

img = img.crop((left, top, right, bottom))
txt = tess.image_to_string(img, config='--psm 6')
for i in txt:
    if i in allowed_nr:
        print(i)
"""

# Define a class to handle input operations
class input:

    # Initialize the input object with the target window name
    def __init__(self,name):
        logger.info("Activate")
        hwnd = win32gui.FindWindow(None, name)
        hwndChild = win32gui.GetWindow(hwnd, win32con.GW_CHILD)
        hwndChild2 = win32gui.GetWindow(hwndChild, win32con.GW_CHILD) #from stackoverflow
        #controll = Controller()#makes pickle error
        self.hwndChild2 = hwndChild2 # no idea to change name, handle to the child window
        self.hwnd = hwnd # no idea to change name either, handle to the main window
        #self.controll = controll#makes pickle error
        win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)
        #time.sleep(1)

    # Method to send a key press to the window
    def Keypress(self,command):
        logger.info(command)
        win32gui.SendMessage(self.hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)
        #win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, ord(command), 0)
        win32api.SendMessage(self.hwndChild2, win32con.WM_KEYDOWN, ord(command), 0)
        time.sleep(0.5)
        #win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, ord(command), 0)
        win32api.SendMessage(self.hwndChild2, win32con.WM_KEYUP, ord(command), 0)
        time.sleep(2)

    # Method to send a key combination to the window
    def Keycombination(self,special,command):
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, ord(special), 0)
        time.sleep(.3)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, ord(command), 0)
        time.sleep(.5)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, ord(command), 0)
        time.sleep(.3)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, ord(special), 0)
        time.sleep(.5)
    
    # Method to send a shortcut specific to BlueStacks
    def Bluestackshorts(self,special,addition,command):
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, special, 0)
        time.sleep(.2)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, addition, 0)
        time.sleep(.2)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, command, 0)
        time.sleep(.5)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, command, 0)
        time.sleep(.2)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, addition, 0)
        time.sleep(.2)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, special, 0)
        time.sleep(.5)

    # Method to take a screenshot of the window
    def screenshot(self):#https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)
        #print (result)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        if result == 1:
            #PrintWindow Succeeded
            im.save("main.png")
            #im.save("method.png")
            #time.sleep(1)
        #logger.info("Screen capture")
        #time.sleep(5)

    # Alternative method to take a screenshot using a key combination
    def alt_screenshot(self):#https://nitratine.net/blog/post/simulate-keypresses-in-python/
        self.controll.press(Key.ctrl)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYDOWN, ord('0'), 0)
        win32api.PostMessage(self.hwndChild2, win32con.WM_KEYUP, ord('0'), 0)
        self.controll.release(Key.ctrl)
        time.sleep(3)

#used this during testing
if __name__ == '__main__':
    hwnd = win32gui.FindWindow(None, 'BlueStacks 1 N-64')
    hwndChild = win32gui.GetWindow(hwnd, win32con.GW_CHILD)
    hwndChild2 = win32gui.GetWindow(hwndChild, win32con.GW_CHILD)
    
    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)

    time.sleep(1) # Without this delay, inputs are not executing in my case

    def keypress(hwndChild2,command):
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, command, 0)
        time.sleep(.5)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, command, 0)
        time.sleep(.5)

    def keycombination(hwndChild2,special,command):
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, special, 10)
        #time.sleep(.2)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, command, 10)
        #time.sleep(.5)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, command, 10)
        #time.sleep(.2)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, special, 10)
        #time.sleep(.5)

    def bluestackshorts(hwndChild2,special,addition,command):
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, special, 0)
        time.sleep(.1)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, addition, 0)
        time.sleep(.1)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, command, 0)
        time.sleep(.5)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, special, 0)
        time.sleep(.1)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, addition, 0)
        time.sleep(.1)
        win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, command, 0)
        time.sleep(.5)
    def screenshot():
        controll = Controller()
        controll.press(Key.ctrl)
        controll.press('0')
        controll.release('0')
        controll.release(Key.ctrl)
    """
    #tab key
    win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, win32con.VK_TAB, 0)
    time.sleep(.5)
    win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, win32con.VK_TAB, 0)
    """
    #keypress(hwndChild2,win32con.VK_TAB)
    # time.sleep(2)
    
    # """
    # #tab key down
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, ord('I'), 0)
    # time.sleep(.5)
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, ord('I'), 0)
    # """
    # keypress(hwndChild2,ord('I'))
    # """
    # #3 key
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, ord('3'), 0)
    # time.sleep(.5)
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, ord('3'), 0)
    # """
    # keypress(hwndChild2,ord('3'))
    # keypress(hwndChild2,ord('1'))
    # """
    # #shift+5 combinations
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, win32con.VK_SHIFT, 0)
    # time.sleep(.5)
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, ord('5'), 0)
    # time.sleep(.5)
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, ord('5'), 0)
    # time.sleep(.5)
    # win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, win32con.VK_SHIFT, 0)
    # """
    controll = Controller()
    controll.press(Key.ctrl)
    win32api.PostMessage(hwndChild2, win32con.WM_KEYDOWN, ord('0'), 0)
    win32api.PostMessage(hwndChild2, win32con.WM_KEYUP, ord('0'), 0)
    controll.release(Key.ctrl)
    #keypress(hwndChild2,ord('A'))
    #keypress(hwndChild2,ord('S'))
    #keypress(hwndChild2,ord('D'))
    #keypress(hwndChild2,ord('S'))
    #keypress(hwndChild2,ord('A'))
    #win32api.PostMessage(hwndChild2, win32con.WM_SYSKEYDOWN, Key.F, 2^29)
    #keycombination(hwndChild2,win32con.VK_CONTROL,ord('0'))
    #win32api.SendMessage(hwndChild2, win32con.WM_COMMAND, win32con.VK_SHIFT, 10)
    #keycombination(hwndChild,win32con.WM_KEYDOWN,win32con.VK_SNAPSHOT)
    #keycombination(hwndChild,win32con.WM_KEYDOWN,win32con.VK_SNAPSHOT)
    #time.sleep(1)
    #keypress(hwndChild2,win32con.VK_SNAPSHOT)
    #keycombination(hwndChild,win32con.VK_CONTROL,win32con.VK_F1)
    #bluestackshorts(hwndChild,win32con.VK_CONTROL,win32con.VK_SHIFT,ord('I'))
    #bluestackshorts(hwndChild,ord('D'),ord('A'),ord('S'))