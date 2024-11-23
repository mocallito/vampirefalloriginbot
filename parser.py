import logging
import readimg
import random
import math
import threading
import queue
import utils
import time

# Set up logger
logger = logging.getLogger("methods." + __name__)

# Allowed characters for signs and numbers
allowed_sign = ['0','1','2','3','4','5','6','7','8','9','/','-',' ']
allowed_nr = ['0','1','2','3','4','5','6','7','8','9']
# Characters to exclude during OCR processing
#exceptions = ['y', 'o', 'u', 'h', 'a', 'v', 'e', 'u', 'n', 'c', 'o', 'l', 'l', 'e', 'c', 't', 'e', 'd',
#            'l', 'o', 'o', 't', '.', 'l', 'e', 'a', 'v', 'e', 'a', 'n', 'y', 'w', 'a', 'y', '?']
exceptions = 'leave anyway?'

name_exceptions = 'ina'
# Maximum digits allowed
max_digit = 4
# Characters that can be confused and their correct replacements
confused = {'Q':'0', 'O':'0', '@':'0', 'B':'8', 'I':'1', 'T':'1', 'l':'1', 'S':'5', '§':'5', '*':'-',
'°':'-', '\'':'1', '+':'-', '[': '1'}
# A flag indicating if the battle is finished
battle_finished = [False]
# Path to the image gallery
gallery = r"C:\Users\tridao\Pictures\BlueStacks\*png"
#maybe | instead of blank

def find_in_confused(char):
    if char in confused:
        char = confused[char]
    if char in [key.lower() for key in list(confused.keys())]:
        char = confused[char.capitalize()]
    return char

# Scan the text and return an array 
def scan(txt, split_by, max=3):
    '''
    Cleaning the text, mostly for health, mana and dmg_left.
    Finally splitting the text and return the left side, which often means current state of the game.
    '''
    array = []
    skipping = False
    for i in txt:
        if not skipping:
            i = find_in_confused(i)
            if i in allowed_sign:
                if i != split_by and len(array)<max and i != ' ':
                    array.append(i)
                    #print(i)#debug
                elif i == split_by:
                    skipping = True
        elif skipping:
            if i in allowed_sign:
                if i != ' ':
                    continue
                elif i == ' ':
                    skipping = False
    array = ''.join(array)#https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python#:~:text=To%20convert%20a%20list%20to%20a%20string%2C%20use%20Python%20List,and%20return%20it%20as%20output.
    utils.normalize(array, r'[^\d\s/-]')
    array = array.split(split_by)
    return array

# Extract relevant data from the image based on the reason parameter
# focusing on one line at a time although atleast 30 pixels for height
def data_colect(img, reason, actions=None):#https://www.geeksforgeeks.org/how-to-pass-multiple-arguments-to-function/
    '''
    1.Calling functions from readimg and focusing on regions of the image.
    2.The text returned is often dirty and require cleaning. There are many unessecary words and empty spaces.
    3.Finally the text is cleaned through multiple process and important data are saved in the local variables.
    4.Repeat the process for other specified regions.
    '''
    #img = cv2.imread(name)
    #print(img.shape)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    split_by = ''
    if "Normal" in reason:
        #txt = color_read(img, [int(img.shape[0]/1.25),int(img.shape[0]/1.17)], [int(img.shape[1]/5),int(img.shape[1]/1.85)])
        txt = readimg.color_read(img, [int(img.shape[0]/1.25*random.uniform(0.995,1.005)),int(img.shape[0]/1.17*random.uniform(0.995,1.005))], [int(img.shape[1]/5*random.uniform(0.995,1.005)),int(img.shape[1]/1.85*random.uniform(0.995,1.005))])
        txt = ' '.join(txt.split('|'))
        txt = txt.split(' ')
        #print(txt)
        split_by = '-'
    elif "Combo" in reason:
        txt = readimg.color_read(img, [int(img.shape[0]/1.12),int(img.shape[0]/1.08)], [int(img.shape[1]/6.5),int(img.shape[1]/1.7)])
        #print(txt)
        txt = ' '.join(txt.split('|'))
        txt = txt.split(' ')
        split_by = '-'
        text = readimg.blue_read(img, [int(img.shape[0]/1.08),int(img.shape[0]/1.01)], [int(img.shape[1]/7),int(img.shape[1]/1.6)])#cant read focus because its blue
        focus = ' '.join(text.split('|'))
        focus = focus.split(' ')
        array = [scan(dmg,split_by, 2) for dmg in focus]
        #focus = focus[0].split(' ')
        while [''] in array:
            array.remove([''])
        logger.info(str(array))
        #temporare order
        #focus = text.split(' ')
        if len(array) < 4:
            array.insert(1,['0'])#one hand combo
        if len(array) < 4:#currently is str
            text = readimg.red_read(img, [int(img.shape[0]/1.06),int(img.shape[0]/1.02)], [int(img.shape[1]/7.5),int(img.shape[1]/1.8)])
            focus = ' '.join(text.split('|'))
            focus = focus.split(' ')
            array = [scan(dmg,split_by, 2) for dmg in focus]
        logger.info(str(array))
        #order base on command on bluestack
        actions["A+Z"] = int(array[0][0])
        actions["A+X"] = int(array[1][0])
        actions["A+C"] = int(array[2][0])
        actions["A+V"] = int(array[3][0])
        actions["D+Z"] = int(array[0][0])
    array = [scan(dmg,split_by, max_digit) for dmg in txt]
    while [''] in array:
        array.remove([''])
    if len(array) == 3:
        array.insert(1,['0'])
    split_by = '/'
    
    #entering battle texts always has an 'A' 420:490, 300:880 then black background, pixel 640, 90
    #the character name is often read as TRINATLINA
    #print(array)#debug

    #player health bar
    txt = readimg.odd_read(img, [int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))], [0, int(img.shape[1]/10.38*random.uniform(0.995,1.005))])
    print(txt)
    health = scan(txt,split_by, max_digit)
    """for digit in health[0]:
        if digit not in allowed_nr:# change to not number
            txt = red_read(img, [int(img.shape[0]/153),int(img.shape[0]/15.32)], [0, int(img.shape[1]/10.38)])
            health = scan(txt,split_by, max_digit)
            break"""
    #print(health)
    #player focus bar
    txt = readimg.random_white_read(img, [int(img.shape[0]/10.89*random.uniform(0.995,1.005)),int(img.shape[0]/8.3*random.uniform(0.995,1.005))], [int(img.shape[1]*0.01*random.uniform(0.995,1.005)), int(img.shape[1]/10.38*random.uniform(0.995,1.005))])
    print(txt)
    focus = scan(txt,split_by, max_digit)
    """for digit in focus[0]:
        if digit not in allowed_nr:#change to not number
            txt = red_read(img, [int(img.shape[0]/13),int(img.shape[0]/9)], [int(img.shape[1]/1.2),int(img.shape[1]/1.05)])
            dmg_left = scan(txt,split_by, max_digit)
            break"""
    #print(focus)
    #oponent
    txt = readimg.odd_read(img, [int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))], [int(img.shape[1]/1.12*random.uniform(0.995,1.005)),int(img.shape[1]/0.99*random.uniform(0.995,1.005))])
    print(txt)
    dmg_left = scan(txt,split_by, max_digit)
    #print(dmg_left)
    """for digit in dmg_left[0]:
        if digit not in allowed_nr:#change to not number
            txt = red_read(img, [int(img.shape[0]/13*random.uniform(0.995,1.005)),int(img.shape[0]/9*random.uniform(0.995,1.005))], [int(img.shape[1]/1.2*random.uniform(0.995,1.005)),int(img.shape[1]/1.05*random.uniform(0.995,1.005))])
            dmg_left = scan(txt,split_by, max_digit)
            break"""
        #print(txt)
    #print(dmg_left)
    logger.info("health " + str(health) + " focus " + str(focus) + " dmg_left " + str(dmg_left))
    return array, health, focus, dmg_left

def all_in_allowed (arr):
    if len(arr) > 0:
        return all(digit in allowed_nr for digit in arr)
    return False

def identify_turn(img):
    '''
    quest = readimg.name_read(img, [int(img.shape[0]/76.8*random.uniform(0.995,1.005)),int(img.shape[0]/19.2*random.uniform(0.995,1.005))], [int(img.shape[1]/2.4*random.uniform(0.995,1.005)),int(img.shape[1]/2.09*random.uniform(0.995,1.005))])
    vampire = readimg.name_read(img, [int(img.shape[0]/76.8*random.uniform(0.995,1.005)),int(img.shape[0]/19.2*random.uniform(0.995,1.005))], [int(img.shape[1]/1.94*random.uniform(0.995,1.005)),int(img.shape[1]/1.71*random.uniform(0.995,1.005))])
    quest_exist = utils.cosine_similarity(quest.lower(), 'quest', utils.q) > 0.5
    vampire_exist = utils.cosine_similarity(vampire.lower(), 'vampire', utils.q) > 0.5
    if not(vampire_exist or quest_exist):
    '''

    school = []
    school.append(readimg.name_read(img, [int(img.shape[0]/1.35*random.uniform(0.995,1.005)),int(img.shape[0]/1.3*random.uniform(0.995,1.005))], [int(img.shape[1]/4.14*random.uniform(0.995,1.005)),int(img.shape[1]/3.59*random.uniform(0.995,1.005))]))
    school.append(readimg.name_read(img, [int(img.shape[0]/1.35*random.uniform(0.995,1.005)),int(img.shape[0]/1.3*random.uniform(0.995,1.005))], [int(img.shape[1]/2.1*random.uniform(0.995,1.005)),int(img.shape[1]/1.89*random.uniform(0.995,1.005))]))
    #total_length = sum(len(s) for s in school)
    combo_turn = ['left','right']
    normal_turn = ['main','dual']
    is_combo = 0
    is_normal = 0
    for style in school:
        #print(style)
        if len(style) > 0 and utils.cosine_similarity(style.lower(), combo_turn[school.index(style)], utils.q) > 0.5:
            is_combo += 1
        if len(style) > 0 and utils.cosine_similarity(style.lower(), normal_turn[school.index(style)], utils.q) > 0.5:
            is_normal += 1
    
    if is_combo > is_normal:
        #queue.put(('reason', "Combo"))
        return "Combo"
    elif is_combo < is_normal:
        #queue.put(('reason', "Normal"))
        return "Normal"
    
    '''
    else:
        #queue.put(('reason', "Ended"))
        return False
    '''

# Extract relevant data from the image based on the reason parameter
# focusing on one line at a time although atleast 30 pixels for height
def data_colect_dac(img, reason, actions=None):#https://www.geeksforgeeks.org/how-to-pass-multiple-arguments-to-function/
    '''
    Like data_colect but a thread for each region, divide and conquer(DAC)

    1.Calling functions from readimg and focusing on regions of the image.
    2.The text returned is often dirty and require cleaning. There are many unessecary words and empty spaces.
    3.Finally the text is cleaned through multiple process and important data are saved in the local variables.
    4.Repeat the process for other specified regions.
    '''
        
    def collect_dmg (reason, queue, errors):
        nonlocal img, actions
        start = time.time()
        array = []
        while not (len(array) >= 4):
            array = []
            if "Normal" in reason:
                actions["A+Z"] = 0
                actions["A+X"] = math.inf
                actions["A+C"] = 20
                actions["A+V"] = math.inf
                reconstructing = []
                for i in range(5):
                    txt = readimg.random_color_read(img, [int(img.shape[0]/1.25*random.uniform(0.995,1.005)),int(img.shape[0]/1.17*random.uniform(0.995,1.005))], [int(img.shape[1]/5*random.uniform(0.995,1.005)),int(img.shape[1]/1.85*random.uniform(0.995,1.005))])
                    #print("In ", txt)
                    txt = ''.join([find_in_confused(char) for char in txt])
                    reconstructing.append(utils.normalize(txt, r'[^\d\s-]'))

                # Average q-gram length
                avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))

                # Extract q-gram frequencies and their positions from phrases
                q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q, False)

                # Reconstruct the most accurate phrase
                txt = utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q)
                #print("Pre ", txt)
                txt = ' '.join(txt.split('|'))
                txt = txt.split(' ')
                for el in txt:
                    if '-' not in el:
                        txt.remove(el)
                #print("Post ", txt)

                split_by = '-'
            elif "Combo" in reason:
                reconstructing = []
                for i in range(5):
                    txt = readimg.random_color_read(img, [int(img.shape[0]/1.12),int(img.shape[0]/1.08)], [int(img.shape[1]/6.5),int(img.shape[1]/1.7)])
                    #print(txt)
                    txt = ''.join([find_in_confused(char) for char in txt])
                    reconstructing.append(utils.normalize(txt, r'[^\d\s/-]'))

                # Average q-gram length
                avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))

                # Extract q-gram frequencies and their positions from phrases
                q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q, False)

                # Reconstruct the most accurate phrase
                txt = utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q)

                #print(txt)
                txt = ' '.join(txt.split('|'))
                txt = txt.split(' ')
                split_by = '-'
                reconstructing = []
                for i in range(5):
                    text = readimg.blue_read(img, [int(img.shape[0]/1.08),int(img.shape[0]/1.01)], [int(img.shape[1]/7),int(img.shape[1]/1.6)])#cant read focus because its blue
                    reconstructing.append(text)

                # Average q-gram length
                avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))

                # Extract q-gram frequencies and their positions from phrases
                q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q, False)

                # Reconstruct the most accurate phrase
                text = utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q)

                focus = ' '.join(text.split('|'))
                focus = focus.split(' ')
                all_focus = [scan(dmg,split_by, 2) for dmg in focus]
                #focus = focus[0].split(' ')
                while [''] in all_focus:
                    all_focus.remove([''])
                #temporare order
                #focus = text.split(' ')
                if utils.one_hand and len(all_focus) < 4:
                    all_focus.insert(1,['0'])#one hand combo
                '''
                if len(all_focus) < 4:#currently is str
                    text = readimg.red_read(img, [int(img.shape[0]/1.06),int(img.shape[0]/1.02)], [int(img.shape[1]/7.5),int(img.shape[1]/1.8)])
                    focus = ' '.join(text.split('|'))
                    focus = focus.split(' ')
                    all_focus = [scan(dmg,split_by, 2) for dmg in focus]
                '''
                #logger.info("Focus "+str(all_focus))
                #order base on command on bluestack
                for i, key in enumerate(list(actions.keys())[-4:]):
                    replace = int(all_focus[i][0])
                    if math.isclose(replace, actions[key], abs_tol=10):
                        actions[key] = replace
                if utils.one_hand:
                    actions["A+X"] = math.inf
            array = [scan(dmg,split_by) for dmg in txt]
            while [''] in array:
                array.remove([''])
                
            if utils.one_hand and len(array) < 4 and "Combo" in reason:
                array.insert(2,['0'])
            #if "Normal" in reason and len(array[0]) > len(array[1]):
            #    array.pop(1)
            array.append(['0'])
            array.append(['0'])
            logger.info(str(array))
            if int(time.time()-start) > 30:
                array = utils.calc_dmg(reason)
                break
                #raise Exception("Time limit reached for scanning damage")
        # if bad raiseerror
        queue.put(('array', array))
    
    def health_bar(queue):
        nonlocal img
        #player health bar
        start = time.time()
        reconstructing = []
        while True:
            for i in range(5):
                health = ['']
                while not all_in_allowed(health[0]):
                    txt = readimg.random_odd_read(img, [int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))], [0, int(img.shape[1]/10.38*random.uniform(0.995,1.005))])
                    #print(txt)
                    health = scan(txt,'/', max_digit)
                    if int(time.time()-start) > 30:
                        break
                if len(health[0]) > 0 and int(health[0]) <= utils.cap_health+30:
                    reconstructing.append(health[0])
            if len(reconstructing) > 1 or int(time.time()-start) > 30:
                break
        #print(reconstructing)
        
        # Average q-gram length
        avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))

        # Extract q-gram frequencies and their positions from phrases
        q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q)

        # Reconstruct the most accurate phrase
        health.clear()
        health.append(utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q))
        queue.put(('health', health))
    
    def focus_bar(queue):
        nonlocal img
        #player focus bar
        start = time.time()
        reconstructing = []
        while True:
            for i in range(10):
                focus = ['']
                while not all_in_allowed(focus[0]):
                    txt = readimg.random_white_read(img, [int(img.shape[0]/10.89*random.uniform(0.995,1.005)),int(img.shape[0]/8.3*random.uniform(0.995,1.005))], [int(img.shape[1]*0.01*random.uniform(0.995,1.005)), int(img.shape[1]/10.38*random.uniform(0.995,1.005))])
                    #print(txt)
                    focus = scan(txt,'/', max_digit)
                    if int(time.time()-start) > 30:
                        break
                if len(focus[0]) > 0 and int(focus[0]) <= utils.cap_focus:
                    reconstructing.append(utils.normalize(focus[0], r'[^\d\s/]'))
            if len(reconstructing) > 1 or int(time.time()-start) > 60:
                break
        #print(reconstructing)

        # Average q-gram length
        avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))

        # Extract q-gram frequencies and their positions from phrases
        q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q)

        # Reconstruct the most accurate phrase
        focus.clear()
        focus.append(utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q))
        queue.put(('focus', focus))

    def dmg_left_bar(queue):
        nonlocal img
        #oponent
        start = time.time()
        reconstructing = []
        while True:
            for i in range(5):
                dmg_left = ['']
                while not all_in_allowed(dmg_left[0]):
                    txt = readimg.random_odd_read(img, [int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))], [int(img.shape[1]/1.12*random.uniform(0.995,1.005)),int(img.shape[1]/0.99*random.uniform(0.995,1.005))])
                    #print(txt)
                    dmg_left = scan(txt,'/', max_digit)
                    if int(time.time()-start) > 30:
                        break
                reconstructing.append(dmg_left[0])
            if len(reconstructing) > 1 or int(time.time()-start) > 30:
                break
            
        # Average q-gram length
        avg_q = round(sum([len(el) for el in reconstructing])/len(reconstructing))
            
        # Extract q-gram frequencies and their positions from phrases
        q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(reconstructing, avg_q)

        # Reconstruct the most accurate phrase
        dmg_left.clear()
        dmg_left.append(utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q))
        queue.put(('dmg_left', dmg_left))
    #img = cv2.imread(name)
    #print(img.shape)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    split_by = ''
    
    # Create a queue to collect results from threads
    result_queue = queue.Queue()

    error_msg = []

    # Create threads and assign them tasks
    thread1 = threading.Thread(target = collect_dmg, args = (reason, result_queue, error_msg))
    thread2 = threading.Thread(target = health_bar, args = (result_queue, ))
    thread3 = threading.Thread(target = focus_bar, args = (result_queue, ))
    thread4 = threading.Thread(target = dmg_left_bar, args = (result_queue, ))
    
    #entering battle texts always has an 'A' 420:490, 300:880 then black background, pixel 640, 90
    #the character name is often read as TRINATLINA
    #print(array)#debug

    # Start threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    # Join threads, waiting for them to complete
    thread1.join()
    #logger.info("dmg")
    thread2.join()
    #logger.info("health")
    thread3.join()
    #logger.info("focus")
    thread4.join()
    #logger.info("dmg_left")

    return_queue = []
    while not result_queue.empty():
        return_queue.append(result_queue.get())

    return tuple(i[1] for i in sorted(return_queue))

# Read the name from the screen
def read_name(screen):
    '''
    Read both opponent and the bottom left corner. If both exist then combat started.
    Note: the name_read for bottom left corner is not perfect.
    '''
    #screen.screenshot()
    txt = ''
    img = readimg.read_screen(screen)

    txt = readimg.name_read(img, [int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))], [int(img.shape[1]/1.48*random.uniform(0.995,1.005)),int(img.shape[1]/1.13*random.uniform(0.995,1.005))])

    school = []
    '''
    school.append(readimg.name_read(img, [int(img.shape[0]/1.26*random.uniform(0.995,1.005)), int(img.shape[0]/1.21*random.uniform(0.995,1.005))], [int(img.shape[0]/39*random.uniform(0.995,1.005)), int(img.shape[0]/5*random.uniform(0.995,1.005))]))
    school.append(readimg.name_read(img, [int(img.shape[0]/1.16*random.uniform(0.995,1.005)), int(img.shape[0]/1.11*random.uniform(0.995,1.005))], [int(img.shape[0]/39*random.uniform(0.995,1.005)), int(img.shape[0]/5*random.uniform(0.995,1.005))]))
    school.append(readimg.name_read(img, [int(img.shape[0]/1.06*random.uniform(0.995,1.005)), int(img.shape[0]/1.02*random.uniform(0.995,1.005))], [int(img.shape[0]/39*random.uniform(0.995,1.005)), int(img.shape[0]/5*random.uniform(0.995,1.005))]))
    #total_length = sum(len(s) for s in school)
    known_words = ['controll','instinct','weapon']
    '''
    school.append(readimg.name_read(img, [int(img.shape[0]/76.8*random.uniform(0.995,1.005)),int(img.shape[0]/19.2*random.uniform(0.995,1.005))], [int(img.shape[1]/2.4*random.uniform(0.995,1.005)),int(img.shape[1]/2.09*random.uniform(0.995,1.005))]))
    school.append(readimg.name_read(img, [int(img.shape[0]/76.8*random.uniform(0.995,1.005)),int(img.shape[0]/19.2*random.uniform(0.995,1.005))], [int(img.shape[1]/1.94*random.uniform(0.995,1.005)),int(img.shape[1]/1.71*random.uniform(0.995,1.005))]))
    known_words = ['quest','vampire']

    count = 0
    for style in school:
        #print(style)
        if len(style) > 0 and utils.cosine_similarity(style.lower(), known_words[school.index(style)], utils.q) > 0.5:
            count += 1
        if count <= 0: # was if count > 0:
            return txt
    '''
    #https://www.w3schools.com/python/ref_math_isclose.asp
    if (math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[0], 29, abs_tol = 5)#read
    or math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[1], 29, abs_tol = 5)
    and math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[2], 95, abs_tol = 5)#[29, 29, 95]
    or (math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[0], 0, abs_tol=5)
    and math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[1], 0, abs_tol=5)
    and math.isclose(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()[2], 0, abs_tol=5))#black
    ):
        """and not(math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[0], 29, abs_tol = 5)#read
        or math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[1], 29, abs_tol = 5)
        and math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[2], 95, abs_tol = 5)#[29, 29, 95]
        or (math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[0], 0, abs_tol=5)
        and math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[1], 0, abs_tol=5)
        and math.isclose(img[int(img.shape[0]*0.12),int(img.shape[1]*0.04)].tolist()[2], 0, abs_tol=5))"""
        #battle_finished.clear()
        #battle_finished.append(True)
        text = readimg.name_read(img,[int(img.shape[0]/153*random.uniform(0.995,1.005)),int(img.shape[0]/15.32*random.uniform(0.995,1.005))],[int(img.shape[1]*0.24*random.uniform(0.995,1.005)),int(img.shape[1]*0.32*random.uniform(0.995,1.005))])
        if name_exceptions in text.lower():
            #print(text)
            return txt
        #logger.info("Reading out of combat")
        #logger.info(str(img[int(img.shape[0]/25.53),int(img.shape[1]/1.1)].tolist()))
        #return ''#maybe also decide out of combat or not
    #if txt == '':#didnt test properly
    #    print("redo")
    #    white_read(img, [int(img.shape[0]/14),int(img.shape[0]/8)], [int(img.shape[1]/1.5),int(img.shape[1]/1.12)])
    '''
    #logger.info("Reading out of combat")
    return ''

# Check if the battle is finished based on the screen content
def reward(screen):
    txt = ''
    img = readimg.read_screen(screen)
    #txt = name_read(img,[int(img.shape[0]/2.88*random.uniform(0.995,1.005)),int(img.shape[0]/1.65*random.uniform(0.995,1.005))], [int(img.shape[1]/2.48*random.uniform(0.995,1.005)),int(img.shape[1]/1.99*random.uniform(0.995,1.005))])
    txt = readimg.name_read(img,[int(img.shape[0]/2.48*random.uniform(0.995,1.005)),int(img.shape[0]/1.99*random.uniform(0.995,1.005))], [int(img.shape[1]/2.88*random.uniform(0.995,1.005)),int(img.shape[1]/1.65*random.uniform(0.995,1.005))])
    print(txt)
    #for char in txt.lower():
    #for char in txt.lower():
        #if char in exceptions and img[int(img.shape[0]/2.7),int(img.shape[1]/2.55)].tolist()[0] == 0:
    if len(txt) > 0 and utils.cosine_similarity(txt.lower(), exceptions, utils.q) > 0.5 and img[int(img.shape[0]/2.7),int(img.shape[1]/2.55)].tolist()[0] == 0:
        battle_finished.clear()
        battle_finished.append(True)
        logger.info("Finnished "+str(battle_finished[0]))
        return

def sweep(screen, reason, actions=None, four_last=None, prev_dmg={}):#reason is combo and normal
    logger.info("sweep " + reason)
    #screen.screenshot()
    #img = cv2.imread("test.png")
    with utils.mutex:
        img = readimg.read_screen(screen)
    #if type(img) == type(None):
    #    raise Exception("None type imread")
    #img_copy = img.copy()
    #print(type(img_copy[650,250].tolist()))
    #if img_copy[650,250].tolist()[0] == 0:
    #    battle_finished.clear()
    #    battle_finished.append(True)
        #print("black found")
        #return [], [], [], []
#    if (img[300, 240] == [0,0,0]).all:#https://stackoverflow.com/questions/10062954/valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous
#        battle_finished = True
    health, focus, dmg_left = [],[],[]
    all_dmg = {}
    if reason != (right_reason := identify_turn(img)):
        # TODO: remove this, don't want 2 to break and kill child
        if not right_reason:
            raise Exception("Battle Ended")
        logger.info("Wrong round")
        reason = right_reason
    #while not (len(all_dmg) >= 4 and all_in_allowed(health[0]) and all_in_allowed(focus[0]) and all_in_allowed(dmg_left[0])):
        #while True:
    array, dmg_left, focus, health = data_colect_dac(img,reason, actions)
    
    #array = array[0].split(' ')
    logger.info(str(array) + str(health) + str(focus) + str(dmg_left))
    #array.append([0])
    #array.append([0])
    #temporare all skill dmg need to return this to method
    if four_last != None:
        for element in range(4):
            all_dmg[four_last[element]]=int(array[element][0])
    #print(all_dmg)#debug
    if len(prev_dmg)>1 and all_in_allowed(health[0]) and all_in_allowed(focus[0]) and all_in_allowed(dmg_left[0]):
        all_dmg.update(prev_dmg)
    return all_dmg, abs(int(health[0])), abs(int(focus[0])), abs(int(dmg_left[0])), reason

if __name__ == '__main__':
    import click
    """
    arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12 = "", "", "", "", "", "", "", ""
    arg1, arg2, arg3, arg4 = data_colect("battle.png", "Normal")
    #arg1, arg2, arg3, arg4 = data_colect("battle2.png", "Combo")
    #isolating scanned skill dmg
    arg1 = arg1[0].split(' ')#then if is int so save
    print(arg1, arg2, arg3, arg4)
    """
    '''
    actions =   {"D+Z": 30, "D+X": 40, "D+C": 35, "D+V": 50,
            "S+Z": 20, "S+X": 45, "S+C": 17, "S+V": 35,
            "A+Z": 0, "A+C": 20, "A+X": math.inf, "A+V": math.inf} #check keyplacement on blustacks
            '''
    #handle = click.input("BlueStacks 1 N-64")
    #handle = click.input("BlueStacks")
    #handle.alt_screenshot()
    #handle.screenshot()
    four_last = list(utils.actions.keys())#https://stackoverflow.com/questions/16819222/how-to-return-dictionary-keys-as-a-list-in-python
    #all_dmg = {key:actions[key] for key in four_last[8:]}#https://stackoverflow.com/questions/21084066/copy-part-of-dictionary-both-its-keys-and-its-value

    #print(sweep("test.png","Normal",actions, four_last[8:]))
    #temp = max(glob.glob(gallery),key=os.path.getctime)#https://datatofish.com/latest-file-python/
    #print(read_name(temp))
    #print(read_name("test.png"))
    #reward("test.png")
    #print(battle_finished[0])
    #os.remove(temp)#https://favtutor.com/blogs/delete-file-python

    img = readimg.cv2.imread(utils.main_picture)
    print(identify_turn(img))
    #print(data_colect_dac(img, "Normal", utils.actions))
    #print(img[300, 240])#[0,0,0] if black#https://www.geeksforgeeks.org/python-pil-getpixel-method/
    #txt = pytesseract.image_to_string(img,config='--psm 6')
    #print(txt)