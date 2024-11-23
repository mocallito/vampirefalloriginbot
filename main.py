import multiprocessing
import time
import logging
import os
import threading

import utils
#import readimg_old as parser
import parser
import click
import opponent
#import methods_old as methods
import methods
import multilifoqueue

# Maximum size for the queue in bytes
maxqueue = 8 #size in byte

# Lock for synchronizing thread access to shared resources
lock = threading.Lock()

def main():#multiple thread but same file
    """Main function to handle multiple threads that read and log file contents."""
    #battle_finnished = [False]
    log = logging.getLogger('debug')
    log.setLevel(logging.INFO)

    # Create file handler for logging which logs even debug messages
    fileh = logging.FileHandler('all.log','w')
    #fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formattr = logging.Formatter('parent:%(thread)d-child:%(message)s')
    fileh.setFormatter(formattr)

    # Add handler to the logger
    log.addHandler(fileh)

    # Iterate over files in the specified directory
    for filename in os.listdir(os.getcwd()): #https://stackoverflow.com/questions/5137497/find-the-current-directory-and-files-directory
            root, ext = os.path.splitext(filename)
            if ext == '.log' and root != 'all':
                #print (filename)
                logfile = open(filename, "r")
                t = threading.Thread(target = printFile, args = (logfile,log))
                t.start()
    
    # Continuously refresh the logger handlers
    while True:
        for handler in log.handlers:
            log.removeHandler(handler)
            handler.close()
        fileh = logging.FileHandler('all.log','a')
        #fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formattr = logging.Formatter('parent:%(thread)d-child:%(message)s')
        fileh.setFormatter(formattr)
        log.addHandler(fileh)
        time.sleep(1)
    
def follow(thefile):
    """Generator function that yields new lines from a file as they are written."""
    thefile.seek(0,os.SEEK_END) # Move to the end of the file
    #thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line or not line.endswith('\n'):
            time.sleep(0.1)
            continue
        yield line

def printFile(logfile, log):
    """Function to print lines from a logfile with thread synchronization."""
    loglines = follow(logfile)
    for line in loglines:
        #only one thread at a time can print to the user
        """temp = []
        for word in line:
            #log.info(str([ord(c) for c in word]))
            if ord(word) != 10:
                temp.append(word)"""
        lock.acquire()
        #log.info(''.join(temp))
        log.info(line)
        #print(line)#fine
        lock.release()

def makeprocess(name, handle, battle_end, latest_life, battle_variables, mutex):#handle is missleading it should be picture name
    """Function to create a new multiprocessing process for combat."""
    process = multiprocessing.Process(target=methods.combatv2,args=[name, handle, battle_end, latest_life, battle_variables, mutex])
    #process = multiprocessing.Process()
    #p1 = multiprocessing.Process(target=f)
    return process

if __name__ == "__main__":
    # Uncomment and configure these lines if needed
    #handle = click.input("BlueStacks")
    
    # Uncomment and configure this section if needed
    """handle = click.input("BlueStacks 1 N-64")
    temp = [False]
    process = makeprocess(handle,temp)
    debounce = False
    
    while True:
        if temp[0]:
            print("Finnished")
            process.kill()
            time.sleep(10)
            methods.take_reward()#take reward Alt+1
            debounce = True
        elif not temp[0] and not debounce and not process.is_alive():
            process.start()
            debounce = True
        #after the elif below the elif above runs 
        elif not temp[0] and debounce and not process.is_alive():
            process = makeprocess(handle, temp)
            process.start()#make new process because old one cant be restarted
            debounce = False
        time.sleep(5)"""
    
    # Reset other loggers
    fh = logging.FileHandler('methods.log','w')
    fh = logging.FileHandler('click.log','w')
    
    # Create logger for 'Autocombat'
    # create logger with 'spam_application'
    logger = logging.getLogger('Autocombat')
    logger.setLevel(logging.INFO)

    # Create file handler for the logger which logs even debug messages
    fh = logging.FileHandler('main.log','w')
    #fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s\n- %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(fh)

    # Uncomment and configure these lines if needed
    #handle = click.input("BlueStacks 1 N-64")
    handle = click.input("BlueStacks")
    finish = multiprocessing.Event()
    latest_life = multiprocessing.Queue()
    left_over = multiprocessing.Queue() # queue.Queue doesn't work on Windows
    mutex = multiprocessing.Lock()
    handle.screenshot()
    #left_over.put([methods.rounds[0],methods.sharp[0]])
    left_over.put([utils.rounds[0], utils.sharp[0], utils.bat_swarm[0], utils.arm_low[0], utils.more_arm[0], utils.freeze[0], utils.to_swarm])

    # Create the initial process for combat
    process = makeprocess(utils.combat_picture, handle, finish, latest_life, left_over, mutex)
    debounce = False
    finish.set()
    name = ""
    old_name = ""
    tagged = False
    waiting = False
    #main()

    # Start the main logger process
    log_process = multiprocessing.Process(target=main)
    log_process.start()

    execution_time = 0.0
    read_at = 0.0
    while True:
            with mutex:
                handle.screenshot()

            # Wait 10s before read_name when process is alive otherwise go ham
            #if listen_click.poll():  # Check if there's data to receive
            #    execution_time = listen_click.recv()
            ground_truth = []
            if process.is_alive():
                if int(time.time()-read_at) > 30:
                    for i in range(5):
                        try:
                            with mutex:
                                name = parser.read_name(utils.main_picture)
                        except Exception as e:
                            logger.exception("Failed" + str(e))
                            #continue
                        ground_truth.append(name)
                    read_at = time.time()
            else:
                for i in range(5):
                    try:
                        with mutex:
                            name = parser.read_name(utils.main_picture)
                    except Exception as e:
                        logger.exception("Failed" + str(e))
                        #continue
                    ground_truth.append(name)
                read_at = time.time()
            if utils.get_most_accurate_name(ground_truth) > 0.3:
                '''
                # Average q-gram length
                avg_q = round(sum([len(el) for el in ground_truth])/len(ground_truth))
                    
                # Extract q-gram frequencies and their positions from phrases
                q_gram_counter, q_gram_positions = utils.extract_q_gram_frequencies_and_positions(ground_truth, avg_q)
                
                for position in q_gram_positions.values():
                    if len(position) > 0:
                        # Reconstruct the most accurate phrase
                        utils.name.clear()
                        utils.name.append(utils.reconstruct_phrase(q_gram_counter, q_gram_positions, avg_q))
                        break
                '''
                tagged = True
            else:
                tagged = False
            #tagged = False if name == "" else True
            #name = parser.read_name(file)
            try:
                with mutex:
                    parser.reward(utils.main_picture)
            except Exception as e:
                logger.exception("Failed" + str(e))
                #continue

            #logger.info("Opponent " + str([ord(c) for c in name]))#https://stackoverflow.com/questions/8452961/convert-string-to-ascii-value-python
            #logger.info("Opponent exists " + str(tagged))
            #print(name)

            if tagged:# and (not len(parser.sweep(file,"Normal")[0]) == 0, methods.actions, methods.four_last[8:]):
                if not debounce:
                    if utils.cosine_similarity(old_name, name, utils.q) < 0.3:
                        while left_over.empty():
                            left_over.get()
                        left_over.put([utils.rounds[0], utils.sharp[0], utils.bat_swarm[0], utils.arm_low[0], utils.more_arm[0], utils.freeze[0], utils.to_swarm])
                        old_name = name
                    logger.info("child started")
                    process.start()
                    debounce = True

                if not finish.is_set():
                    logger.info("killing child")
                    #time.sleep(10)#used sleep for a while
                    #methods.take_reward(handle)#take reward Alt+1
                    handle.Keypress('G')
                    #process.kill() #have always used killing
                    process.join()
                    process = makeprocess(utils.combat_picture, handle, finish, latest_life, left_over, mutex)
                    #process.start()
                    debounce = False
                    handle.Keypress('4')
                    handle.Keypress('K')
                    handle.Keypress('H')
                    handle.Keypress('Y')
                    finish.set()
                #time.sleep(5)

            if not process.is_alive():
                #handle.Keypress('A')
                handle.Keypress('4')
                handle.Keypress('E')
                handle.Keypress('L')
                time.sleep(1)

            if parser.battle_finished[0]:#last resort, might fk the logics
                parser.battle_finished.clear()
                parser.battle_finished.append(False)
                handle.Keypress('4')
                handle.Keypress('Y')
                handle.Keypress('G')
                handle.Keypress('H')
                handle.Keypress('Y')
                #left_over.put([1,0])#here might be unreliable
                if process.is_alive():#actually might never called
                    left_over.put([1,0,0,0,0,0,[]])#maybe clear before put
                    logger.info("killing left alive child, expected never call")
                    #time.sleep(10)#used sleep for a while
                    #methods.take_reward(handle)#take reward Alt+1
                    #process.kill() #have always used killing
                    process.join()
                    process = makeprocess(utils.combat_picture, handle, finish, latest_life, left_over, mutex)
                    #process.start()
                    debounce = False
                    finish.set()
                    
            #if process.is_alive() and int(time.time()-execution_time) > 180 and execution_time > 1e-05:# and not  tagged:
            ''' if process is alive and scan saw nothing then kill
                but except when the process was still running
                more difficult since now include mutexes and queues.
                Topping it of with execution time
            '''
            if process.is_alive() and not tagged:
                if not latest_life.empty():
                    message = latest_life.get()
                    execution_time = message[1]
                    waiting = message[0] == False
                if waiting and execution_time > 1e-05 and int(time.time() - execution_time) > 60:
                    #left_over.put([1,0])#maybe clear before put
                    #left_over.put(None)
                    logger.info("killing left alive child")
                    #time.sleep(10)#used sleep for a while
                    #methods.take_reward(handle)#take reward Alt+1
                    #process.kill() # have always used killing
                    process.join()
                    process = makeprocess(utils.combat_picture, handle, finish, latest_life, left_over, mutex)
                    #process.start()
                    execution_time = 0.0
                    debounce = False
                    waiting = False
                    handle.Keypress('4')
                    handle.Keypress('K')
                    handle.Keypress('H')
                    handle.Keypress('Y')
                    finish.set()
                    
            # tagged = name != ""
            time.sleep(1)
    """
    #in ms
keyDown Up
wait 100
keyDown Down
keyUp Up
wait 100
keyDown Right
keyUp Down
wait 100
keyDown Left
keyUp Right
wait 100
keyUp Left

keyDown Up
wait 700

keyDown Down
keyUp Up
wait 700

keyDown Right
keyUp Down
wait 700
  
keyDown Left
keyUp Right
wait 700
keyUp Left

keyDown Left
keyDown Up
wait 700
keyUp Left
keyUp Up
"""