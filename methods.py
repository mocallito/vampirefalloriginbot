import logging
import math
import numpy as np
import graphviz
import time
import random

import utils
import parser
import opponent
import aima

# Initialize logger
# create logger with 'spam_application'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create file handler which logs even debug messages
fh = logging.FileHandler(__name__ + '.log','a')
#fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s: %(process)d - %(funcName)s - %(lineno)d - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
#logger.info("Start")
#########

def update(state):
    state.round += 1
    if round%4:
        state.focus += utils.focus_increase
    elif not round%4:
        state.focus += utils.focus_combo
    state.old_focus.append(state)
    return state
#state = {'health': health, 'focus': focus, 'dmg_dealt': dmg_dealt, 'sharp': sharp}
#current_status = State(health,focus,dmg_dealt,sharp)

def enuff_focus(action):
    global focus, actions
    if utils.focus[0] >= utils.actions[action]:
        return True
    return False

def combo_true(round):#might not need
    global actions
    if not round%4:
        utils.actions["A+V"] = 0
        utils.actions["A+X"] = 0
        return True
    else:
        utils.actions["A+V"] = 0
        utils.actions["A+X"] = 0
        return False

def focusflow():#might not need
    global rounds, focus, focus_combo
    if not utils.rounds%4:
        utils.focus[0] += utils.focus_combo
    else:
        utils.focus[0] += utils.focus_increase
    return utils.focus[0]

def sharpen():
    global sharp, dmg_dealt, focus
    utils.sharp[0] += 4
    return utils.to_execute("S+Z", utils.dmg_dealt[0], utils.focus[0])

def bite(health, focus, dmg_dealt):
    global bite_dmg
    health += math.floor(utils.bite_dmg*random.uniform(0.5,1))
    focus -= utils.actions["S+X"]
    dmg_dealt += utils.bite_dmg
    return (health, focus, dmg_dealt, "S+X")

def regular(health, focus, dmg_dealt):
    global dmg_got, all_dmg, sharp
    if health <= utils.dmg_got[0]:
        #if enuff_focus("S+X"): #Bite
        #    return bite(health, focus, dmg_dealt)
        #else:#run
            return (health, focus, dmg_dealt, 'F')
    elif health >= utils.dmg_got[0]:
        if enuff_focus("S+Z") and utils.sharp[0] == 0:
            return (health,) + sharpen()
        else:
            return (health,) + utils.to_execute("A+Z",dmg_dealt, focus)#normal
    elif len(utils.dmg_left) > 0 and utils.all_dmg["A+X"] >= utils.dmg_left[0]:
        if enuff_focus("A+C"):#combine
            return (health,) + utils.to_execute("A+C",dmg_dealt, focus)

#returns a list of actions then state
def get_combo(health, focus, dmg_dealt, depth):
    global bite_dmg, focus_increase, name, dmg_left
    if depth == 7:
        return ('I', health, focus, dmg_dealt)
    for_guessing = [utils.name[0], utils.guess_dmg]
    """if cap_health-health >= bite_dmg:
        if focus >= actions["S+X"]:
            health, focus, dmg_dealt, action = bite(health, focus, dmg_dealt)
            return (action,) + get_combo(health, focus, dmg_dealt, depth+1)"""
    if len(utils.dmg_left) > 0 and dmg_dealt >= utils.dmg_left[0]:#reach goal during combo#was elif
        return ('I', health, focus, dmg_dealt)#combo finish
    elif focus >= utils.actions["A+V"]:
        focus, dmg_dealt, action = utils.to_execute("A+V",dmg_dealt, focus)
        return (action,) + get_combo(health, focus, dmg_dealt, depth+1)
    """elif health <= opponent.get_dmg(for_guessing):#focus enough for bite after combo because health <= get_dmg()
        guess_focus = focus 
        guess_focus += focus_increase
        if guess_focus >= actions["S+X"]:#bite
            return ('I', health, focus, dmg_dealt)#combo finish"""
    if focus >= utils.actions["A+Z"]:#cheap or expensive on high lv#was elif
        focus, dmg_dealt, action = utils.to_execute("A+Z",dmg_dealt, focus)
        return (action,) + get_combo(health, focus, dmg_dealt, depth+1)
    elif focus >= utils.actions["A+C"]:#stab
        focus, dmg_dealt, action = utils.to_execute("A+C",dmg_dealt, focus)
        return (action,) + get_combo(health, focus, dmg_dealt, depth+1)
    elif health <= opponent.get_dmg(for_guessing) and depth == 0:#run
        return ('F', health, focus, dmg_dealt)
    else:
        return ('I', health, focus, dmg_dealt)#combo finish

def possible_actions(state):
        #normal turn
        if state["round"]%4:
            for key in utils.actions.keys():
                if state["focus"] >= utils.actions[key]:
                    yield [key]
        #combo turn, might return a whole set of action
        elif not state["round"]%4:
            for object in utils.powerset(utils.actions):#https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
                for subobject in utils.itertools.permutations(object):#https://www.geeksforgeeks.org/permutation-and-combination-in-python/
                    required_focus = 0
                    for action in subobject:
                        required_focus += utils.actions[action]
                    if required_focus <= state["focus"]:
                        yield list(subobject)
        logger.info("finnish")

# Define the best_move function which appears to implement the minimax algorithm with alpha-beta pruning
#maybe minimax returns like get combo, but how to use to_execute
def best_move(node, is_maximizing, problem, alpha, beta, dot):
    global dmg_left, name, guess_dmg
    #for_opponent = [utils.name[0], utils.guess_dmg]
    count = 0
    best_value = -math.inf if is_maximizing else math.inf
    best_move = None

    # Inner minimax function for recursive decision making
    def minimax(node, is_maximizing, problem, alpha, beta):

        if problem.goal_test(node):
            return problem.evaluate(node)
        '''
        # Check if the current node is a terminal node
        elif ((score := problem.goal_test(node)) is not None):#is_maximizing True
            #print(node.path())
            return score
        '''
        # Update node state based on whether it is a maximizing node or not
        node.state.is_maximizing = is_maximizing

        #logger.info(str(node.action) + " " + str(node.state.round))
        
        scores = []
        if is_maximizing:
            max_eval = -math.inf
            # Iterate through child nodes
            for child in node.expand(problem):

                score = minimax(child, not is_maximizing, problem, alpha, beta)
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return max_eval
        # Iterate through child nodes
        else:
            min_eval = math.inf
            for child in node.expand(problem):
                #logger.info("It's here" + str(is_maximizing))
                score = minimax(child, not is_maximizing, problem, alpha, beta)
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return min_eval
        #logger.info("Shouldn't be here" + str(is_maximizing))
        #return 0
        
    root = aima.Node(problem.initial)

    for i,new_state in enumerate(root.expand(problem)):
        value = minimax(new_state, is_maximizing=is_maximizing, problem=problem, alpha=alpha, beta=beta)
        if (is_maximizing and value > best_value) or (not is_maximizing and value < best_value):
            best_value = value
            best_move = new_state
    return (best_value, best_move)

def score_only(the_tuple):
    return the_tuple[0]

def combo_minimax(state, depth, is_maximizing, action = ""):
    """if action in actions.keys():
        state.focus -= actions[action]
        state.sharp -= 1"""
    #for_opponent = [name[0], utils.guess_dmg]
    enough_focus = []
    prediction = ()
    prediction[0] = state
    best_state = aima.State(-math.inf,-math.inf,-math.inf,-math.inf,0)
    #if depth == 0:
    #    return (action, state)
    for action in utils.actions:
        if utils.actions[action] >= state.focus:
            enough_focus.append(action)
    combinations = list(utils.itertools.combinations_with_replacement(enough_focus, len(enough_focus)))
    for combination in combinations:
        required_focus = 0
        for action in combination:
            required_focus += utils.actions[action]
        if required_focus > state.focus:
            combinations.remove(combination)
        for action in combination:
            prediction = utils.to_executev2(prediction[0],action)
        if utils.leftbetterthan(prediction[0],best_state):
                #best_state.health = prediction[0].health
                #best_state.focus = prediction[0].focus
                #best_state.dmg_dealt = prediction[0].dmg_dealt
                best_state = prediction
                best_actions = combination
    return (best_state, best_actions)




def combatv2(file, handle, battle_end, still_alive, left_over, mutex):
    '''
    similar to combat but using best_move which calls minimax
    still_alive now send time sweep starting
    '''
    #global health, dmg_dealt, guess_incoming, dmg_got, name, forecast, all_dmg, focus, dmg_left, sharp
    #handle = click.input("BlueStacks 1 N-64")
    #handle = click.input("BlueStacks")
    focus = utils.start_focus
    dmg_left = utils.dmg_left[0] if len(utils.dmg_left) > 0 else 0
    prev = [1,0,0,0,0,0,[]]
    if not left_over.empty():
        temp = left_over.get()
        if temp != None:
            prev = temp
    rounds = prev[0]
    #utils.sharp[0] = prev[1]
    #utils.bat_swarm[0] = prev[2]
    #utils.arm_low[0] = prev[3]
    logger.info("left over" + str(prev))
    state = aima.State(utils.health[0],utils.focus[0],utils.dmg_dealt[0],prev[1],prev[2],prev[3], prev[4], prev[5], prev[6], rounds)
    state.focus_cost = aima.copy.deepcopy(utils.actions)
    reserve_state = aima.copy.deepcopy(state)
    forecast = [rounds, aima.copy.deepcopy(state)]
    #still_alive.send(time.time())
    dot = graphviz.Digraph('Minimax-'+str(time.time()), comment='Action graph')
    while True:
        #print("round", rounds, "done", readimg.battle_finished)

        execute(handle, "A")#return to action bar
        with mutex:
            handle.screenshot()
        time.sleep(1)
        old_health = state.health
        read_dmg = {}
        #state.round = rounds
        logger.info("Old state " + str(state.__dict__))
        if rounds%4:
            '''
            #reset the cost since it is modified
            utils.actions["A+Z"] = 0
            utils.actions["A+X"] = math.inf
            utils.actions["A+C"] = 20
            utils.actions["A+V"] = math.inf
            '''
            #save the returned value
            #sweep oponent_found or the get_dmg
            if rounds == 1:#first round
                old_health = state.health
                state.focus = focus

            #time.sleep(3)
            #while not len(all_dmg) >= 4:

            # Mark starting to sweep
            while not still_alive.empty():
                still_alive.get()
            still_alive.put([False, time.time()])
            try:
                with mutex:
                    read_dmg, state.health, focus, dmg_left, reason = parser.sweep(file,"Normal", state.focus_cost, utils.four_last[-4:])
                state.focus = focus if math.isclose(state.focus, focus, abs_tol=utils.focus_dodge) else state.focus
                utils.dmg_left.clear()
                utils.dmg_left.append(dmg_left)
                state.dmg.update(read_dmg)
                #utils.all_dmg.update(read_dmg)
                """if len(all_dmg) == 0:
                    left_over.put([rounds,sharp])
                    battle_end.clear()
                    return"""
            except Exception as e:
                #battle_end.clear()
                #battle_end.append(True)
                while not left_over.empty():
                    left_over.get()
                if str(e) == "Battle Ended":
                    left_over.put([utils.rounds[0], utils.sharp[0], utils.bat_swarm[0], utils.arm_low[0], utils.more_arm[0], utils.freeze[0], utils.to_swarm])
                    battle_end.clear()
                    return
                # Probably combo turn
                reserve_state = aima.copy.deepcopy(state)
                left_over.put([rounds+1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                battle_end.clear()
                #logger.info("sweep failed " + str(e))
                logger.error("Failed "+str(e),exc_info=1)
                #all_dmg, health, focus, dmg_left = readimg_copy.sweep(file,"Normal",actions, four_last[8:])
                '''
                if len(read_dmg) == 0:
                    try:
                        elem = left_over.get_nowait()
                    except Exception as e:
                        left_over.put([1,0])
                    battle_end.clear()
                    logger.info(f'Queue had value {elem}, putting new value 1, 0')
                    return
                '''
                return
            if reason != "Normal":
                rounds = utils.round_to_nearest_four(rounds)
                state.round = rounds
            if rounds == 1:
                if utils.guess_difficulty > utils.dmg_left[0]:
                    utils.true_difficulty[0] = 0
                if (utils.guess_difficulty + 4000) <= utils.dmg_left[0]:
                    utils.true_difficulty[0] = utils.dmg_left[0]
                
            # Done sweep, inform main
            while not still_alive.empty():
                still_alive.get()
            still_alive.put([True, time.time()])
            #state = aima.State(utils.health[0], utils.focus[0], utils.dmg_dealt[0], prev[1], prev[2], prev[3], prev[4], rounds)
            logger.info(str(state.__dict__))
            try:
                #forecast = best_move(state, True, aima.InstrumentedProblem(aima.AStarProblem(aima.copy.deepcopy(state))), -math.inf, math.inf, dot)
                forecast = aima.astar_search(aima.InstrumentedProblem(aima.AStarProblem(aima.copy.deepcopy(state))), h=None, display=False).path()[1]
            except Exception as e:
                logger.error("Failed "+str(e),exc_info=1)
                while not left_over.empty():
                    left_over.get()
                # Probably combo turn
                reserve_state = aima.copy.deepcopy(state)
                left_over.put([rounds+1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                battle_end.clear()
                #logger.info("sweep failed " + str(e))
                logger.error("Failed "+str(e),exc_info=1)
                return
            '''
            if rounds > 4:
                try:
                    utils.forecast = best_move(state.__dict__, True, aima.MinimaxProblem(state.__dict__), -math.inf, math.inf, dot)
                except Exception as e:
                    logger.error("Failed "+str(e),exc_info=1)
            else:
                utils.forecast = regular(state.health, state.focus, utils.dmg_dealt[0])
            '''
            #state  = [execute(handle, forecast[x], state) for x in range(len(forecast)-3)][-1] #take a list of action and execute
            logger.info("Executing "+str(forecast[0])+" "+str(forecast[1].action))
            #for action in forecast[1]:#doesnt work since action becomes char in actions
                #logger.info(str(action))
                #state = execute(handle, action, state)
            # it happens that aima give duplicate I
            #state.action = aima.copy.deepcopy(forecast[1].state.action)
            for action in aima.copy.deepcopy(forecast[1].action):
                state = execute(handle, action, state)
                #still_alive.send(time.time())
            logger.info("Executed "+str(forecast[1].action))
            utils.guess_incoming[0] = forecast[1].state
            time.sleep(7)
        else:#combo turn
            #time.sleep(3)
            # Mark starting to sweep
            while not still_alive.empty():
                still_alive.get()
            still_alive.put([False, time.time()])
            try:
                with mutex:
                    read_dmg, state.health, focus, dmg_left, reason = parser.sweep(file,"Combo", state.focus_cost, utils.four_last[-4:])
                state.focus = focus if math.isclose(state.focus, focus, abs_tol=utils.focus_dodge) else state.focus
                utils.dmg_left.clear()
                utils.dmg_left.append(dmg_left)
                utils.all_dmg.update(read_dmg)
                state.dmg.update(read_dmg)
                """if len(all_dmg) == 0:
                    left_over.put([rounds,sharp])
                    battle_end.clear()
                    return"""
            except Exception as e:
                while not left_over.empty():
                    left_over.get()
                if str(e) == "Battle Ended":
                    left_over.put([utils.rounds[0], utils.sharp[0], utils.bat_swarm[0], utils.arm_low[0], utils.more_arm[0], utils.freeze[0], utils.to_swarm])
                    battle_end.clear()
                    return
                # Probably normal turn
                reserve_state = aima.copy.deepcopy(state)
                if rounds > 1:
                    left_over.put([rounds-1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                else:
                    left_over.put([1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                battle_end.clear()
                #logger.info("sweep failed " + str(e))
                logger.error("Failed "+str(e),exc_info=1)
                #all_dmg, health, focus, dmg_left = readimg_copy.sweep(file,"Combo",actions, four_last[8:])
                '''
                if len(read_dmg) == 0:
                    try:
                        elem = left_over.get_nowait()
                    except Exception as e:
                        left_over.put([1,0])
                    battle_end.clear()
                    logger.info(f'Queue had value {elem}, putting new value 1, 0')
                    return
                '''
                return
            if reason != "Combo":
                rounds = utils.minus_until_empty(rounds, 2, 1)
                state.round = rounds
            # Done sweep, inform main
            while not still_alive.empty():
                still_alive.get()
            still_alive.put([True, time.time()])
            #state = aima.State(utils.health[0], utils.focus[0], utils.dmg_dealt[0], prev[1], prev[2], prev[3], prev[4], rounds)
            #state.combo_focus = aima.copy.deepcopy(utils.actions)
            #state.combo_dmg = aima.copy.deepcopy(read_dmg)
            #state.focus += focus_combo
            #forecast = get_combo(state.health, state.focus, dmg_dealt, 0) #add to save list of action
            #print(forecast)#debug
            #take a list of action and execute
            #save the final state to state
            logger.info(str(state.__dict__))
            try:
                #forecast = best_move(state, True, aima.InstrumentedProblem(aima.AStarProblem(aima.copy.deepcopy(state))), -math.inf,math.inf, dot)
                forecast = aima.astar_search(aima.InstrumentedProblem(aima.AStarProblem(aima.copy.deepcopy(state))), h=None, display=False).path()[1]
            except Exception as e:
                logger.error("Failed "+str(e),exc_info=1)
                while not left_over.empty():
                    left_over.get()
                # Probably normal turn
                reserve_state = aima.copy.deepcopy(state)
                if rounds > 1:
                    left_over.put([rounds-1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                else:
                    left_over.put([1, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, reserve_state.to_swarm])
                battle_end.clear()
                return
            '''
            if round > 4:
                try:
                    utils.forecast = best_move(state.__dict__, True, aima.MinimaxProblem(state.__dict__), -math.inf,math.inf, dot)
                except Exception as e:
                    logger.error("Failed "+str(e),exc_info=1)
            else:
                utils.forecast = get_combo(state.health, state.focus, utils.dmg_dealt[0], 0)
                '''
            logger.info("Executing Combo " + str(forecast[1].action) + " " + str(forecast[1].state.round))
            #forecast[1].state.action.append("I")
            #state  = [execute(handle, forecast[x], state) for x in range(len(forecast)-3)][-1] #take a list of action and execute
            #state.action = aima.copy.deepcopy(utils.remove_all_but_one(forecast[1].state.action, "I"))
            for action in aima.copy.deepcopy(utils.remove_all_but_one(forecast[1].action, "I")):
                state = execute(handle, action, state)
                #still_alive.send(time.time())
            logger.info("Executed Combo "+str(state.action))
            utils.guess_incoming[0] = forecast[1].state
            dot.unflatten(stagger=3)
            dot.render(directory = "minimax.gv")
            time.sleep(12)
        execute(handle, "A")#return to action bar
        #still_alive.send(time.time())
        rounds+=1
        state.round = rounds
        #state.bat_swarm = utils.minus_until_empty(state.bat_swarm, 1)
        #state.petrify = utils.minus_until_empty(state.petrify, 1)
        #state.fortify = utils.minus_until_empty(state.fortify, 1)

        if rounds%4:
            state.focus = utils.add_until_full(state.focus, utils.focus_increase, utils.cap_focus)
        else:
            state.focus = utils.add_until_full(state.focus, utils.focus_combo, utils.cap_focus)

        # Dequeue all items to empty the queue
        while not left_over.empty():
            left_over.get()
        reserve_state = aima.copy.deepcopy(state)

        left_over.put([rounds, reserve_state.sharp, reserve_state.bat_swarm, reserve_state.fracture, reserve_state.fortify, reserve_state.petrify, aima.copy.deepcopy(reserve_state.to_swarm)])
        #if sharp > 0:
        #    sharp -= 1
        #save a list of expected health, focus, dmg_left and name them as forecast
        utils.dmg_got[0] = old_health-state.health
        if not opponent.oponent_found:#also save first round where dmg_got = 0
            opponent.save(utils.name[0], old_health-state.health)#get name from sweeping somehow
        elif utils.dmg_got[0] > (old_health - utils.guess_incoming[0]["health"])*2 and opponent.oponent_found:
            opponent.save(utils.name[0], utils.dmg_got[0])
        #currently dont think ahead long so might not need this
        #if health != forecast or dmg_left > forecast or focus < forecast:
        #    replan
        #usually get black screen before this and is detected by sweep
        elif len(utils.dmg_left) > 0 and (utils.dmg_left[0] == 0 or utils.dmg_dealt == utils.dmg_left):
            #call init and reset all variables
            #break
            battle_end.clear()
            return
        #left_over.put([1,0])
        #print("Done")

#helper function, dont need in planner
def execute(handle, to_do, state = None):#handle for the input class from click
    #global sharp
    if state is not None:
        state = utils.to_executev4(state, to_do)
        logger.info("Execute " + to_do)
    if '+' in to_do:
        all = to_do.split('+')#test to make sure, maybe split will disrupt other
        handle.Keypress(all[0])
        handle.Keypress(all[1])
    else:
        handle.Keypress(to_do)#was thinking of forecast.pop but it has no action
    if to_do == "D+Z":
        time.sleep(15)
    time.sleep(1)
    return state

'''
def take_reward(handle):
    handle.Keycombination(click.SHIFT,'1')
'''


def combat(file, handle, battle_end, left_over):
    '''
    ...logic
    calls regular for non combo round and call get_combo for combo round
    '''
    global health, dmg_dealt, guess_incoming, dmg_got, name, forecast, all_dmg, focus, dmg_left, sharp
    #handle = click.input("BlueStacks 1 N-64")
    #handle = click.input("BlueStacks")
    temp = left_over.get()
    if temp == None:
        temp = [1,0]
    rounds = temp[0]
    state = aima.State(utils.health[0], utils.focus[0], utils.dmg_dealt[0], utils.sharp[0], utils.bat_swarm[0], rounds)
    utils.sharp[0] = temp[1]
    execute(handle, "A")#return to action bar
    while True:
        #print("round", rounds, "done", readimg.battle_finished)
        logger.info("round " + str(rounds) + " sharp" + " "+ str(utils.sharp[0]))
        logger.info("left over" + str(temp))
        #old_health = health
        old_health = state.health
        if rounds%4:
            #save the returned value
            #sweep oponent_found or the get_dmg
            if rounds == 1:#first round
                old_health = state.health
                utils.name[0] = parser.read_name(file)
            #time.sleep(3)
            #while not len(all_dmg) >= 4:
            try:
                utils.all_dmg, state.health, state.focus, holder = parser.sweep(file,"Normal", utils.actions, utils.four_last[8:])
                utils.dmg_left.append(holder)
                """if len(all_dmg) == 0:
                    left_over.put([rounds,sharp])
                    battle_end.clear()
                    return"""
            except Exception as e:
                #battle_end.clear()
                #battle_end.append(True)
                left_over.put([rounds, utils.sharp[0]])
                battle_end.clear()
                #logger.info("sweep failed " + str(e))
                logger.error("Failed "+str(e),exc_info=1)
                #all_dmg, health, focus, dmg_left = readimg_copy.sweep(file,"Normal",actions, four_last[8:])
                if len(utils.all_dmg) == 0:
                    left_over.put([rounds, utils.sharp[0]])
                    battle_end.clear()
                    return
                return
            state.focus += utils.focus_increase
            #minimax()#invoke later
            utils.forecast = regular(state.health, state.focus, utils.dmg_dealt[0])#expected health, focus, dmg_left
            state = execute(handle, utils.forecast[3], state) #take a list of action and execute
            utils.guess_incoming[0] = utils.forecast[0]
            time.sleep(5)
        else:#combo turn
            #time.sleep(3)
            try:
                utils.all_dmg, state.health, state.focus, holder= parser.sweep(file,"Combo", utils.actions, utils.four_last[8:])
                utils.dmg_left.append(holder)
                """if len(all_dmg) == 0:
                    left_over.put([rounds,sharp])
                    battle_end.clear()
                    return"""
            except Exception as e:
                left_over.put([rounds, utils.sharp[0]])
                battle_end.clear()
                #logger.info("sweep failed " + str(e))
                logger.error("Failed "+str(e),exc_info=1)
                #all_dmg, health, focus, dmg_left = readimg_copy.sweep(file,"Combo",actions, four_last[8:])
                if len(utils.all_dmg) == 0:
                    left_over.put([rounds, utils.sharp[0]])
                    battle_end.clear()
                    return
                return
            state.focus += utils.focus_combo
            utils.forecast = get_combo(state.health, state.focus, utils.dmg_dealt[0], 0) #add to save list of action
            #print(forecast)#debug
            #take a list of action and execute
            #save the final state to state
            state  = [execute(handle, utils.forecast[x], state) for x in range(len(utils.forecast)-3)][-1]
            utils.guess_incoming[0] = utils.forecast[-3]
            time.sleep(10)
        execute(handle, "A")#return to action bar
        rounds+=1
        #if sharp > 0:
        #    sharp -= 1
        #save a list of expected health, focus, dmg_left and name them as forecast
        utils.dmg_got[0] = old_health-state.health
        if not opponent.oponent_found:#also save first round where dmg_got = 0
            opponent.save(utils.name[0], old_health-state.health)#get name from sweeping somehow
        elif utils.dmg_got[0] > (old_health - utils.guess_incoming[0])*2 and opponent.oponent_found:
            opponent.save(utils.name[0], utils.dmg_got[0])
        #currently dont think ahead long so might not need this
        #if health != forecast or dmg_left > forecast or focus < forecast:
        #    replan
        #usually get black screen before this and is detected by sweep
        elif len(utils.dmg_left) > 0 and (utils.dmg_left[0] == 0 or utils.dmg_dealt == utils.dmg_left):
            #call init and reset all variables
            #break
            battle_end.clear()
            return
        #left_over.put([1,0])
        #print("Done")

if __name__ == '__main__':
    import click
    #replace round with length of operators
    #get_combo call itself recursively like get_combo, execute
    #sharp will have the index of action and only is sharpen if len-index=4
    #dont need a forecast queue
    #has health, focus, dmg_left as argument and recursive as func(health-dmg_got)

    #handle = click.input("BlueStacks")
    handle = click.input("BlueStacks 1 N-64")
    while True:
            handle.screenshot()
        #if not readimg.battle_finnished:
            print("round", utils.rounds[0])
            old_health = utils.health[0]
            if utils.rounds[0]%4:
                #save the returned value
                #sweep oponent_found or the get_dmg
                if utils.rounds[0] == 1:#first round
                    old_health = utils.health[0]
                    utils.name[0] = parser.read_name("test.png")
                #time.sleep(3)

                # Sweep and update game state
                utils.all_dmg, utils.health[0], utils.focus[0], holder = parser.sweep("test.png","Normal", utils.actions, utils.four_last[8:])
                utils.dmg_left[0].append(holder)
                utils.focus[0] += utils.focus_increase
                #minimax()#invoke later
                
                # Forecast next moves
                utils.forecast = regular(utils.health[0], utils.focus[0], utils.dmg_dealt[0])#expected health, focus, dmg_left

                # Execute forecasted actions
                execute(handle, utils.forecast[3]) #take a list of action and execute
                utils.guess_incoming[0] = utils.forecast[0]
                time.sleep(5)
            else:#combo turn
                #time.sleep(3)
                utils.all_dmg, utils.health[0], utils.focus[0], holder = parser.sweep("test.png", "Combo", utils.actions, utils.four_last[8:])
                utils.dmg_left[0].append(holder)
                utils.focus[0] += utils.focus_combo
                utils.forecast = get_combo(utils.health[0], utils.focus[0], utils.dmg_dealt[0], 0) #add to save list of action
                #print(forecast)#debug
                [execute(handle, utils.forecast[x]) for x in range(len(utils.forecast)-3)] #take a list of action and execute
                utils.guess_incoming[0] = utils.forecast[-3]
                time.sleep(10)

            execute(handle, "A")
            utils.rounds[0] += 1
            #if sharp > 0:
            #    sharp -= 1
            #save a list of expected health, focus, dmg_left and name them as forecast
            utils.dmg_got[0] = old_health - utils.health[0]
            if not opponent.oponent_found:#also save first round where dmg_got = 0
                opponent.save(utils.name[0], old_health-utils.health[0])#get name from sweeping somehow
            elif utils.dmg_got[0] > (old_health - utils.guess_incoming[0])*2 and opponent.oponent_found:
                opponent.save(utils.name[0], utils.dmg_got[0])
            #currently dont think ahead long so might not need this
            #if health != forecast or dmg_left > forecast or focus < forecast:
            #    replan
            #usually get black screen before this and is detected by sweep
            elif len(utils.dmg_left) > 0 and ( utils.dmg_left[0] == 0 or utils.dmg_dealt == utils.dmg_left):
                #call init and reset all variables
                break
        #else:
        #    take_reward()#take reward Alt+1