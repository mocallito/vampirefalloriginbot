# Modified aima
import copy
import opponent
import itertools
import utils
import math
import random

class State():
    def __init__(self,health,focus,dmg_dealt,sharp, bat_swarm, fracture, fortify, petrify, to_swarm, round):
        self.health = health
        self.focus = focus
        self.dmg_dealt = dmg_dealt
        self.sharp = sharp
        self.round = round
        self.bat_swarm = bat_swarm
        self.fracture = fracture
        self.fortify = fortify
        self.petrify = petrify
        self.ran = False
        self.is_maximizing = True
        self.to_swarm = to_swarm
        self.combo_dmg = {}
        self.focus_cost = {}
        self.dmg = {}
        self.action = []
        #self.old_focus = []
        #self.old_focus.append(focus)

    def content(self):
        return (self.health, self.focus, self.dmg_dealt, self.sharp)

    def __lt__(self, other):
        ''' State is considered bigger if it is later or
        if it has more stat
        '''
        lt = False
        lt = lt or self.dmg_dealt < other.dmg_dealt
        lt = lt or self.round < other.round
        lt = lt or self.compare_state(other, "petrify")
        lt = lt or self.compare_state(other, "sharp")
        lt = lt or self.compare_state(other, "fracture")
        lt = lt or self.compare_state(other, "fortify")
        lt = lt or self.compare_state(other, "bat_swarm")
        return lt
    
    def compare_state(self, other, stat):
        ''' If stat exist then find whether it is counting down or
        if it is initializing
        '''
        current_stat = getattr(self, stat)
        previous_stat = getattr(other, stat)
        diff = previous_stat - current_stat
        return diff < 0 or diff > 0

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        #return [self.child_node(problem, action)                for action in problem.actions(self.state)]
        for action in problem.actions(self.state):
            yield self.child_node(problem, action)
                #like possible actions, for nim, the player can take all in one pile

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)#next state, probably taken from possible new states
        #logger.info(str(next_state))
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""


    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal


    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError


    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal


    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1


    def value(self, state):#does not raise error if is not used before overriding
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __repr__(self):
        return '<{:4d}/{:4d}/{:4d}/{}>'.format(self.succs, self.goal_tests,
                                               self.states, str(self.found)[:12])


class MinimaxProblem(Problem):
    def __init__(self, initial):
        super().__init__(initial)
    def goal_test(self, node):
        #if state is terminal
        
        premature_end = False

        steps_ahead = 0
        if utils.guess_difficulty < utils.true_difficulty[0] or utils.is_difficult:
            steps_ahead = 2

        if node.state.round - node.path()[0].state.round > steps_ahead: # increase this as the code works better
            premature_end = True

        if (len(utils.dmg_left)>0 and utils.dmg_left[0] <= node.state.dmg_dealt) or round(node.state.health) <= 0:
            premature_end = True
        
        return premature_end
            
    def evaluate(self, node):
        if node.state.ran == True:
            if node.state.is_maximizing:
                return -math.inf#dont want this solution, -1 means never choose unless it is the only way
            return math.inf
        list_dmg = [node.state.dmg_dealt for node in node.path()]#add all parent node dmg_dealt score and return
        total_dmg = sum(list_dmg)

        list_cost = []
        for node in node.path():
            if node.action is not None:
                for action in node.action:
                    # found many unknown keys from nodeaction including None
                    if action in node.state.focus_cost:
                        list_cost.append(node.state.focus_cost[action])
        total_cost = sum(list_cost)

        # Average dmg per turn
        #return total_dmg/len(list_dmg)
        #print(node.state.health)
        '''
        # Ratio or efficiency, between total_dmg and dmg lost until now
        health_lost = (node.path()[0].state.health-node.state.health)
        if node.state.is_maximizing:
            if health_lost == 0 or node.state.health >= node.parent.state.health:
                if node.action[0] == "I":
                    return 0
                return total_dmg
            return total_dmg/health_lost
        else:
            health_dif = node.parent.state.health-node.state.health
            return -(health_dif) if health_dif > 0 else health_dif
        '''
        # Ratio or efficiency, between total_dmg and focus lost until now
        health_lost = (node.path()[0].state.health-node.state.health)
        if node.state.is_maximizing:
            if node.action[0] == "I":
                return 0
            efficiency = 0
            if total_cost > 0:
                efficiency = total_dmg/total_cost * max(0, health_lost)/utils.cap_health
            else:
                efficiency = 1
            return efficiency
        else:
            return max(0, health_lost)/utils.cap_health

    def action_combo(self, state, combo_actions, post_poned):
        actions = []
        post_poned_actions = []
        s_set = [x for x in combo_actions if "S" in x]
        d_set = [x for x in combo_actions if "D" in x]
        a_set = [x for x in combo_actions if "A" in x]

        list_to_str = ''.join(d_set)
        if list_to_str.find('D+X') > -1:
            # Batswarm should be executed last
            a_set.append(d_set.pop(d_set.index('D+X')))
        if list_to_str.find('D+V') > -1:
            # Petrify should be executed last
            a_set.append(d_set.pop(d_set.index('D+V')))
        
        for s_subset in utils.powerset(s_set):#https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
            #if not utils.fit_heuristic(s_subset):
            #    continue
            for d_subset in utils.powerset(d_set):
    
                # Apply the rules
                if (len(s_subset) + len(d_subset)) > utils.max_total_SD_combinations:
                    continue  # Skip subsets with more than 8 different combinations of 'S' or 'D'
                if len(s_subset) > utils.max_repeats_per_set or len(d_subset) > utils.max_repeats_per_set:
                    continue  # Skip subsets with more than 5 occurrences of elements containing 'S' or 'D'
                
                # Apply heuristic
                if not utils.fit_heuristic(d_subset):
                    continue

                for a_subset in utils.powerset(a_set):

                    # Apply heuristic
                    if not utils.fit_heuristic(a_subset):
                        continue

                    combined_subset = s_subset + d_subset + a_subset

                    required_focus = 0
                    for action in combined_subset:
                        required_focus += state.focus_cost[action]
                    if required_focus <= state.focus:
                        full_set = list(combined_subset)
                        full_set.append("I")
                        too_much_s = 0
                        for action in set(s_subset):
                            dublicate = s_subset.count(action)
                            if dublicate > 2:
                                too_much_s = dublicate
                                break
                        too_much_d = d_subset.count("D+V")
                        if (too_much_s or (too_much_d>2)) or any(combined_subset.count(elem) > 1 for elem in post_poned):
                            post_poned_actions.append(full_set)
                        else:
                            yield full_set
                            #return#stops the yield bellow from executing, doesnt seems to return value
                        #https://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
                    else:
                        #state.action.append("I")
                        #yield state.action
                        continue
        # Post poned first, the actions() will reverse it later
        #post_poned_actions.extend(actions)
        for move in post_poned_actions:
            yield move
        #return post_poned_actions
                
    def actions(self, state): #https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide
        """what to do, different implementation has different actions
        like possible actions, for nim, the player can take all in one pile
        """
        #state = copy.deepcopy(state)
        actions = []
        #state["action"] = ["I"]
        
        if (state.health <= opponent.get_dmg([utils.name[0], utils.guess_dmg]) and (not utils.is_difficult 
            or utils.guess_difficulty < utils.true_difficulty[0])):#decide to quit if health <= opponent_dmg
            #state.action = ["F"]
            yield ["F"]

        post_poned = []
        if state.sharp > 0:
            post_poned.append("S+Z")
        if state.bat_swarm > 0:
            post_poned.append("D+X")
        if state.fortify > 0:
            post_poned.append("S+V")
        if state.fracture > 0:
            post_poned.append("S+C")
        if state.petrify > 0:
            post_poned.append("D+V")

        immediate_combinations = []
        postponed_combinations = []
        
        #normal turn
        for key in state.focus_cost.keys():
            if state.focus >= state.focus_cost[key]:
                actions.append([key])
                if key in post_poned:
                    postponed_combinations.append(key)
                else:
                    immediate_combinations.append(key)
                    #return
            #state.action.append("I")
            #yield state.action
        #combo turn, might return a whole set of action
        '''
        if not state.round%4:
            #https://careerkarma.com/blog/python-remove-key-from-a-dictionary/#:~:text=To%20remove%20a%20key%20from,item%20after%20the%20del%20keyword.
            #del combo_actions["I"]
            #del combo_actions["F"]

            immediate_combinations = self.action_combo(state, immediate_combinations)
            postponed_combinations = self.action_combo(state, postponed_combinations)
        '''
        for subset in (
            self.action_combo(state, state.focus_cost.keys(), postponed_combinations) if (not state.round%4) else list(reversed(immediate_combinations))):
            yield subset if type(subset) == type([]) else [subset]

        for subset in (list(reversed(postponed_combinations))):
            yield subset if type(subset) == type([]) else [subset]

        if len(actions) < 1:
            #state.action.append("I")
            yield ["I"]

        #logger.info("done")

    '''
    # result for astar
    def result(self, state, action):
        state = copy.deepcopy(state)
        state["action"] = action
        for move in action:
            state = utils.to_executev3(state, move)
        if state['ran'] == True:
            fail_chance = 1 - utils.escape_chance[0]
            if math.floor( random.uniform(0, 1/(1-fail_chance)) ):
                state["health"] -= opponent.get_dmg([utils.name[0], utils.guess_dmg])
        else:
            state["health"] -= opponent.get_dmg([utils.name[0], utils.guess_dmg])
        state["round"] += 1
        if state["round"]%4:
            state["focus"] += utils.focus_increase
        else:
            state["focus"] += utils.focus_combo
        return state
'''
    
    # result for minimax
    def result(self, state, action):
        #consequence of the action like the toexecute_v2/possible new state
        #next state, probably taken from possible new states
        #could make nested function to call toexecute for each move in action
        #add state.health -= opponent.get_dmg(for_opponent)
        #for_opponent = [name, guess_dmg]#both these could also be global
        global focus_increase, focus_combo
        #state["health"] -= opponent.get_dmg(for_opponent)

        state = copy.deepcopy(state)#without deep.copy the state pointer remains the same
        #state = copy.deepcopy(self.initial)
        more_armor = utils.armor
        dodged = random.uniform(0, 1) > utils.dodge
        #state.action = action
        if not action == "I":
            if state.is_maximizing:
                for move in action:
                    state = utils.to_executev4(state, move)
            '''
            if state.ran == True:
                #https://stackoverflow.com/questions/3203099/percentage-chance-to-make-action
                fail_chance = 1 - utils.escape_chance[0]
                if math.floor( random.uniform(0, 1/(1-fail_chance)) ) and dodged and random.uniform(0, 1) > utils.shield_block:
                    state.health -= opponent.get_dmg([utils.name[0], utils.guess_dmg])*(1-more_armor/(more_armor+utils.base))
            '''
        if state.is_maximizing:
            state.round += 1
            if state.bat_swarm > 0:
                state.dmg_dealt += state.to_swarm.pop()
                state.bat_swarm -= 1
            if state.fortify > 0:
                more_armor = utils.armor + utils.fortify
            if state.petrify > 0:
                state.petrify -= 1

        if (action == "I" or not state.is_maximizing) and dodged and random.uniform(0, 1) > utils.shield_block:
            state.health -= opponent.get_dmg([utils.name[0], utils.guess_dmg])*(1-more_armor/(more_armor+utils.base))
            #state.is_maximizing = True # not optimal to put this here
        if state.is_maximizing:
            if dodged:
                state.focus = utils.add_until_full(state.focus, utils.focus_dodge, utils.cap_focus)
            if state.round%4:
                state.focus = utils.add_until_full(state.focus, utils.focus_increase, utils.cap_focus)
            else:
                state.focus = utils.add_until_full(state.focus, utils.focus_combo, utils.cap_focus)
        return state
        

    def path_cost(self, c, state1, action, state2):
        #return abs(state1.health - state2.health) + abs(state2.dmg_dealth - state2.dmg_left)
        pass
    
    def h(self, state):
        """Estimate the cost from the current state to the goal."""
        # Simple heuristic: difference in HP between player and machine
        #return abs(state.health - state.opponent_health) + abs(state.dmg_dealth - state.dmg_taken)
        return 0

    def possible_states(self):
        state = copy.deepcopy(self.initial)
        for action in self.actions(state):
            #state = copy.deepcopy(self.initial)#no idea but testing
            yield self.result(state, action)

class AStarProblem(Problem):
    def __init__(self, initial):
        super().__init__(initial)
    def goal_test(self, state):
        #if state is terminal
        
        premature_end = False

        if (len(utils.dmg_left)>0 and utils.dmg_left[0] <= state.dmg_dealt) or round(state.health) <= 0:
            premature_end = True
        
        return premature_end
            
    def evaluate(self, node):
        if node.state.ran == True:
            return -math.inf#dont want this solution, -1 means never choose unless it is the only way
        list_dmg = [node.state.dmg_dealt for node in node.path()]#add all parent node dmg_dealt score and return
        total_dmg = sum(list_dmg)

        list_cost = []
        for node in node.path():
            if node.action is not None:
                for action in node.action:
                    # found many unknown keys from nodeaction including None
                    if action in node.state.focus_cost:
                        list_cost.append(node.state.focus_cost[action])
        total_cost = sum(list_cost)

        # Ratio or efficiency, between total_dmg and focus lost until now
        health_lost = (node.path()[0].state.health-node.state.health)
        if node.action[0] == "I":
            return 0
        efficiency = 0
        if total_cost > 0:
            efficiency = total_dmg/total_cost * health_lost/utils.cap_health
        else:
            efficiency = 1
        return efficiency

    def action_combo(self, state, combo_actions, post_poned):
        actions = []
        post_poned_actions = []
        s_set = [x for x in combo_actions if "S" in x]
        d_set = [x for x in combo_actions if "D" in x]
        a_set = [x for x in combo_actions if "A" in x]

        '''
        list_to_str = ''.join(d_set)
        if list_to_str.find('D+X') > -1:
            # Batswarm should be executed last
            a_set.append(d_set.pop(d_set.index('D+X')))
        if list_to_str.find('D+V') > -1:
            # Petrify should be executed last
            a_set.append(d_set.pop(d_set.index('D+V')))
        '''

        for s_subset in utils.powerset(s_set):#https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
            #if not utils.fit_heuristic(s_subset):
            #    continue
            for d_subset in utils.powerset(d_set):
    
                # Apply the rules
                if (len(s_subset) + len(d_subset)) > utils.max_total_SD_combinations:
                    continue  # Skip subsets with more than 8 different combinations of 'S' or 'D'
                if len(s_subset) > utils.max_repeats_per_set or len(d_subset) > utils.max_repeats_per_set:
                    continue  # Skip subsets with more than 5 occurrences of elements containing 'S' or 'D'
                '''
                # Apply heuristic
                if not utils.fit_heuristic(d_subset):
                    continue
                '''

                for a_subset in utils.powerset(a_set):

                    '''
                    # Apply heuristic
                    if not utils.fit_heuristic(a_subset):
                        continue
                    '''

                    combined_subset = s_subset + d_subset + a_subset

                    required_focus = 0
                    for action in combined_subset:
                        required_focus += state.focus_cost[action]
                    if required_focus <= state.focus:
                        full_set = list(combined_subset)
                        full_set.append("I")
                        '''
                        too_much_s = 0
                        for action in set(s_subset):
                            dublicate = s_subset.count(action)
                            if dublicate > 2:
                                too_much_s = dublicate
                                break
                        too_much_d = d_subset.count("D+V")
                        if (too_much_s or (too_much_d>2)) or any(combined_subset.count(elem) > 1 for elem in post_poned):
                            post_poned_actions.append(full_set)
                        else:
                            yield full_set
                            #return#stops the yield bellow from executing, doesnt seems to return value
                        '''
                        yield full_set
                        #https://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
                    else:
                        #state.action.append("I")
                        #yield state.action
                        continue
        # Post poned first, the actions() will reverse it later
        #post_poned_actions.extend(actions)
        #for move in post_poned_actions:
            #yield move
        #return post_poned_actions
                
    def actions(self, state): #https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide
        """what to do, different implementation has different actions
        like possible actions, for nim, the player can take all in one pile
        """
        #state = copy.deepcopy(state)
        actions = []
        #state["action"] = ["I"]
        
        if (state.health <= opponent.get_dmg([utils.name[0], utils.guess_dmg]) and (not utils.is_difficult 
            or utils.guess_difficulty < utils.true_difficulty[0])):#decide to quit if health <= opponent_dmg
            #state.action = ["F"]
            yield ["F"]

        post_poned = []
        '''
        if state.sharp > 0:
            post_poned.append("S+Z")
        if state.bat_swarm > 0:
            post_poned.append("D+X")
        if state.fortify > 0:
            post_poned.append("S+V")
        if state.fracture > 0:
            post_poned.append("S+C")
        if state.petrify > 0:
            post_poned.append("D+V")
        '''

        immediate_combinations = []
        postponed_combinations = []
        
        #normal turn
        for key in state.focus_cost.keys():
            if state.focus >= state.focus_cost[key]:
                actions.append([key])
                if key in post_poned:
                    postponed_combinations.append(key)
                else:
                    immediate_combinations.append(key)
                    #return
            #state.action.append("I")
            #yield state.action
        #combo turn, might return a whole set of action
        '''
        if not state.round%4:
            #https://careerkarma.com/blog/python-remove-key-from-a-dictionary/#:~:text=To%20remove%20a%20key%20from,item%20after%20the%20del%20keyword.
            #del combo_actions["I"]
            #del combo_actions["F"]

            immediate_combinations = self.action_combo(state, immediate_combinations)
            postponed_combinations = self.action_combo(state, postponed_combinations)
        '''
        for subset in (
            self.action_combo(state, state.focus_cost.keys(), postponed_combinations) if (not state.round%4) else immediate_combinations):
            yield subset if type(subset) == type([]) else [subset]

        #for subset in postponed_combinations:
            #yield subset if type(subset) == type([]) else [subset]

        if len(actions) < 1:
            #state.action.append("I")
            yield ["I"]

        #logger.info("done")

    '''
    # result for astar
    def result(self, state, action):
        state = copy.deepcopy(state)
        state["action"] = action
        for move in action:
            state = utils.to_executev3(state, move)
        if state['ran'] == True:
            fail_chance = 1 - utils.escape_chance[0]
            if math.floor( random.uniform(0, 1/(1-fail_chance)) ):
                state["health"] -= opponent.get_dmg([utils.name[0], utils.guess_dmg])
        else:
            state["health"] -= opponent.get_dmg([utils.name[0], utils.guess_dmg])
        state["round"] += 1
        if state["round"]%4:
            state["focus"] += utils.focus_increase
        else:
            state["focus"] += utils.focus_combo
        return state
'''
    
    # result for minimax
    def result(self, state, action):
        #consequence of the action like the toexecute_v2/possible new state
        #next state, probably taken from possible new states
        #could make nested function to call toexecute for each move in action
        #add state.health -= opponent.get_dmg(for_opponent)
        #for_opponent = [name, guess_dmg]#both these could also be global
        global focus_increase, focus_combo
        #state["health"] -= opponent.get_dmg(for_opponent)

        state = copy.deepcopy(state)#without deep.copy the state pointer remains the same
        #state = copy.deepcopy(self.initial)
        more_armor = utils.armor
        dodged = random.uniform(0, 1) > utils.dodge
        state.action = action
        if not action == "I":

            for move in action:
                state = utils.to_executev4(state, move)
            '''
            if state.ran == True:
                #https://stackoverflow.com/questions/3203099/percentage-chance-to-make-action
                fail_chance = 1 - utils.escape_chance[0]
                if math.floor( random.uniform(0, 1/(1-fail_chance)) ) and dodged and random.uniform(0, 1) > utils.shield_block:
                    state.health -= opponent.get_dmg([utils.name[0], utils.guess_dmg])*(1-more_armor/(more_armor+utils.base))
            '''

        state.round += 1
        if state.bat_swarm > 0:
            state.dmg_dealt += state.to_swarm.pop()
            state.bat_swarm -= 1
        if state.fortify > 0:
            more_armor = utils.armor + utils.fortify
        if state.petrify > 0:
            state.petrify -= 1

        if dodged and random.uniform(0, 1) > utils.shield_block:
            state.health -= opponent.get_dmg([utils.name[0], utils.guess_dmg])*(1-more_armor/(more_armor+utils.base))
            #state.is_maximizing = True # not optimal to put this here

        if dodged:
            state.focus = utils.add_until_full(state.focus, utils.focus_dodge, utils.cap_focus)
        if state.round%4:
            state.focus = utils.add_until_full(state.focus, utils.focus_increase, utils.cap_focus)
        else:
            state.focus = utils.add_until_full(state.focus, utils.focus_combo, utils.cap_focus)
        return state
        

    def path_cost(self, c, state1, action, state2):
        if state1.ran == True:
            return -math.inf#dont want this solution, -1 means never choose unless it is the only way
        total_dmg = state1.dmg_dealt + state2.dmg_dealt

        list_cost = []
        for action in state2.action:
            # found many unknown keys from nodeaction including None
            if action in state2.focus_cost:
                list_cost.append(state2.focus_cost[action])
        for action in state1.action:
            # found many unknown keys from nodeaction including None
            if action in state1.focus_cost:
                list_cost.append(state1.focus_cost[action])
        total_cost = sum(list_cost)

        # Ratio or efficiency, between total_dmg and focus lost until now
        health_lost = (state1.health-state2.health)
        if len(state1.action) > 1 and state1.action[0] == "I":
            return 0
        efficiency = 0
        if total_cost > 0:
            efficiency = total_dmg/total_cost * health_lost/utils.cap_health
        else:
            efficiency = 1
        return efficiency
    
    def h(self, node):
        """Estimate the cost from the current state to the goal.
        Currently 2 choices"""
        # Simple heuristic: difference in HP between player and machine
        #return abs(state.health - state.opponent_health) + abs(state.dmg_dealth - state.dmg_taken)

        # Prioritize states where the player has higher HP
        return -node.state.health/utils.cap_health

class graph_node:
    def __init__(self, initial, state):
        self.id = initial
        self.state = state

import heapq
# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue
# Stack and FIFOQueue are implemented as list and collection.deque
# PriorityQueue is implemented here


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = utils.memoize(h or problem.h, 'h')
    print("astar_search")
    f = utils.memoize(lambda n: n.path_cost + h(n), 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('max', f)
    frontier.append(node)
    explored = set()
    count=0
    while frontier:
        node = frontier.pop()
        print(node.__dict__,count)
        count+=1
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            print(node.path())
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None