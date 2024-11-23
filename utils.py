import math
import functools
import itertools
import numpy as np
import random
from collections import defaultdict, Counter
import re
from multiprocessing import Lock

###q-gram
'''using for read_name: most of the char read are con, inst, wea so q = 2
'''
q = 2

mutex = Lock()

# Rules:
'''using for aima.actions: rules for combo turn
'''
max_repeats_per_set = 5  # Max repetitions of elements containing 'S' or 'D'
max_total_SD_combinations = 8  # Max different combinations of elements containing 'S' or 'D'

###manual change, static variables
#stats
guess_dmg = 300 #guessing opponent's dmg
guess_armor = 600 #guessing opponent's armor
guess_dodge = 0.4 #guessing opponent's dodge
guess_difficulty = 5000 #hp for normal difficulty
is_difficult = True
focus_increase = 26
start_focus = 95
cap_focus = 148
cap_health = 1585
crit_chance = 0.19
crit_bonus = 0.66
dodge = 0.68
shield_block = 0.34
focus_dodge = 12 # Focus increase when dodge
armor = 559 # Resistance value = (Armor points)/((Armor points)+250) #https://www.reddit.com/r/gamedesign/comments/10d6txi/actionrpg_armor_formula/
base = 250 #https://www.desmos.com/calculator/shfqvtxz5n

#skills
wp_increase = 1.25 # 5% more per points
focus_combo = 19 + focus_increase + 9#guessing, was 32
bite_dmg = 16 #11-15 dmg per lv, lifesteal 50-100% of dmg
bat_dmg = 826
backstab = 0.63 #maximum 5 strikes
fracture = 20 # When you use fracture, you can actually lower the enemies resistances into negative values, but there is a hard cap of -33
conjure = 508
fortify = 150
anti_dodge = 0.45

#build
one_hand = True #1 hand offense
poison = True #dagger has status

#set to infinite if ability not available
actions =   {"D+Z": 30, "D+X": 35, "D+C": 35, "D+V": 50,
            "S+Z": 20, "S+X": 45, "S+C": 12, "S+V": 35,
            "A+Z": 15, "A+C": 28, "A+X": math.inf, "A+V": 54}
four_last = list(actions.keys())
all_dmg = {"D+Z": 0, "D+X": 0, "D+C": 0, "D+V": 0,
            "S+Z": 0, "S+X": 0, "S+C": 0, "S+V": 0,
            "A+Z": 20, "A+C": 309, "A+X": 0, "A+V":647}

##########
#dynamic variables
main_picture = "main.png"
combat_picture = "main.png"
true_difficulty = [0] #initial hp of the mob
escape_chance = [0.0] #00% but should be read next time
handle = [None]
health = [0]
rounds = [1] #initial 0 then minimax will increment
dmg_dealt = [0]
sharp = [0] #initial 0
bat_swarm = [0] #initial 0
arm_low = [0] #initial 0
more_arm = [0] #initial 0
freeze = [0]
to_swarm = [] #intial empty
guess_incoming = [0]
dmg_got = [0]
name = [""]
focus = [0]
dmg_left = []
forecast = ()

def leftbetterthan(state, comparestate):
    if type(state) != None:
        if ((state.health>=comparestate.health and state.focus >= comparestate.focus and state.dmg_dealt >= comparestate.dmg_dealt)
            or (state.sharp > comparestate.sharp)
            or (state.health>=comparestate.health and state.dmg_dealt >= comparestate.dmg_dealt) or state.health>0):
            return True
    return False

def remove_all_but_one(arr, element):
    # Check if element is in the list
    if element in arr:
        # Find the first occurrence index of the element
        first_occurrence = arr.index(element)
        # Create a new list with one occurrence of the element and other elements
        return [x for i, x in enumerate(arr) if x != element or i == first_occurrence]
    return arr  # If element is not in the array, return original array

def add_until_full(total, increment, cap):
    if total + increment > cap:
        total += cap - total  # Add just enough to reach cap
    else:
        total += increment
    return total

def minus_until_empty(total, decrement, zero = 0, bottom = 0):
    if total - decrement < zero:
        total = bottom  # Add just enough to reach bottom
    else:
        total -= decrement
    return total

def round_to_nearest_four(x):
    return int(x / 4) * 4 if x > 4 else 4

def generate_list(target_sum):
    #https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
    return (np.random.dirichlet(np.ones(5),size=1)*target_sum).tolist()[0]

def qgrams(string, q):
    # Generate q-grams
    return [string[i:i+q] for i in range(len(string) - q + 1)]

def cosine_similarity(s1, s2, q):
    # Generate q-grams for both strings
    qgrams_s1 = qgrams(s1, q)
    qgrams_s2 = qgrams(s2, q)
    
    # Create frequency vectors using Counter
    freq_s1 = Counter(qgrams_s1)
    freq_s2 = Counter(qgrams_s2)
    
    # Get the set of all unique q-grams from both strings
    all_qgrams = set(freq_s1.keys()).union(set(freq_s2.keys()))
    
    # Create vectors for each string in the space of all unique q-grams
    vector_s1 = [freq_s1[qgram] for qgram in all_qgrams]
    vector_s2 = [freq_s2[qgram] for qgram in all_qgrams]
    
    # Calculate the dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector_s1, vector_s2))
    
    # Calculate the magnitudes of the vectors
    magnitude_s1 = math.sqrt(sum(v ** 2 for v in vector_s1))
    magnitude_s2 = math.sqrt(sum(v ** 2 for v in vector_s2))
    
    # Calculate cosine similarity
    if magnitude_s1 == 0 or magnitude_s2 == 0:
        return 0.0
    return dot_product / (magnitude_s1 * magnitude_s2)

def normalize(variation, pattern = r'[^\w\d\s/-]'):
    # Remove punctuation and convert to lowercase
    variation = re.sub(pattern, '', variation).lower()
    return variation

def get_most_accurate_name(names):

    all_words = []
    for name in names:
        all_words.append(normalize(name))

    # Initialize a dictionary to store the average similarity score for each name
    similarity_scores = 0
    
    # Compare each name with every other name
    for i, name in enumerate(all_words):
        total_similarity = 0
        # Compare with every other name
        for j, other_name in enumerate(all_words):
            if i != j:
                # Use SequenceMatcher to get a similarity ratio
                similarity = cosine_similarity(name, other_name, 2)
                total_similarity += similarity
        # Calculate the average similarity score
        similarity_scores += total_similarity / (len(all_words) - 1)
    
    return similarity_scores/ (len(all_words) - 1)

def generate_q_grams_with_positions(phrase, q, remove):
    # Generate q-grams with their starting positions for a given phrase
    if remove:
        phrase = normalize(phrase)  # Normalize spaces and lowercase
    q_grams_with_positions = [(phrase[i:i+q], i) for i in range(len(phrase) - q + 1)]
    return q_grams_with_positions

def extract_q_gram_frequencies_and_positions(phrases, q, remove = True):
    # Extract q-gram frequencies and their positions from the list of phrases
    q_gram_counter = Counter()
    q_gram_positions = defaultdict(list)
    
    for phrase in phrases:
        q_grams_with_positions = generate_q_grams_with_positions(phrase, q, remove)
        for q_gram, position in q_grams_with_positions:
            q_gram_counter[q_gram] += 1
            q_gram_positions[q_gram].append(position)
    
    return q_gram_counter, q_gram_positions

def reconstruct_phrase(q_gram_counter, q_gram_positions, q):
    # Reconstruct a representative phrase from q-grams using their positions
    sorted_q_grams = q_gram_counter.most_common()
    reconstructed_chars = [''] * (max(max(positions) + q for positions in q_gram_positions.values()) + 1)

    for q_gram, _ in sorted_q_grams:
        positions = q_gram_positions[q_gram]
        for pos in positions:
            if all(reconstructed_chars[pos + i] == '' for i in range(q)):
                reconstructed_chars[pos:pos + q] = q_gram

    # Join the characters to form the final reconstructed phrase
    reconstructed_phrase = ''.join(reconstructed_chars).strip()
    # Clean up and normalize spaces
    reconstructed_phrase = re.sub(r'\s+', ' ', reconstructed_phrase)
    
    return reconstructed_phrase

'''
A character may only select the same ability five times in one Combo Turn, and only up to a maximum of eight
abilities in total. Abilities are those in the Control and Instinct categories. Anything in the Weapons list
is only limited by the available focus pool.
'''

# Function for generating combinations of elements in a set
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations_with_replacement(s, r) for r in range(len(s)+1, 0, -1))

def limited_combinations(elements, max_combinations):
    """Generates all combinations of elements up to max_combinations."""
    return itertools.chain.from_iterable(
        itertools.combinations(elements, r) for r in range(min(len(elements), max_combinations) + 1)
    )

def calc_dmg(reason):
    array = list(all_dmg.values())[-4:]
    if "Normal" in reason:
        normal_dmg = array[-1]/3
        array = [[normal_dmg], [normal_dmg-array[0]], [0], [0]]
    return array

def count_SD_elements(subset):
    s_elements = 0
    d_elements = 0
    for elem in subset:
        if 'S' in elem:
            s_elements += 1
        if 'D' in elem:
            d_elements += 1

    return s_elements, d_elements

def fit_heuristic(subset):
    if len(subset) > 1:
        fit = True
        array = ''.join(subset)
        backstab_in = array.find('D+Z')
        batswarm_in = array.find('D+X')
        conjure_in = array.find('D+C')
        petrify_in = array.find('D+V')

        # backstab goes before petrify
        if backstab_in >= 0 and petrify_in >= 0:
            if backstab_in > petrify_in:
                fit = False

        # conjure goes before petrify, don't run if previous already set False
        if fit and (conjure_in >= 0 and petrify_in >= 0):
            if conjure_in > petrify_in and backstab_in < 0:
                fit = False
        
        if batswarm_in >= 0 and subset[-1] != 'D+X':
            fit = False

        return fit
    return True

def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn

def to_execute(action, dmg_dealt, focus):#add sharp
    global all_dmg, wp_increase, sharp, actions
    focus -= actions[action]
    if "A" in action and sharp[0] > 0:
        dmg_dealt += math.floor(all_dmg[action]*wp_increase)
        sharp[0] -= 1
    elif "A" in action:
        dmg_dealt += all_dmg[action]
    return (focus, dmg_dealt, action)

def to_executev2(state, action, round):#is change_state
    global all_dmg, wp_increase, sharp, actions
    if action != "":
        state.focus -= actions[action]
        if "A" in action and state.sharp > 0:
            state.dmg_dealt += math.floor(all_dmg[action]*wp_increase)
            state.sharp -= 1
        elif "A" in action:
            state.dmg_dealt += all_dmg[action]
    if round > len(state.old_focus):
        state.old_focus.append(state)
    else:
        state = state.old_focus[round]
    return (state, action)

def strike(state, action):
    fracture_multiplier = 1
    sharp_multiplier = 1
    if state["fracture"] > 0:
        fracture_multiplier += (fracture*state["fracture"])/100
    if state["sharp"] > 0:
        sharp_multiplier += wp_increase
        if "C" in action:
            state["sharp"] -= 2
        else:
            state["sharp"] -= 1
    state["dmg_dealt"] += math.floor(all_dmg[action]*sharp_multiplier*fracture_multiplier)

def to_executev3(state, action):
            global all_dmg, wp_increase, sharp, actions
            if action != "" and action != "I" and len(action) > 1:
                if "A" in action:
                    strike(state, action)
                    '''
                elif "I" in action:
                    if (not state["round"]%4) or len(state["action"]) < 2:
                        state["health"] -= opponent.get_dmg([name[0], guess_dmg])
                    return state
                    '''
                elif "S" in action:
                    if "Z" in action:
                        state["sharp"] += 4
                    elif "X" in action:
                        state['health'] += math.floor(bite_dmg*random.uniform(0.5,1))
                        state["dmg_dealt"] += all_dmg[action]
                    elif "C" in action: # and fracture**state['fracture'] < 33:
                        state['fracture'] += 1
                elif "D" in action:
                    if "Z" in action:
                        max_backstab = 0
                        strike(state, action)
                        max_backstab += 1
                        while random.uniform(0, 1) < backstab and max_backstab < 5:
                            strike(state, action)
                            max_backstab += 1
                    elif "X" in action:
                        state["bat_swarm"] += 5
                        state["to_swarm"].extend(generate_list(bat_dmg))
                    elif "C" in action:
                        state["dmg_dealt"] += conjure
                elif "F" in action:
                    '''
                    #https://stackoverflow.com/questions/3203099/percentage-chance-to-make-action
                    fail_chance = 1 - escape_chance[0]
                    if math.floor( random.uniform(0, 1/(1-fail_chance)) ):
                        #state["health"] -= opponent.get_dmg([name[0], guess_dmg])
                        state['ran'] = True
                        '''
                    state['ran'] = True
                    return state
                state["focus"] -= actions[action]
            return state

def strikev2(state, action):
    sharp_multiplier = 1
    base_dmg = 0

    fracture_multiplier = (1-(guess_armor - fracture*state.fracture)/(guess_armor - fracture*state.fracture + base))
    if state.sharp > 0:
        sharp_multiplier += wp_increase
        if "Z" in action:
            state.sharp -= 1
        else:
            state.sharp -= 2
    hit = guess_dodge
    if state.petrify > 0:
        hit -= anti_dodge
        state.petrify -= 1
    if state.round % 4:
        base_dmg = state.dmg[action]
    else:
        base_dmg = all_dmg[action]
    dmg = 0
    if random.uniform(0, 1) > hit:
        dmg = math.floor(base_dmg*sharp_multiplier*fracture_multiplier)
    state.dmg_dealt += dmg

def to_executev4(state, action):
            global all_dmg, wp_increase, sharp, actions
            if action != "" and action != "I" and len(action) > 1:
                if "A" in action:
                    strikev2(state, action)
                    '''
                elif "I" in action:
                    if (not state["round"]%4) or len(state["action"]) < 2:
                        state["health"] -= opponent.get_dmg([name[0], guess_dmg])
                    return state
                    '''
                elif "S" in action:
                    if "Z" in action:
                        state.sharp += 4
                    elif "X" in action:
                        state.health = add_until_full(state.health, math.floor(bite_dmg*random.uniform(0.5,1)), cap_health)
                        state.dmg_dealt += all_dmg[action]
                    elif "C" in action and state.fracture < 5:
                        state.fracture += 1
                    elif "V" in action:
                        state.fortify += 3
                elif "D" in action:
                    if "Z" in action:
                        max_backstab = 0
                        strikev2(state, "A+Z")
                        max_backstab += 1
                        while random.uniform(0, 1) < backstab and max_backstab < 5:
                            strikev2(state, "A+Z")
                            max_backstab += 1
                    elif "X" in action:
                        state.bat_swarm += 5
                        state.to_swarm.extend(generate_list(bat_dmg))
                    elif "C" in action:
                        state.dmg_dealt += conjure
                    elif "V" in action:
                        state.petrify += 3
                elif "F" in action:
                    #https://stackoverflow.com/questions/3203099/percentage-chance-to-make-action
                    fail_chance = 1 - escape_chance[0]
                    if math.floor( random.uniform(0, 1/(1-fail_chance)) ):
                        #state["health"] -= opponent.get_dmg([name[0], guess_dmg])
                        state.health -= state.health
                        state.ran = True
                        
                    #state.ran = True
                    return state
                state.focus = minus_until_empty(state.focus, state.focus_cost[action], bottom = state.focus)
            return state