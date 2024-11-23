import random
import json
import math
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler which logs even debug messages
fh = logging.FileHandler(__name__ + '.log', 'w')
#fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)

# Global flag to indicate if opponent is found
oponent_found = False

# Function to save damage after opponent attack (Alternative 1) might remove this
def find(name, damage):
    # Open and read the opponent.txt file
    file = open('opponent.txt', 'r')
    list_klass = json.load(file)#https://pynative.com/python-json-load-and-loads-to-parse-json/
    #debugging
    #print(list_klass)
    file.close()

    # Check if the opponent's name is in the list
    if name in list_klass.keys():
        element = 0
        bigger = 0
        all_saved = len(list_klass[name])

        # Compare the damage with saved damages
        for saved in list_klass[name]:
            if damage > saved:
                bigger += 1
            elif bigger == (all_saved - 1):  # If damage is bigger than all saved damages
                list_klass[name].append(damage)
                return False
            elif element == (all_saved - 1):
                return True
            element += 1
    else: # If name not in the list_klass, add the name with damage
        list_klass[name] = [damage]#https://careerkarma.com/blog/python-add-to-dictionary/#:~:text=There%20is%20no%20add()%20%2C%20append()%20%2C%20or%20insert(),assigning%20it%20a%20particular%20value.
    
    # Write the updated list back to opponent.txt
    with open('opponent.txt', 'w') as file:
        data = json.JSONEncoder().encode(list_klass)
        file.write(data)
        file.close()
    return False

# Function to save damage after opponent attack (Alternative 2)
def save(name, damage):
    # Open and read the opponent.txt file
    file = open('opponent.txt', 'r')
    list_klass = json.load(file)#https://pynative.com/python-json-load-and-loads-to-parse-json/
    #debugging
    #print(list_klass)
    file.close()

    # Check if the opponent's name is in the list
    if name in list_klass.keys():
        list_klass[name].append(damage)
    else:  # If name not in the list_klass, add the name with damage
        list_klass[name] = [damage]#https://careerkarma.com/blog/python-add-to-dictionary/#:~:text=There%20is%20no%20add()%20%2C%20append()%20%2C%20or%20insert(),assigning%20it%20a%20particular%20value.
    
    # Write the updated list back to opponent.txt
    with open('opponent.txt', 'w') as file:
        data = json.JSONEncoder().encode(list_klass)
        file.write(data)
        file.close()
    return False

# Function to get damage before opponent attack, with damage as fallback input if not found
def get_dmg(name_damage):
    global oponent_found
    multiplier = random.uniform(1, 2)#https://pynative.com/python-get-random-float-numbers/
    file = open('opponent.txt', 'r')
    list_klass = json.load(file)
    file.close()

    # Check if the opponent's name is in the list
    if name_damage[0] in list_klass.keys():
        oponent_found = True
        # Return the maximum saved damage multiplied by the random multiplier
        # return math.ceil(max(list_klass[name_damage[0]])*multiplier)
    oponent_found = False
    # Return the input damage multiplied by the random multiplier
    return math.ceil(name_damage[1] * multiplier)

if __name__ == '__main__':#debugging
    #https://docs.python.org/3/library/json.html
    #https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
    with open('opponent.txt', 'w') as file:            
        data = json.JSONEncoder().encode({"A":[22],"B":[1,2]})
        file.write(data)
        file.close()
    #print(find('B',2))
    #print(get_dmg('B',120))
    dmg_get, oponent_found = get_dmg('C',1)
    print(dmg_get, oponent_found)