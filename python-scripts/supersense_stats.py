import json
import sys
import os
import pprint

file = open("updated-train.txt", "r")

supersense1_dict = {}
supersense2_dict = {}
combo_dict = {}

for line in file.readlines():
    str_list = line.strip().split()
    separator_index = str_list.index("|||")

    supersense1 = str_list[separator_index + 1]
    if supersense1 not in supersense1_dict.keys():
        supersense1_dict[supersense1] = 1
    else:
        supersense1_dict[supersense1] = supersense1_dict[supersense1] + 1
    
    supersense2 = str_list[separator_index + 2]
    if supersense2 not in supersense2_dict.keys():
        supersense2_dict[supersense2] = 1
    else:
        supersense2_dict[supersense2] = supersense2_dict[supersense2] + 1

    combo_key = supersense1 + ", " + supersense2
    if combo_key not in combo_dict.keys():
        combo_dict[combo_key] = 1
    else:
        combo_dict[combo_key] = combo_dict[combo_key] + 1


pp = pprint.PrettyPrinter(indent=4)


sys.stdout.write("Supersense1 Dictionary: \n")
pp.pprint(supersense1_dict)

sys.stdout.write("\n")
sys.stdout.write("Supersense2 Dictionary: \n")
pp.pprint(supersense2_dict)

sys.stdout.write("\n")
sys.stdout.write("Combo Dictionary: \n")
pp.pprint(combo_dict)