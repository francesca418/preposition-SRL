# method that prints out the stats of a dataset for preposition srl

import sys
import os
import pprint

def get_stats():

    instances = open("data/test-dep.txt", "r")
    instances = [x for x in instances]

    prep_dict = {}
    count = 0

    for instance in instances:
        instance = instance.split(" ")
        separator_index = instance.index("|||")
        sentence = instance[1:(separator_index)]
        tags = instance[(separator_index + 3):-1]
        prep = ""
        #sys.stdout.write(str(len(sentence)))
        #sys.stdout.write(str(len(tags)))
        #sys.stdout.write(str(sentence) + "\n")
        #sys.stdout.write(str(tags) + "\n")
        assert len(sentence) == len(tags)
        for i, tag in enumerate(tags):
            if "PREP" in tag:
                prep = prep + sentence[i] + " "
        prep = prep.lower()
        if prep in prep_dict.keys():
            prep_dict[prep] += 1
        else:
            prep_dict[prep] = 1
        count += 1 

    sys.stdout.write("Counted " + str(count) + " instances.\n")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(prep_dict)

get_stats()