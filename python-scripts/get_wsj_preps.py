import sys
import os
import json

fileName = sys.argv[1]
outFile = sys.argv[2]
instances = []

with open(fileName, "r") as f:
    instances = f.readlines()

preps = {}
count = 0
weirdos = {}
for line in instances:
    str_list = line.split()
    separator_index = str_list.index("|||")
    sentence_list = str_list[1:separator_index]
    tags = str_list[(separator_index+3):]
    prep_indices = []
    if "B-PREP" not in tags:
        continue
    start = tags.index("B-PREP")

    if (len(sentence_list) > len(tags)):
        sys.stdout.write(str_list[0] + "\n")
        tags.append("O")
    if (len(sentence_list) + 1 == len(tags)):
        del tags[0]
    
    if (len(sentence_list) < len(tags)):
        sys.stdout.write(str(str_list[0]) + "\n")
        count = count + 1
        if str_list[0] not in weirdos:
            weirdos[str_list[0]] = 1
        else:
            weirdos[str_list[0]] = weirdos[str_list[0]] + 1
        for i in range(len(tags)):
            tag = tags[i]
            word = ""
            if i >= len(sentence_list):
                word = "@"
            else:
                word = sentence_list[i]
            sys.stdout.write("( " + word + " " + tag + " )\n")
        sys.stdout.write("\n")
        continue
    assert(len(sentence_list) == len(tags))
    while True:
        if start >= len(tags) or "PREP" not in tags[start]:
            break
        prep_indices.append(start)
        start = start + 1

    preposition = ""
    for index in prep_indices:
        preposition = preposition + sentence_list[index] + " "

    preposition = preposition.lower()
    if preposition not in preps:
        preps[preposition] = 1
    else:
        preps[preposition] = preps[preposition] + 1
sys.stdout.write( str(count) + "wonky ones\n")
sys.stdout.write(str(len(weirdos)))
with open(outFile, "w+") as f:
    json.dump(preps, f,  ensure_ascii=True, indent=4)
