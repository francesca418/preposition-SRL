'''
This script converts Test Annotation JSON formatted preposition SRL data to conll formatted SRL data

data will be in a single file in the format:
filename sentence ||| supersense1 supersense2 sentence_tags
'''
import json
import sys
import os
#import pprint

with open("manually-annotated-wsj.txt", "w") as f:
        f.write("")

for filename in os.listdir("wsj-sentences"):

    sentence_dict = {}

    with open(os.path.join("wsj-sentences", filename), "r") as f:
        sentence_dict = json.load(f)

    predictions_dict = {}
    with open(os.path.join("ma-wsj-predictions", filename), "r") as f:
        predictions_dict = json.load(f)

    output_line = ""
    output_line = filename + " " + sentence_dict["sentence"] + " " + "|||" + " " + "supersense1 " + "supersense2 "

    prepositions = predictions_dict["prepositions"]

    old_output_line = output_line

    for i in range(len(prepositions)):
        output_line = old_output_line
        tags = predictions_dict["prepositions"][i]["tags"]
        for tag in tags:
            output_line = output_line + tag + " "
        with open("manually-annotated-wsj.txt", "a") as f:
            f.write(output_line + "\n")
