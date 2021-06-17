# this is a script for computing the accuracy of objects and governors of the preposition
# if an argument predicted by the dependency parser falls within the annotated range of the 
# gold annotations, then it is marked as correct
# this method computes a simple accuracy for objects, governors, and overall and prints these

import sys
import os
import json

gold_directory = "annotation-ta"
dependency_directory = "train-dep-json"

num_obj = 0
num_gov = 0
correct_gov = 0
correct_obj = 0

for filename in os.listdir(dependency_directory):

    filename_old = filename

    filename = filename.replace("-", "_")

    if filename not in os.listdir(gold_directory):
        continue

    gold_dict = {}
    dep_dict = {}

    with open(os.path.join(gold_directory, filename), "r") as f:
        gold_dict = json.load(f)

    with open(os.path.join(dependency_directory, filename_old), "r") as f:
        dep_dict = json.load(f)

    gold_views = gold_dict["views"]
    gold_constituents = []
    for x in gold_views:
        if x["viewData"][0]["viewName"] == "NER_CONLL":
            if "constituents" in x["viewData"][0].keys():
                gold_constituents = x["viewData"][0]["constituents"]

    dep_views = dep_dict["views"]
    dep_constituents = []
    for x in dep_views:
        if x["viewData"][0]["viewName"] == "PREPOSITION_SRL":
            if "constituents" in x["viewData"][0].keys():
                dep_constituents = x["viewData"][0]["constituents"]

    gold_gov = []
    gold_obj = []
    dep_gov = []
    dep_obj = []

    for cons in gold_constituents:
        if cons["label"] == "OBJ":
            gold_obj.append((cons["start"], cons["end"]))
        if cons["label"] == "GOV":
            gold_gov.append((cons["start"], cons["end"]))

    for cons in dep_constituents:
        if cons["label"] == "OBJ":
            dep_obj.append((cons["start"], cons["end"]))
        if cons["label"] == "GOV":
            dep_gov.append((cons["start"], cons["end"]))

    for (s1, e1) in dep_gov:
        num_gov += 1
        for (s2, e2) in gold_gov:
            if s1 >= s2 and e1 <= e2:
                correct_gov += 1

    for (s1, e1) in dep_obj:
        num_obj += 1
        for (s2, e2) in gold_obj:
            if s1 >= s2 and e1 <= e2: 
                correct_obj += 1

overall_accuracy = (correct_obj + correct_gov) / (num_gov + num_obj)
gov_accuracy = correct_gov / num_gov
obj_accuracy = correct_obj / num_obj

sys.stdout.write("Overall Accuracy: " + str(overall_accuracy) + "\n")
sys.stdout.write("Governor Accuracy: " + str(gov_accuracy) + "\n")
sys.stdout.write("Object Accuracy: " + str(obj_accuracy) + "\n")




    


    
