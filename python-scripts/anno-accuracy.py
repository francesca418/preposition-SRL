import sys
import os
import json

gold_directory = "text-annotation-streusle/annotated-ta"
dependency_directory = "text-annotation-streusle/train-dep-json"
other_directory = "text-annotation-streusle/inter-annotator-agreement/streusle-annotation-nick"

num_obj = 0
num_gov = 0
correct_gov = 0
correct_obj = 0

for filename in os.listdir(dependency_directory):

    filename_old = filename

    filename = filename.replace("-", "_")

    if filename not in os.listdir(other_directory):
        continue

    gold_dict = {}
    dep_dict = {}
    other_dict = {}

    with open(os.path.join(gold_directory, filename), "r") as f:
        gold_dict = json.load(f)

    with open(os.path.join(dependency_directory, filename_old), "r") as f:
        dep_dict = json.load(f)

    with open(os.path.join(other_directory, filename), "r") as f:
        other_dict = json.load(f)

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

    other_views = other_dict["views"]
    other_constituents = []
    for x in other_views:
        if x["viewData"][0]["viewName"] == "NER_CONLL":
            if "constituents" in x["viewData"][0].keys():
                other_constituents = x["viewData"][0]["constituents"]

    gold_gov = []
    gold_obj = []
    dep_gov = []
    dep_obj = []
    other_gov = []
    other_obj = []

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

    for cons in other_constituents:
        if cons["label"] == "OBJ":
            other_obj.append((cons["start"], cons["end"]))
        if cons["label"] == "GOV":
            other_gov.append((cons["start"], cons["end"]))

    for (s1, e1) in dep_gov:
        num_gov += 1
        for (s2, e2) in gold_gov:
            if s1 >= s2 and e1 <= e2:
                for (s3, e3) in other_gov:
                    if s1 >= s3 and e1 <= e3:
                        correct_gov += 1

    for (s1, e1) in dep_obj:
        num_obj += 1
        for (s2, e2) in gold_obj:
            if s1 >= s2 and e1 <= e2:
                for (s3, e3) in other_obj:
                    if s1 >= s3 and e1 <= e3:
                        correct_obj += 1

overall_accuracy = (correct_obj + correct_gov) / (num_gov + num_obj)
gov_accuracy = correct_gov / num_gov
obj_accuracy = correct_obj / num_obj

sys.stdout.write("Overall Accuracy: " + str(overall_accuracy) + "\n")
sys.stdout.write("Governor Accuracy: " + str(gov_accuracy) + "\n")
sys.stdout.write("Object Accuracy: " + str(obj_accuracy) + "\n")




    


    
