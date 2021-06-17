import json
import sys
import os

inFolder = sys.argv[1]
num_tokens = 0
num_unique_tokens = 0
num_unique_preps = 0
unique_toks = {}
unique_preps = {}

for filename in os.listdir(inFolder):
    sentence_dict = {}
    with open(inFolder + "/" + filename, "r") as f:
        sentence_dict = json.load(f)
    num_tokens = num_tokens + len(sentence_dict["tokens"])
    for token in sentence_dict["tokens"]:
        unique_toks[token] = 0
    for view in sentence_dict["views"]:
        #sys.stdout.write(str(view))
        if view["viewName"] == "PREPOSITION_SRL":
            cons = []
            try:
                cons = view["viewData"][0]["constituents"]
            except (KeyError): 
                pass
            #sys.stdout.write(str(cons))
            for con in cons:
                if con["label"] == "PREP":
                    #sys.stdout.write("here")
                    prep = sentence_dict["tokens"][con["start"]]
                    unique_preps[prep] = 0
num_unique_preps = len(unique_preps)
num_unique_tokens = len(unique_toks)

#sys.stdout.write(str(unique_preps) + "\n")
sys.stdout.write("Num tokens: " + str(num_tokens) + "\n")
sys.stdout.write("Num Unique Preps: " + str(num_unique_preps) + "\n")
sys.stdout.write("Num Unique Tokens: " + str(num_unique_tokens) + "\n")

