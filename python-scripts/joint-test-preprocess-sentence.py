import os
import sys
import json

inFile = sys.argv[1]
outFolder = sys.argv[2]

testData = []
with open(inFile, "r") as f:
    testData = f.readlines()

for line in testData:
    line = line.split()
    fname = line[0] + ".txt"
    separator_idx = line.index("|||")
    sent_list = line[1:separator_idx]
    sentence = ""
    for i in range(len(sent_list)):
        sentence = sentence + sent_list[i] + " "
    out = {}
    out["sentence"] = sentence
    out["indices"] = "[" + str(len(sentence)) + "]"

    with open(os.path.join(outFolder, fname), "w+") as f:
        json.dump(out, f)
        
