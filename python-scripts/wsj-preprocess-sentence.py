'''
This script takes each JSON formatted WSJ file and converts it to a sentence file of the format 

{"sentence": "This is a sentence. This is another sentence."}

The filenames are the same .txt, this way we can still pair the JSON files with the prep predictions.
'''

import json
import sys
import os

inFolder = sys.argv[1]
outFolder = sys.argv[2]

for filename in os.listdir(inFolder):
    current_dict = {}
    with open(os.path.join(inFolder, filename)) as f:
        current_dict = json.load(f)

    document_id = current_dict["corpusId"]

    # get the sentence format for the new file
    output_dict_list = [] # want to make a list of each output disctionary
    #get sentence endpoints
    sentence_end_positions = current_dict["sentences"]["sentenceEndPositions"]
    tokens = current_dict["tokens"]

    current_start = 0
    for i, end_pos in enumerate(sentence_end_positions):
        output_dict = {}
        length = len(tokens[current_start:end_pos])
        output_dict["sentence"] = (tok + " " for tok in tokens[current_start:end_pos - 1])
        current_start = end_pos
        output_dict["indices"] = "[" + str(length) + "]"
        fname = document_id + "_" + str(i) + ".txt"
        with open(os.path.join(outFolder, fname), "w+") as f:
            f.write(str(output_dict))