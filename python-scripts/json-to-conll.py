'''
This script converts Test Annotation JSON formatted preposition SRL data to conll formatted SRL data

data will be in a single file in the format:
filename sentence ||| supersense1 supersense2 sentence_tags
'''
import json
import sys
import os
#import pprint

def convert():

    #files_to_ignore = open("files_to_ignore.txt").readlines()
    #files_to_ignore = [f[:-1] for f in files_to_ignore]

    streusle = []
    with open ("streusle/test/streusle.ud_test.govobj.json", "r") as f:
        streusle = json.load(f)

    with open("clean-test.txt", "w+") as f:
        f.write("")

    count = 0

    for fileName in os.listdir("streusle-clean-ta-test/"):

        #if fileName in files_to_ignore:
            #continue

        current_dict = {}
        with open(os.path.join("streusle-clean-ta-test", fileName), "r") as f:
            current_dict = json.load(f)

        output_line = ""
        output_line = fileName + " "

        sent_id = current_dict["id"]
        sent_id = sent_id.replace('_', '-')
        sent_id = sent_id.rsplit('-', 1)
        sent_id = sent_id[0]

        sentence_dict = {}
        sentence = ""
        for x in streusle:
            if sent_id == x["sent_id"]:
                sentence_dict = x
                sentence_list = x["toks"]
                for word_dict in sentence_list:
                    sentence = sentence + word_dict["word"] + " "

        output_line = output_line + sentence + "||| "

        cons_dict = {}

        temp_list = current_dict["views"]
        try:
            for x in temp_list:
                if x["viewName"] == "PREPOSITION_SRL":
                    cons_dict = x["viewData"][0]["constituents"]
        except KeyError:
            continue

        tags = ["O" for i in range(len(current_dict["tokens"]))]

        single_words = sentence_dict["swes"]
        multi_words = sentence_dict["smwes"]
        supersense1 = ""
        supersense2 = ""

        predicate = []
        for cons in cons_dict:
            start = cons["start"]
            end = cons["end"]
            label = cons["label"]
            predicate = current_dict["tokens"][start:end]
            if label == "PREP":
                if end - start == 1:
                    try:
                        supersense1 = single_words[str(end)]["ss"] 
                        supersense2 = single_words[str(end)]["ss2"] 
                    except KeyError:
                        sys.stdout.write(str(sentence_dict) + "\n" + str(i) + "\n")
                        continue
                else:
                    try:                      
                        for i in range(len(multi_words) + 20):
                            if str(i) not in multi_words.keys():
                                continue
                            mw = multi_words[str(i)]["lexlemma"]
                            mw = mw.split()
                            if len(mw) != len(predicate):
                                continue
                            tester = True
                            for j in range(len(predicate)):
                                if predicate[j].casefold() == mw[j].casefold():
                                    continue
                                else:
                                    tester = False
                                    break
                            if tester: 
                                supersense1 = multi_words[str(i)]["ss"] 
                                supersense2 = multi_words[str(i)]["ss2"]
                                break
                    except KeyError:
                        sys.stdout.write(str(sentence_dict) + "\n" + str(i) + "\n")
                        continue

            for idx in range(start, end):
                if idx == start:
                    tags[start] = "B-" + label
                else:
                    tags[idx] = "I-" + label
                                
        if supersense1 == "" or supersense2 == "" or supersense1 == None or supersense2 == None:
            output_line = ""
            continue
        output_line = output_line + supersense1 + " "
        output_line = output_line + supersense2 + " "

        for tag in tags:
            output_line = output_line + tag + " "

        with open("clean-test.txt", "a") as f:
            f.write(output_line + "\n")

        output_line = ""

convert()