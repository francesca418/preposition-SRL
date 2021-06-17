import sys
import os

goldData = sys.argv[1]
silverData = sys.argv[2]

goldLines = open(goldData, "r").readlines()
silverLines = open(silverData, "r").readlines()

false_positives = 0 #labeled something as an argument when it should not be
false_negatives = 0 #did not label something as an argument when it should be
true_positives = 0 #correctly labeled something as an argument
true_negatives = 0 #correctly did not label a non-argument

for sline in silverLines:
    str_list = sline.strip().split()
    separator_index = str_list.index("|||")
    sentence = str_list[1:separator_index]
    sentence_string = ""
    for i in range(1, separator_index):
        sentence_string = sentence_string + str_list[i] + " "
    supersense1 = str_list[separator_index + 1]
    supersense2 = str_list[separator_index + 2]
    tags = str_list[(separator_index + 1):]
    try:
        predicate_location = str_list.index("B-PREP")
    except ValueError:
        continue

    #sys.stdout.write(sentence_string + "\n")

    for gline in goldLines:
        str_list2 = gline.strip().split()
        separator_index2 = str_list2.index("|||")
        sentence2 = str_list2[1:separator_index2]
        sentence_string2 = ""
        for i in range(1, separator_index2):
            sentence_string2 = sentence_string2 + str_list2[i] + " "
        supersense12 = str_list2[separator_index2 + 1]
        supersense22 = str_list2[separator_index2 + 2]
        tags2 = str_list2[(separator_index2 + 1):]
        predicate_location2 = str_list2.index("B-PREP")

        #sys.stdout.write(sentence_string2 + "\n")
        
        if sentence_string is sentence_string2:
            sys.stdout.write("HERE\n")
            if predicate_location == predicate_location2:
                # contributes to PRFA
                assert(len(tags) == len(tags2))

                for i in range(len(tags2)):
                    # false
                    if tags[i] != tags2[i]:
                        # false positive
                        if tags2[i] == "O":
                            false_positives = false_positives + 1
                        elif tags2[i][2:] == tags[i][2:]:
                            true_positives = true_positives + 1
                        else: # false negative
                            false_negatives = false_negatives + 1

                    else: # true
                        # true positive
                        if tags2[i] != "O":
                            true_positives = true_positives + 1
                        else: #true negative
                            true_negatives = true_negatives + 1
       # sys.stdout.write("Count: " + str(true_negatives + true_positives + false_negatives + false_positives) + "\n")

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (recall * precision) / (recall + precision)
accuracy = (true_positives + true_negatives) / (true_negatives + true_positives + false_negatives + false_positives)

sys.stdout.write("Accuracy: " + str(accuracy) + "\n")
sys.stdout.write("Precision: " + str(precision) + "\n")
sys.stdout.write("Recall: " + str(recall) + "\n")
sys.stdout.write("F1: " + str(f1) + "\n")

                        
