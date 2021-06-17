import os
import sys
import json

predictionFolder = sys.argv[1]
testDataFile = sys.argv[2]

testData = []
with open(testDataFile, "r") as f:
    testData = f.readlines()

supersense1_correct = 0
supersense2_correct = 0
combined_supersense_correct = 0
total_instances = 0

for line in testData:
    line = line.split()
    fname = line[0]

    separator_idx = line.index("|||")

    supersense1 = line[separator_idx+1]
    supersense2 = line[separator_idx+2]

    prediction = {}
    with open(os.path.join(predictionFolder, fname + ".txt"), "r") as f:
        prediction = json.load(f)

    prep_indices_gold = []
    gold_tags = line[separator_idx+3:]
    for i in range(len(gold_tags)):
        if "PREP" in gold_tags[i]:
            prep_indices_gold.append(gold_tags.index(gold_tags[i])) 
    
    instance_predictions = prediction["prepositions"]
    supersense_prediction = ""
    for pred in instance_predictions:
        if pred["predicate_index"][0] in prep_indices_gold:
            supersense_prediction = pred["tags"][pred["predicate_index"][0]]
        else:
            continue
    senses = supersense_prediction.split(".")
    #print(str(senses))
    if len(senses) == 4:
        supersense1_pred = "p." + senses[1]
        supersense2_pred = "p." + senses[3]
    else:
        supersense1_pred = ""
        supersense2_pred = ""

    total_instances = total_instances + 1
    if supersense1_pred == supersense1:
        supersense1_correct = supersense1_correct + 1
    else:
        print("Pred: " + supersense1_pred + ", Actual: " + supersense1)
    if supersense2_pred == supersense2:
        supersense2_correct = supersense2_correct + 1
    else:
        print("Pred: " + supersense2_pred + ", Actual: " + supersense2)
    if supersense1_pred == supersense1 and supersense2_pred == supersense2:
        combined_supersense_correct = combined_supersense_correct + 1
    else:
        print("Pred: (" + supersense1_pred + ", " + supersense2_pred + "), Actual: (" + supersense1 + ", " + supersense2 + ")")

ss1_acc = supersense1_correct / total_instances
ss2_acc = supersense2_correct / total_instances
ss_acc = combined_supersense_correct / total_instances

print("Supersense1 Accuracy: " + str(ss1_acc))
print("Supersense2 Accuracy: " + str(ss2_acc))
print("Combined Supersense Accuracy: " + str(ss_acc))
