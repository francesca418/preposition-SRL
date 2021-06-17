'''
Helper method which transforms the Streusle data from its json format in json.govobj into the following format:
{Sentence: +++++++,
PrepositionLabels: [{
    start: ,
    end: ,
    role_label: ,
    function_label: 
}]}
Eventually might want to do something with the governor - object labels as well...
'''
import json
import sys
import os

inPath = sys.argv[1]
outPath= sys.argv[2]

def convert():
    '''
    Load the Streusle Dictionary which is of the following format:
        List[Dict[sent_id: str, text: str, streusle_sent_id: str, mwe: str, toks: List[Dict[]], etoks: List[Dict[]] ]]
    Convert into the following format:
        List[Dict[Sentence, PrepositionLabels]]

    '''
    sentence_dict_list_to_return = []
    sentence_dict_list = []

    with open(inPath, "r") as f:
        sentence_dict_list = json.load(f)

    # iterating over each sentence (only get sentences which have prepositions - start by just getting all sentences)
    i = 0
    for sentence_dict in sentence_dict_list:

        # check if the sentence has prepositions, and if it does not, skip it
        has_preps = False
        for token_dict in sentence_dict["toks"]:
            if (token_dict["upos"] == "ADP"):
                has_preps = True
        if not has_preps:
            continue

        # otherwise, there are prepositions, and we want to train on this sentence
        new_dict = {}

        # copy over fields from the old JSON doc which could be useful
        new_dict["sent_id"] = sentence_dict["sent_id"]
        new_dict["text"] = sentence_dict["text"]
        new_dict["streusle_sent_id"] = sentence_dict["streusle_sent_id"]

        new_dict["tokens"] = []
        new_dict["preposition_labels"] = []

        index_counter = 0

        # get all of the individual tokens (based on whatever original tokenizer they used)
        for token_dict in sentence_dict["toks"]:
            word_dict = {}
            word_dict["token"] = token_dict["word"] 
            word_dict["token_number"] = token_dict["#"]
            word_dict["start_index"] = index_counter
            word_dict["end_index"] = index_counter + len(token_dict["word"]) - 1
            index_counter = index_counter + 1
            new_dict["tokens"].append(word_dict)
        
        # get all preposition spans - starting with any possible multiword expressions (then single word prepositions)
        prep_dict = {}
        for mw_dict in sentence_dict["smwes"].values():
            if mw_dict["lexcat"] == "P":
                prep_dict["preposition"] = mw_dict["lexlemma"]
                prep_dict["token_range"] = mw_dict["toknums"]
                prep_dict["ss1"] = mw_dict["ss"]
                prep_dict["ss2"] = mw_dict["ss2"]

        for w_dict in sentence_dict["swes"].values():
            if w_dict["lexcat"] == "P":
                new_prep = True
                for mw_dict in sentence_dict["smwes"].values():
                    if w_dict["toknums"][0] in mw_dict["toknums"]:
                        new_prep = False
                if new_prep:
                    prep_dict["preposition"] = w_dict["lexlemma"]
                    prep_dict["token_range"] = w_dict["toknums"]
                    prep_dict["ss1"] = w_dict["ss"]
                    prep_dict["ss2"] =  w_dict["ss2"]
          
        new_dict["preposition_labels"].append(prep_dict)
        sentence_dict_list_to_return.append(new_dict) 

        i += 1
        #sys.stdout.write(str(i) + "\n")
    
    with open(outPath, "w+") as f:
        json.dump(sentence_dict_list_to_return, f, ensure_ascii=False, indent=4)

convert()