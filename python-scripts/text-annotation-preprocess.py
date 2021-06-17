# the code for pre-processing streusle data to be annotated in TALEN for object and governor

import json
import sys
import os

inPath = sys.argv[1]
outDir= sys.argv[2] # directory to dump all the new files into

def convert():
  '''
  Load the Streusle dataset and convert into the format that can be annotated in TALEN.
  This makes each sentence into separate file(s).
  For example, the sentence: 
  "I went to the store on Monday" 
  would lead to 2 resulting documents, the first of which would have the preposition "to" highlighted, 
  and the second having "on" highlighted.
  This is for the facilitation of annotating governors and objects of the preposition, 
  where we need it to be unambiguous which objects and governors are attached to which prepositions.
  '''

  sentence_dict_list = []

  with open(inPath, "r") as f:
    sentence_dict_list = json.load(f)

  # iterating over each sentence (only get sentences which have prepositions - start by just getting all sentences)
  for sentence_dict in sentence_dict_list:

    prep_list = []

    # get the number of prepositions in the sentence
    num_preps = 0
    token_counter = 0
    for token_dict in sentence_dict["toks"]:
      if token_dict["upos"] == "ADP": # we know it is an ADP
        token_num = token_dict["#"]
        for mw_dict in sentence_dict["smwes"].values():
          if token_num == mw_dict["toknums"][0]:
            endpoint = token_counter + len(mw_dict["toknums"])
            prep_list.append({"preposition": mw_dict["lexlemma"], "start": token_counter, "end": endpoint})
            num_preps += 1
        for w_dict in sentence_dict["swes"].values():
          if token_num == w_dict["toknums"][0]:
            new_prep = True
            for mw_dict in sentence_dict["smwes"].values():
              if w_dict["toknums"][0] in mw_dict["toknums"]:
                new_prep = False
            if new_prep:
              endpoint = token_counter + 1
              prep_list.append({"preposition": w_dict["lexlemma"], "start": token_counter, "end": endpoint})
              num_preps += 1
      token_counter += 1

    if num_preps <= 0:
      continue

    # make a new annotation document for each annotatable preposition in each sentence
    for val in range(num_preps):

      doc = {}
      doc["corpusId"] = ""
      doc["id"] = sentence_dict["sent_id"] + "_" + str(val)
      doc["text"] = ""
      num = 0
      for token_dict in sentence_dict["toks"]:
        num = num + 1
        if num == len(sentence_dict["toks"]):
          doc["text"] = doc["text"] + token_dict["word"]
        else:
          doc["text"] = doc["text"] + token_dict["word"] + " "
      doc["tokens"] = [x["word"] for x in sentence_dict["toks"]]

      doc["tokenOffsets"] = []
      token_offset_counter = 0
      for token_dict in sentence_dict["toks"]:
        offset_dict = {}
        offset_dict["form"] = token_dict["word"]
        offset_dict["startCharOffset"] = token_offset_counter
        token_offset_counter += len(token_dict["word"])
        offset_dict["endCharOffset"] = token_offset_counter
        token_offset_counter += 1
        doc["tokenOffsets"].append(offset_dict)

      doc["sentences"] = {}
      doc["sentences"]["generator"] = "UserSpecified"
      doc["sentences"]["score"] = 1.0
      sentence_length = len(sentence_dict["toks"])
      doc["sentences"]["sentenceEndPositions"] = [sentence_length]

      # setting up the document views
      doc["views"] = []

      # sentence view
      sentence_view_dict = {}
      sentence_view_dict["viewName"] = "SENTENCE"
      sentence_view_dict["viewData"] = []
      data_dict = {}
      data_dict["viewType"] = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.SpanLabelView"
      data_dict["viewName"] = "SENTENCE"
      data_dict["generator"] = "UserSpecified"
      data_dict["score"] = 1.0
      data_dict["constituents"] = []
      cons_dict = {}
      cons_dict["label"] = "SENTENCE"
      cons_dict["score"] = 1.0
      cons_dict["start"] = 0
      cons_dict["end"] = len(sentence_dict["toks"])
      data_dict["constituents"].append(cons_dict)
      sentence_view_dict["viewData"].append(data_dict)
      doc["views"].append(sentence_view_dict)

      # tokens view
      token_view_dict = {}
      token_view_dict["viewName"] = "TOKENS"
      token_view_dict["viewData"] = []
      data_dict = {}
      data_dict["viewType"] = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView"
      data_dict["viewName"] = "TOKENS"
      data_dict["generator"] = "UserSpecified"
      data_dict["score"] = 1.0
      data_dict["constituents"] = []
      token_counter = 0
      for _ in range(len(sentence_dict["toks"])):
        cons_dict = {}
        cons_dict["label"] = ""
        cons_dict["score"] = 1.0
        cons_dict["start"] = token_counter
        token_counter = token_counter + 1
        cons_dict["end"] = token_counter
        data_dict["constituents"].append(cons_dict)
      token_view_dict["viewData"].append(data_dict)
      doc["views"].append(token_view_dict)

      # add preposition SRL view - this will have preposition (PREP) labels but will need to be tagged with GOV and OBJ
      prepsrl_view_dict = {}
      prepsrl_view_dict["viewName"] = "PREPOSITION_SRL"
      prepsrl_view_dict["viewData"] = []
      data_dict = {}
      data_dict["viewType"] = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.View"
      data_dict["viewName"] = "PREPOSITION_SRL"
      data_dict["generator"] = "UserSpecified"
      data_dict["score"] = 1.0
      data_dict["constituents"] = []
      cons_dict = {}
      cons_dict["label"] = "PREP"
      cons_dict["score"] = 1.0
      current_preposition_dict = prep_list[val]
      cons_dict["start"] = current_preposition_dict["start"]
      cons_dict["end"] = current_preposition_dict["end"]
      data_dict["constituents"].append(cons_dict)
      prepsrl_view_dict["viewData"].append(data_dict)
      doc["views"].append(prepsrl_view_dict)

      # write this out to a json file
      with open(os.path.join(outDir, doc["id"]), "w+") as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)

convert()
