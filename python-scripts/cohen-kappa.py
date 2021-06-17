# script to compute Cohen's Kappa Coefficient for measuring inter-annotator agreement

import sys
import os
import json
import numpy as np

# load each file and its mate (3 - one for each annotator)

francesca_dir = "annotated-ta"
celine_dir = "streusle-annotation-celine"
nick_dir = "streusle-annotation-nick"

f_list = []
c_list = []
n_list = []

# iterate through the files concatenating tags into one large string per annotator
for filename in os.listdir(celine_dir):

    assert len(f_list) == len(c_list)
    assert len(c_list) == len(n_list)

    f_dict = {}
    c_dict = {}
    n_dict = {}
    with open(os.path.join(francesca_dir, filename), "r") as f:
        f_dict = json.load(f)
    with open(os.path.join(celine_dir, filename), "r") as f:
        c_dict = json.load(f)
    with open(os.path.join(nick_dir, filename), "r") as f:
        n_dict = json.load(f)

    sentence_length = len(c_dict["tokens"])
    f_tags = ["O" for i in range(sentence_length)]
    c_tags = ["O" for i in range(sentence_length)]
    n_tags = ["O" for i in range(sentence_length)]

    f_views = f_dict["views"]
    c_views = c_dict["views"]
    n_views = n_dict["views"]

    f_cons = []
    c_cons = []
    n_cons = []

    skip = False

    for x in f_views:
        if x["viewName"] == "NER_CONLL":
            if "constituents" not in x["viewData"][0].keys():
                skip = True
            else:
                f_cons = x["viewData"][0]["constituents"]

    for x in c_views:
        if x["viewName"] == "NER_CONLL":
            if "constituents" not in x["viewData"][0].keys():
                skip = True
            else:
                c_cons = x["viewData"][0]["constituents"]

    for x in n_views:
        if x["viewName"] == "NER_CONLL":
            if "constituents" not in x["viewData"][0].keys():
                skip = True
            else:
                n_cons = x["viewData"][0]["constituents"]

    if skip:
        continue

    for con in f_cons:
        if con["label"] == "PREP":
            continue
        tag = "O"
        if con["label"] == "GOV":
            tag = "GOV"
        else:
            tag = "OBJ"
        start = con["start"]
        end = con["end"]
        for i in range(start, end):
            f_tags[i] = tag

    for con in c_cons:
        if con["label"] == "PREP":
            continue
        tag = "O"
        if con["label"] == "GOV":
            tag = "GOV"
        else:
            tag = "OBJ"
        start = con["start"]
        end = con["end"]
        for i in range(start, end):
            c_tags[i] = tag

    for con in n_cons:
        if con["label"] == "PREP":
            continue
        tag = "O"
        if con["label"] == "GOV":
            tag = "GOV"
        else:
            tag = "OBJ"
        start = con["start"]
        end = con["end"]
        for i in range(start, end):
            n_tags[i] = tag

    f_list = f_list + f_tags
    c_list = c_list + c_tags
    n_list = n_list + n_tags

# compute relevant values
total_toks = len(f_list)

f = np.array(f_list)
c = np.array(c_list)
n = np.array(n_list)

fc_agreement_num = np.sum(f == c)
fn_agreement_num = np.sum(f == n)
cn_agreement_num = np.sum(c == n)

f_nothing_num = f_list.count("O")
f_obj_num = f_list.count("OBJ")
f_gov_num = f_list.count("GOV")

c_nothing_num = c_list.count("O")
c_obj_num = c_list.count("OBJ")
c_gov_num = c_list.count("GOV")

n_nothing_num = n_list.count("O")
n_obj_num = n_list.count("OBJ")
n_gov_num = n_list.count("GOV")

po_fc = fc_agreement_num / total_toks
po_fn = fn_agreement_num / total_toks
po_cn = cn_agreement_num / total_toks

fc_nothing_random_agree = (f_nothing_num / total_toks) * (c_nothing_num / total_toks)
fn_nothing_random_agree = (f_nothing_num / total_toks) * (n_nothing_num / total_toks)
cn_nothing_random_agree = (c_nothing_num / total_toks) * (n_nothing_num / total_toks)

fc_gov_random_agree = (f_gov_num / total_toks) * (c_gov_num / total_toks)
fn_gov_random_agree = (f_gov_num / total_toks) * (n_gov_num / total_toks)
cn_gov_random_agree = (c_gov_num / total_toks) * (n_gov_num / total_toks)

fc_obj_random_agree = (f_obj_num / total_toks) * (c_obj_num / total_toks)
fn_obj_random_agree = (f_obj_num / total_toks) * (n_obj_num / total_toks)
cn_obj_random_agree = (c_obj_num / total_toks) * (n_obj_num / total_toks)

pe_fc = fc_nothing_random_agree + fc_gov_random_agree + fc_obj_random_agree
pe_fn = fn_nothing_random_agree + fn_gov_random_agree + fn_obj_random_agree
pe_cn = cn_nothing_random_agree + cn_gov_random_agree + cn_obj_random_agree

# cohens kappa formula
kappa_fc = (po_fc - pe_fc) / (1 - pe_fc)
kappa_fn = (po_fn - pe_fn) / (1 - pe_fn)
kappa_cn = (po_cn - pe_cn) / (1 - pe_cn)

sys.stdout.write("Cohen Kappa for F and C: " + str(kappa_fc) + "\n")
sys.stdout.write("Cohen Kappa for F and N: " + str(kappa_fn) + "\n")
sys.stdout.write("Cohen Kappa for C and N: " + str(kappa_cn) + "\n")
