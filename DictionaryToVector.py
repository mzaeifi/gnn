from IPython.display import Javascript  # Restrict height of output cell.
import os
import json
import re
import time
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 100})'''))
import torch

#imports
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import copy
import datetime
import traceback
import numpy as np
import scipy.spatial.distance
import numpy as np
from collections import OrderedDict
import shutil
import h5py
from pathlib import Path
import math
import statistics
import xml.etree.ElementTree as ET
import fasttext
import fasttext.util
from fasttext import load_model
from torchbiggraph.config import parse_config
#from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

DIR=''
DATA_DIR=DIR+"data/"
code_path=""
wait_time=15

frm_vic_1="model"
frm_vic_2="vec-fl"
faxt_text_model_path='FastText/300d/'
vec_nm=""

#join_str_cnst="-#-"

#top_k=5
#word_sim_ind_1="cosine"
#word_sim_ind_2="euclidean"

#ds_nm_1="Anatomy"
#ds_nm_2="LargeBio"

data_param_json="ontosim.json"

################
def assignVar():
    conf = {
        'dict_fl': DATA_DIR+'dict/dict.txt',
        'dict_json': DATA_DIR+'dict/dict_fast.json',
        'model': 'fast.bin',
        'vec_fl': vec_nm
    }

    return conf
#################
def readDict(conf):
    with open(code_path + conf["dict_fl"], 'r') as f:
        entity_dict = f.readlines()

    return entity_dict
################
def loadDictVec(entity_dict, fasttext_model):
    fast_dict = {}
    for entity_word in entity_dict:
        entity_word = entity_word.replace("\n", "")
        vec = fasttext_model.get_word_vector(entity_word)
        fast_dict[entity_word] = vec.tolist()

    return fast_dict
################
def writeDictVec(conf, entity_dict):
    with open(code_path + conf['dict_json'], 'w') as outfile:
        json.dump(entity_dict, outfile, indent=4)
###############
def loadPreTrainedDictVec(conf):
    fl_nm = faxt_text_model_path + conf['vec_fl']
    with open(fl_nm) as f:
        data = json.load(f)
    return data
#################

def chkAllKeyPresentInVec(entity_dict_pretrained, entity_dict_current):
    for entity_key in entity_dict_current:
        entity_key = entity_key.replace("\n", "")
        if(entity_key not in entity_dict_pretrained.keys()):
            return False
    return True
##################
def dictToVecFile(ind):
    print("Creating Vector from "+ ind)
    conf = assignVar()
    entity_dict_pretrained = loadPreTrainedDictVec(conf)
    entity_dict_current = readDict(conf)
    if(chkAllKeyPresentInVec(entity_dict_pretrained, entity_dict_current)):
        writeDictVec(conf, entity_dict_pretrained)
    else:
        raise Exception("Sorry, key is missing in pre-trained vector file")
###################
def dictToVecModel(ind):
    conf = assignVar()
    print("Creating Vector from " + ind + " " + conf['model'])
    print('fastText model Loads START:- ' + str(datetime.datetime.now()))
    fasttext_model = load_model(faxt_text_model_path + conf["model"])
    print("Model Dimension:- " + str(fasttext_model.get_dimension()))
    print('fastText model Loads END:- ' + str(datetime.datetime.now()))

    entity_dict = readDict(conf)
    entity_dict_vec = loadDictVec(entity_dict, fasttext_model)
    writeDictVec(conf, entity_dict_vec)
###################
def dictToVec(ind):
    try:
        print("#################### DictionaryToVector START ####################")
        if(ind == frm_vic_1):
            dictToVecModel(frm_vic_1)
        elif(ind == frm_vic_2):
            dictToVecFile(frm_vic_2)

        time.sleep(wait_time)
    except Exception as exp:
        raise exp
    finally:
        print("#################### DictionaryToVector FINISH ####################")
##################
### MAIN FUNCTION
if __name__=="__main__":
  dictToVec(frm_vic_1)
