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

#frm_vic_1="model"
#frm_vic_2="vec-fl"
#faxt_text_model_path='/content/drive/MyDrive/OntoConn/FastText/300d/'
#vec_nm=""

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
        'fl_nm_arr': [DATA_DIR+'modifylbl/source.json',
                      DATA_DIR+'modifylbl/target.json'],
        'dict_fl': DATA_DIR+'/dict/dict.txt'
    }

    return conf
################
def loadFile(conf_fl):
    fl_nm = code_path + conf_fl
    with open(fl_nm) as f:
        data = json.load(f)

    return data
################
def crtDict(conf):
    onto_unq_words = []
    for conf_fl in conf["fl_nm_arr"]:
        data = loadFile(conf_fl)
        for key in data.keys():
            words = data[key]['altLbl'].split()
            for word in words:
                if word not in onto_unq_words:
                    onto_unq_words.append(word)

    print('No of Unique Words:- ' + str(len(onto_unq_words)))
    return onto_unq_words
##################
def writeDict(conf, entity_dict):
    with open(code_path + conf['dict_fl'], "w") as file:
        for entity in entity_dict:
            file.write(entity + "\n")
###################
def crtDictMain():
    try:
        print("#################### CreateDictionary START ####################")
        conf = assignVar()
        entity_dict = crtDict(conf)
        writeDict(conf, entity_dict)

        time.sleep(wait_time)
    except Exception as exp:
        raise exp
    finally:
        print("#################### CreateDictionary FINISH ####################")
####################
### MAIN FUNCTION
if __name__=="__main__":
  crtDictMain()
