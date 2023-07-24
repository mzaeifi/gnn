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
        "model": 'fast.bin',
        "conf_arr": [
            {
                'dict_fl_nm': DATA_DIR+'dict/dict_fast.json',
                'ip_fl_nm': DATA_DIR+'modifylbl/source.json',
                'op_fl_nm': DATA_DIR+'fastentity/source_fast.json'
            },
            {
                'dict_fl_nm': DATA_DIR+'dict/dict_fast.json',
                'ip_fl_nm': DATA_DIR+'modifylbl/target.json',
                'op_fl_nm': DATA_DIR+'fastentity/target_fast.json'
            }
        ]
    }

    return conf
################

def getDataParam():
    data_param_fl_nm = code_path + data_param_json
    with open(data_param_fl_nm) as f:
        data_param = json.load(f)
    return data_param
    
################
def loadDictVec(conf):
    fl_nm = code_path + conf['dict_fl_nm']
    with open(fl_nm) as f:
        data = json.load(f)

    return data
#################
def loadFile(conf):
    fl_nm = code_path+conf['ip_fl_nm']
    with open(fl_nm) as f:
        data = json.load(f)

    return data
###############
def getFastTxtVecMean(conf, data, entity_dict_vec, db_param):

    for key in data:
        words = data[key]['altLbl'].split()
        embed_vec = np.asarray([0.0] * db_param["vec_dim"]) #eg 300d for fastText
        for word in words:
            embed_vec = np.add(embed_vec, np.asarray(entity_dict_vec[word]))

        embed_vec /= len(words)
        data[key]['vector'] = embed_vec.tolist()

    return data
##################
def saveFile(fast_dict, conf):
    with open(code_path+conf['op_fl_nm'], 'w') as outfile:
        json.dump(fast_dict, outfile, indent=4)
##################
#################### MAIN CODE START ####################
def entityToVec(db_param):
    try:
        print("#################### EntityToVector START ####################")
        conf = assignVar()
        #####Mean of vectors
        conf_arr = conf["conf_arr"]
        for conf_val in conf_arr:
            entity_dict_vec = loadDictVec(conf_val)
            data = loadFile(conf_val)
            fast_dict = getFastTxtVecMean(conf_val, data, entity_dict_vec, db_param)
            saveFile(fast_dict, conf_val)
            time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### EntityToVector FINISH ####################")

#################### MAIN CODE END ####################
### MAIN FUNCTION
if __name__=="__main__":
  data_param = getDataParam()
  entityToVec(data_param['db'])
