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

top_k=5
word_sim_ind_1="cosine"
word_sim_ind_2="euclidean"

#ds_nm_1="Anatomy"
#ds_nm_2="LargeBio"

data_param_json="ontosim.json"

###############
def assignVar():

    conf = {
        "conf_arr": [
            {
                'ind' : 'source',
                'ip_fl_nm': DATA_DIR+'fastentity/source_fast.json',
                'op_fl_path': DATA_DIR+'gnnentity/source_gnn.json'
            },
            {
                'ind' : 'target',
                'ip_fl_nm': DATA_DIR+'fastentity/target_fast.json',
                'op_fl_path': DATA_DIR+'gnnentity/target_gnn.json'
            }
        ]
    }

    return conf
##################
def loadFile(conf):
    fl_nm = code_path+conf['ip_fl_nm']
    with open(fl_nm) as f:
        data = json.load(f)

    return data
##################
def populateEdgesPerEntity(entity_obj, edges):
    centerNode=entity_obj['iri']
    edges.append([centerNode, 'self', centerNode])

    if(entity_obj['parentCls']):
      for parent in entity_obj['parentCls']:
        edges.append([centerNode, 'parent', parent])
        edges.append([parent, 'parent', centerNode])

    if(entity_obj['childCls']):
      for child in entity_obj['childCls']:
        edges.append([centerNode, 'child', child])
        edges.append([child, 'child', centerNode])

    if(entity_obj['eqCls']):
      for eq in entity_obj['eqCls']:
        edges.append([centerNode, 'equivalent', eq])
        edges.append([eq, 'equivalent', centerNode])

    if(entity_obj['disjointCls']):
      for disjoint in entity_obj['disjointCls']:
        edges.append([centerNode, 'disjoint', disjoint])
        edges.append([disjoint, 'disjoint', centerNode])

    if(entity_obj['restriction']):
      for restriction in entity_obj['restriction']:
        expression = restriction.replace("(", "( ").replace(",", " , ").replace(")", " ) ")
        tokens = expression.split(" ")
        token_lst = []
        for token in tokens:
          if("" != token and " " != token and "," != token):
            token_lst.append(token)
        edges.append([centerNode, 'restriction', token_lst[-2]]) #this needs tobe changed with SHUNTING YARD ALGORITHM
        edges.append([token_lst[-2], 'restriction', centerNode])

    return edges
###################
def saveEdgePerEntity(entity_info, conf_val):
    with open(code_path+conf_val["op_fl_path"], 'w') as outfile:
        json.dump(entity_info, outfile, indent=4)
##################
def populateAndSaveEdges(conf_val, entity_info):
  for entity in entity_info:
    entity_obj=entity_info[entity]
    edges = []
    edges = populateEdgesPerEntity(entity_obj, edges)
    entity_obj['graphEdges']=edges
    entity_info[entity]=entity_obj

  saveEdgePerEntity(entity_info, conf_val)
###################
#################### MAIN CODE START ####################
def populateEdges():
    try:
        print("#################### PopulateEdges START ####################")
        conf = assignVar()
        #####Mean of vectors
        conf_arr = conf["conf_arr"]
        for conf_val in conf_arr:
            entity_info = loadFile(conf_val)
            populateAndSaveEdges(conf_val, entity_info)
            time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### PopulateEdges FINISH ####################")

#################### MAIN CODE END ####################
##########
### MAIN FUNCTION
if __name__=="__main__":
  populateEdges()
