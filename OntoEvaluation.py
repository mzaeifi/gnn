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
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
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

join_str_cnst="-#-"

top_k=5
word_sim_ind_1="cosine"
word_sim_ind_2="euclidean"

ds_nm_1="Anatomy"
ds_nm_2="LargeBio"

data_param_json="ontosim.json"

######

###############
def assignVar():
    conf = {
        'gold_copy_fl': DATA_DIR+"/gold_copy/reference.xml",
        'op_fl': DATA_DIR+'/output/output_final.json',
        'txt_pos_fl': DATA_DIR+"/result/pred_op_pos.txt",
        'txt_neg_fl': DATA_DIR+"/result/pred_op_neg.txt"
    }

    return conf
#############

def populategoldCopy(conf):
    gold_copy_fl = conf["gold_copy_fl"]
    gold_copy_root = ET.parse(gold_copy_fl).getroot()
    gold_copy_dict = {}
    for child in gold_copy_root.findall('map/Cell'):
        gold_copy_dict[child[0].get('resource')] = child[1].get('resource') #mouse(MA_)=human(NCI_)'

    return gold_copy_dict
############
def opCopy(conf):
    with open(code_path+conf['op_fl']) as f:
      op_data = json.load(f)

    op_dict={}
    for op in op_data:
      entity_1=op['entity1']
      entity_2=op['entity2']
      op_dict[entity_1]=entity_2
    
    return op_dict
##############
def testOntoConn(conf, gold_copy_dict, op_dict):
    txt_pos_fl = open(conf["txt_pos_fl"], "a")
    txt_neg_fl = open(conf["txt_neg_fl"], "a")

    A = 0
    R = 0
    tp = 0
    for op in op_dict:
        A = A+1
        if(op in gold_copy_dict):
            gold_op = gold_copy_dict[op]
            if(gold_op in op_dict[op].split(join_str_cnst)):
                tp = tp + 1
                txt_pos_fl.write("Target Key:- "+op+"\n")
                txt_pos_fl.write("Actual OUTPUT:- "+gold_op+"\n")
                txt_pos_fl.write("Predicted OUTPUT:- "+op_dict[op]+"\n")
                txt_pos_fl.write("########################################## \n")
            else:
                txt_neg_fl.write("Target Key:- "+op+"\n")
                txt_neg_fl.write("Actual OUTPUT:- "+gold_op+"\n")
                txt_neg_fl.write("Predicted OUTPUT:- "+op_dict[op]+"\n")
                txt_neg_fl.write("########################################## \n")


    R = len(gold_copy_dict)

    precision = tp / A
    recall = tp / R
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F measure: " + str((2 * precision * recall) / (precision + recall)))

    txt_pos_fl.close()
    txt_neg_fl.close()
##################
def ontoEval(file_arr):
  try:
        print("#################### OntoEvaluation START ####################")
        for fl in file_arr:
          conf = assignVar()
          conf["op_fl"] = DATA_DIR+fl
          print("~~~~~~~~~~~~"+fl)
          gold_copy_dict=populategoldCopy(conf)
          op_dict=opCopy(conf)
          testOntoConn(conf, gold_copy_dict, op_dict)
          print("=====================================")
  except Exception as exp:
    raise exp
  finally:
    print("#################### OntoEvaluation FINISH ####################")
###################
if __name__=="__main__":
  file_arr = ['/output/1_99_output_final.json',
              '/output/1_98_output_final.json',
              '/output/1_97_output_final.json',
              '/output/1_96_output_final.json',
              '/output/1_95_output_final.json',
              '/output/1_94_output_final.json',
              '/output/1_93_output_final.json',
              '/output/1_92_output_final.json',
              '/output/1_91_output_final.json',
              '/output/1_90_output_final.json',
              '/output/3_99_output_final.json',
              '/output/3_98_output_final.json',
              '/output/3_97_output_final.json',
              '/output/3_96_output_final.json',
              '/output/3_95_output_final.json',
              '/output/3_94_output_final.json',
              '/output/3_93_output_final.json',
              '/output/3_92_output_final.json',
              '/output/3_91_output_final.json',
              '/output/3_90_output_final.json',
              '/output/5_99_output_final.json',
              '/output/5_98_output_final.json',
              '/output/5_97_output_final.json',
              '/output/5_96_output_final.json',
              '/output/5_95_output_final.json',
              '/output/5_94_output_final.json',
              '/output/5_93_output_final.json',
              '/output/5_92_output_final.json',
              '/output/5_91_output_final.json',
              '/output/5_90_output_final.json',
           ]
  ontoEval(file_arr)
##################

