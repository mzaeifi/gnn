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
        'ws_fl_nm': DATA_DIR+'output/',
        'op_fl': DATA_DIR+'/output/output_final.json'
    }

    return conf
#############

def loadData(fl):
    with open(code_path + fl) as f:
        data = json.load(f)

    return data
############
def modifyWsMs(word_sim_lst, meta_sim_lst):
    word_sim_lst_modf = copy.deepcopy(word_sim_lst)
    meta_sim_lst_modf = copy.deepcopy(meta_sim_lst)

    word_sim_arr = [item[1] for item in word_sim_lst_modf]  # only similarity values
    meta_sim_arr = [item[1] for item in meta_sim_lst_modf]  # only similarity values

    word_sim_sd = statistics.stdev(word_sim_arr)
    meta_sim_sd = statistics.stdev(meta_sim_arr)

    k = 0.0

    if (word_sim_sd == 0 or meta_sim_sd == 0):
        return word_sim_lst_modf, meta_sim_lst_modf

    k = meta_sim_sd / word_sim_sd

    # print("K: "+str(k))

    if(k>1): # meta-sim is more spread than word-sim, reducing the spread of meta-info
        for idx, _ in enumerate(meta_sim_lst_modf):
            meta_sim_lst_modf[idx][1] = float(meta_sim_lst_modf[idx][1]) / k
    else: # word-sim is more spread than meta-sim, increasing the spread of meta-info
        for idx, _ in enumerate(word_sim_lst_modf):
            word_sim_lst_modf[idx][1] = float(word_sim_lst_modf[idx][1]) * k


    max_word_sim = 0
    max_meta_sim = 0
    for idx, _ in enumerate(meta_sim_lst_modf):
        if(meta_sim_lst_modf[idx][1] > max_meta_sim):
            max_meta_sim = meta_sim_lst_modf[idx][1]
    for idx, _ in enumerate(word_sim_lst_modf):
        if(word_sim_lst_modf[idx][1]>max_word_sim):
            max_word_sim = word_sim_lst_modf[idx][1]

    pow_word_sim = math.floor(math.log(max_word_sim))
    pow_meta_sim = math.floor(math.log(max_meta_sim))
    if(pow_word_sim>pow_meta_sim):
        pow_val = pow_word_sim - pow_meta_sim
        for idx, _ in enumerate(meta_sim_lst_modf):
            meta_sim_lst_modf[idx][1] = float(meta_sim_lst_modf[idx][1]) * pow(10, pow_val)
    else:
        pow_val = pow_meta_sim - pow_word_sim
        for idx, _ in enumerate(word_sim_lst_modf):
            word_sim_lst_modf[idx][1] = float(word_sim_lst_modf[idx][1]) * pow(10, pow_val)

    return word_sim_lst_modf, meta_sim_lst_modf
###################
def saveFinalOP(final_op, conf, db_param):
  with open(code_path+db_param['op_fl'], 'w') as outfile:
    json.dump(final_op, outfile, indent=4)
##################

# meta similarity
def ontoEvalMS(dist_ind, db_param):

    try:
        print("#################### OntoEvaluation START ####################")

        conf = assignVar()

        word_sim_info = None
        meta_sim_info = None
        if (dist_ind == word_sim_ind_1):  # word_sim_ind_1 = "cosine"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_cosine.json")
            meta_sim_info = loadData(conf["ws_fl_nm"] + "meta_sim/meta_sim_cosine.json")
        elif (dist_ind == word_sim_ind_2):  # word_sim_ind_2 = "euclidean"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_euclidean.json")
            meta_sim_info = loadData(conf["ws_fl_nm"] + "meta_sim/meta_sim_euclidean.json")


        word_wt = 0.5
        meta_wt = 0.5
        threshold = 0.0
        if (db_param["db_nm"] == ds_nm_1):
            word_wt = db_param["word_wt_ds"]
            meta_wt = db_param["meta_wt_ds"]
            threshold = db_param["threshold_ds"]

        final_op = []
        for trgt_key in word_sim_info:

            word_sim_lst = [[key, sim] for key, sim in word_sim_info[trgt_key].items()]
            meta_sim_lst = [[key, sim] for key, sim in meta_sim_info[trgt_key].items()]

            word_sim_lst_modf, meta_sim_lst_modf = modifyWsMs(word_sim_lst, meta_sim_lst)
            pred_sim_lst = []
            for idx, word_sim in enumerate(word_sim_lst_modf):
                word_sim_key, word_sim_val = word_sim[0], word_sim[1]
                for m_key, m_val in meta_sim_lst_modf:
                    if(m_key == word_sim_key):
                        new_sim_val = word_wt * word_sim_val + meta_wt * m_val
                        pred_sim_lst.append([word_sim_key, new_sim_val])


            pred_sim_sort = sorted(pred_sim_lst, key=lambda x: x[1], reverse=False)

            pred_final_op_lst = []
            for val, msr in pred_sim_sort:
                if(msr<threshold and len(pred_final_op_lst)<db_param["op_k"]):
                    pred_final_op_lst.append(val)

            join_str=join_str_cnst
            if(len(pred_final_op_lst)>0):
                tmp_op = {"entity1": "", "entity2": "", "measure": ""}
                tmp_op['entity1'] = trgt_key
                tmp_op['entity2'] = join_str.join(pred_final_op_lst)
                tmp_op['measure'] = 1.0
                final_op.append(tmp_op)

        saveFinalOP(final_op, conf, db_param)
        time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### OntoEvaluation FINISH ####################")
############
# word similarity
def ontoEvalWS(dist_ind, db_param):

    try:
        print("#################### OntoEvaluation START ####################")

        conf = assignVar()

        word_sim_info = None
        if (dist_ind == word_sim_ind_1):  # word_sim_ind_1 = "cosine"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_cosine.json")
        elif (dist_ind == word_sim_ind_2):  # word_sim_ind_2 = "euclidean"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_euclidean.json")


        threshold = 0.0
        if (db_param["db_nm"] == ds_nm_1):
            threshold = db_param["threshold_ds"]

        final_op = []
        for trgt_key in word_sim_info:

            word_sim_lst = [[key, sim] for key, sim in word_sim_info[trgt_key].items()]
            pred_sim_lst = word_sim_lst
            pred_sim_sort = sorted(pred_sim_lst, key=lambda x: x[1], reverse=False)

            pred_final_op_lst = []
            for val, msr in pred_sim_sort:
                if(msr<threshold and len(pred_final_op_lst)<db_param["op_k"]):
                    pred_final_op_lst.append(val)

            join_str=join_str_cnst
            if(len(pred_final_op_lst)>0):
                tmp_op = {"entity1": "", "entity2": "", "measure": ""}
                tmp_op['entity1'] = trgt_key
                tmp_op['entity2'] = join_str.join(pred_final_op_lst)
                tmp_op['measure'] = 1.0
                final_op.append(tmp_op)

        saveFinalOP(final_op, conf, db_param)
        time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### OntoEvaluation FINISH ####################")
###########

def ontoEvalCS(db_param):

    try:
        print("#################### OntoEvaluation START ####################")

        conf = assignVar()

        word_sim_info = None
        meta_sim_info = None
        if (db_param["sim_ind"] == word_sim_ind_1):  # word_sim_ind_1 = "cosine"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_cosine.json")
            meta_sim_info = loadData(conf["ws_fl_nm"] + "meta_sim/meta_sim_cosine.json")
        elif (db_param["sim_ind"] == word_sim_ind_2):  # word_sim_ind_2 = "euclidean"
            word_sim_info = loadData(conf["ws_fl_nm"] + "word_sim/word_sim_euclidean.json")
            meta_sim_info = loadData(conf["ws_fl_nm"] + "meta_sim/meta_sim_euclidean.json")


        word_wt = 0.5
        meta_wt = 0.5
        threshold = 0.0
        if (db_param["db_nm"] == ds_nm_1):
            word_wt = db_param["word_wt_ds"]
            meta_wt = db_param["meta_wt_ds"]
            threshold = db_param["threshold_ds"]

        final_op = []
        for trgt_key in word_sim_info:

            word_sim_lst = [[key, sim] for key, sim in word_sim_info[trgt_key].items()]
            meta_sim_lst = [[key, sim] for key, sim in meta_sim_info[trgt_key].items()]

            word_sim_lst_modf, meta_sim_lst_modf = modifyWsMs(word_sim_lst, meta_sim_lst)
            pred_sim_lst = []
            for idx, word_sim in enumerate(word_sim_lst_modf):
                word_sim_key, word_sim_val = word_sim[0], word_sim[1]
                if (word_sim_val == 0):
                    pred_sim_lst = word_sim_lst
                    break
                else:
                    for m_key, m_val in meta_sim_lst_modf:
                        if(m_key == word_sim_key):
                            new_sim_val = word_wt * word_sim_val + meta_wt * m_val
                            pred_sim_lst.append([word_sim_key, new_sim_val])

            pred_sim_sort = sorted(pred_sim_lst, key=lambda x: x[1], reverse=False)

            pred_final_op_lst = []
            for val, msr in pred_sim_sort:
                if(msr<threshold and len(pred_final_op_lst)<db_param["op_k"]):
                    pred_final_op_lst.append(val)

            join_str=join_str_cnst
            if(len(pred_final_op_lst)>0):
                tmp_op = {"entity1": "", "entity2": "", "measure": ""}
                tmp_op['entity1'] = trgt_key
                tmp_op['entity2'] = join_str.join(pred_final_op_lst)
                tmp_op['measure'] = 1.0
                final_op.append(tmp_op)

        saveFinalOP(final_op, conf, db_param)
        time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### OntoEvaluation FINISH ####################")
#################
# ////// VECTOR-DIMENSION START //////
# // int vec_dim = 100;
# //int vec_dim = 200;
# // int vec_dim = 300;
# ////// VECTOR-DIMENSION END //////

# ////// TEST PARAMETER START //////
# //int op_k=5;
# //int op_k=3;
# //int op_k=1;

# //////For Anatomy dataset (distance | similarity)
# // double threshold_ds = 0.01; //0.99
# // double threshold_ds = 0.02; //0.98
# // double threshold_ds = 0.03; //0.97
# // double threshold_ds = 0.04; //0.96
# // double threshold_ds = 0.05; //0.95
# // double threshold_ds = 0.06; //0.94
# // double threshold_ds = 0.07; //0.93
# // double threshold_ds = 0.08; //0.92
# // double threshold_ds = 0.09; //0.91
# // double threshold_ds = 0.10; //0.90
# // double threshold_ds = 0.15; //0.85
# // double threshold_ds = 0.20; //0.80
# // double threshold_ds = 0.25; //0.75
# // double threshold_ds = 0.30; //0.70
# // double threshold_ds = 0.35; //0.65
# // double threshold_ds = 0.40; //0.60
# // double threshold_ds = 0.45; //0.55
# // double threshold_ds = 0.50; //0.50
# // double threshold_ds = 0.55; //0.45
# // double threshold_ds = 0.60; //0.40
# //double threshold_ds = 0.65; //0.35
# // double threshold_ds = 0.70; //0.30
# // double threshold_ds = 0.75; //0.25
# // double threshold_ds = 0.80; //0.20
# // double threshold_ds = 0.85; //0.15
# // double threshold_ds = 0.90; //0.10
# // double threshold_ds = 0.95; //0.05
# // double threshold_ds = 1.00; //0.00
# ////// TEST PARAMETER START //////


if __name__=="__main__":
  # data_param = getDataParam()
  db_param = {
      'db_nm' : 'Anatomy',
      "vec_dim": 200,
      'op_k' : 1,
      'word_wt_ds' : 0.5,
      'meta_wt_ds' : 0.5,
      'threshold_ds' : 0.02,
      "sim_ind": "cosine", #euclidean
      "op_fl": ""
  }
  # ontoEval(db_param)
  op_ks=[1,3,5]
  threshold_dss=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
  for op_k in op_ks:
    for threshold_ds in threshold_dss:
      db_param["op_k"] = op_k
      db_param["threshold_ds"] = threshold_ds
      sim=int((1-threshold_ds)*100)
      fl_nm=str(op_k) + "_" + str(sim) + "_output_final.json"
      db_param["op_fl"] = DATA_DIR+'/output/' + fl_nm
      ontoEvalMS(word_sim_ind_1, db_param)
