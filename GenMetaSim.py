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
        'src_fl_nm': DATA_DIR+'gnnentity/source_gnn_meta.json',
        'trgt_fl_nm': DATA_DIR+'gnnentity/target_gnn_meta.json',
        'op_fl_nm': DATA_DIR+'output/meta_sim/meta_sim'
    }
    return conf
#######
def getDataParam():
    data_param_fl_nm = code_path + data_param_json
    with open(data_param_fl_nm) as f:
        data_param = json.load(f)
    return data_param
########
def makeDir():
  fl_path=DATA_DIR+'output/meta_sim'
  os.makedirs(fl_path, exist_ok=True)
###########
def loadDict(conf):
    with open(code_path+conf['dict_fl_nm']) as f:
        embed_json_data = json.load(f)

    return embed_json_data
###########
def loadSrcTrgt(conf):
    with open(code_path+conf['src_fl_nm']) as f:
        src_data = json.load(f)

    with open(code_path+conf['trgt_fl_nm']) as f:
        trgt_data = json.load(f)

    return src_data, trgt_data
##############
def populateSrcNpVec(src_data):
    src_vec = []
    src_key_arr = []
    for src_key in src_data.keys():
        src_key_arr.append(src_key)
        src_vec.append(src_data[src_key]["vector"])

    src_vec = np.asarray(src_vec)

    return src_key_arr, src_vec
###########
def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    cos-dist = 1 - cos(@) // @ is angle between the two vectors
    @ = 0, cost-dist = 1 - 1 = 0
    @ = 90, cost-dist = 1 - 0 = 1
    @ = 180, cost-dist = 1 - (-1) = 2
    @ = 270, cost-dist = 1 - 0 = 1
    @ = 360, cost-dist = 1 - 1 = 0
    more the distance less the similarity
    """
    # print(type(matrix))
    # print(matrix.shape) #(3306, 300)
    # print(type(vector))
    # print(vector.shape) #(1, 300)
    return scipy.spatial.distance.cdist(matrix, vector, 'cosine').reshape(-1)
#############
def wordSim_Cosine(target_data, src_np_vec , source_keys):

  op_copy_cosine = OrderedDict()

  for trgt_key in target_data:  # For each target data
    op_copy_cosine_tmp = OrderedDict()
    trgt_vec = target_data[trgt_key]["vector"]
    trgt_vec = np.asarray(trgt_vec)
    trgt_vec = trgt_vec.reshape(1, trgt_vec.shape[0]) #making (300,) to (1,300)
    result = cos_cdist(src_np_vec, trgt_vec) #cosine distances between matrix(src), vector(target)
    # get the indices of the five(k) smallest values, # Indices not sorted
    tmp_inds = np.argpartition(result, top_k)[:top_k]
    ## Indices sorted by value from smallest to largest
    top_inds = tmp_inds[np.argsort(result[tmp_inds])]

    for top_ind in top_inds:
      src_key = source_keys[top_ind]
      cosine_val = result[top_ind]

      # print(1 - 0.9999999999999997)
      # print(1 - 0.9999999999999998)
      # print(1 - 0.9999999999999999)
      cosine_lst = [1.1102230246251565e-16, 2.220446049250313e-16, 3.3306690738754696e-16]
      if(cosine_val in cosine_lst):
          cosine_val = 0.0

      op_copy_cosine_tmp[src_key] = cosine_val

    op_copy_cosine[trgt_key] = op_copy_cosine_tmp

  return op_copy_cosine
############
def wordSim_Euclidean(target_data, src_np_vec , source_keys):

  op_copy_euclidean = OrderedDict()

  for trgt_key in target_data:  # For each target data
    op_copy_euclidean_tmp = OrderedDict()
    trgt_vec = target_data[trgt_key]["vector"]
    trgt_vec = np.asarray(trgt_vec)
    trgt_vec = trgt_vec.reshape(1, trgt_vec.shape[0]) #making (300,) to (1,300)
    result = ec_cdist(src_np_vec, trgt_vec) #euclidean distances between matrix(src), vector(target)
    # get the indices of the five(k) smallest values, # Indices not sorted
    tmp_inds = np.argpartition(result, top_k)[:top_k]
    ## Indices sorted by value from smallest to largest
    top_inds = tmp_inds[np.argsort(result[tmp_inds])]

    for top_ind in top_inds:
      src_key = source_keys[top_ind]
      euclidean_val = result[top_ind]
      op_copy_euclidean_tmp[src_key] = euclidean_val

    op_copy_euclidean[trgt_key] = op_copy_euclidean_tmp

  return op_copy_euclidean
#############
def ec_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    more the distance less the similarity
    """
    # print(type(matrix))
    # print(matrix.shape) #(3306, 300)
    # print(type(vector))
    # print(vector.shape) #(1, 300)
    return scipy.spatial.distance.cdist(matrix, vector, 'euclidean').reshape(-1)
############
def saveOp(op_meta, op_fl_nm):
    with open(code_path+op_fl_nm, 'w') as outfile:
        json.dump(op_meta, outfile, indent=4)
###########
#################### Main Code START ####################
def genMetaSim(word_sim_ind):
    try:
        print("#################### GenMetaSim START ####################")
        strt_tm = datetime.datetime.now()
        conf = assignVar()
        makeDir()
        if (word_sim_ind == word_sim_ind_1): ### cosine
            src_data, trgt_data = loadSrcTrgt(conf)
            source_keys, src_np_vec = populateSrcNpVec(src_data)
            op_copy_cosine = wordSim_Cosine(trgt_data, src_np_vec, source_keys)
            saveOp(op_copy_cosine, conf["op_fl_nm"] + '_cosine.json')
        elif (word_sim_ind == word_sim_ind_2): ### euclidean
            src_data, trgt_data = loadSrcTrgt(conf)
            source_keys, src_np_vec = populateSrcNpVec(src_data)
            op_copy_euclidean = wordSim_Euclidean(trgt_data, src_np_vec, source_keys)
            saveOp(op_copy_euclidean, conf["op_fl_nm"] + '_euclidean.json')

        time.sleep(wait_time)
    except Exception as exp:
        raise exp
    finally:
        end_tm = datetime.datetime.now()
        total_time_taken = end_tm - strt_tm
        print("Total time taken :- " + str(total_time_taken))
        print("#################### GenMetaSim FINISH ####################")

#################### Main Code END ####################
if __name__=="__main__":
  data_param = getDataParam()
  data_param = data_param["db"]
  genMetaSim(data_param["sim_ind"])
