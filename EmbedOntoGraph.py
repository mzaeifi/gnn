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
        "conf_arr": [
            {
                'ind' : 'source',
                'ent_fl_nm': DATA_DIR+'gnnentity/source_gnn.json',
                'intr_ent_fl_nm': DATA_DIR+'gnnentity/source_gnn_tmp.json', #intermediate file with timestamp ~ source_gnn_tmp.json
                'graph_fl_path': DATA_DIR+'gnnentity/entity_graph/',
                'op_embed_fl_nm': DATA_DIR+'gnnentity/source_gnn_meta.json'
            },
            {
                'ind' : 'target',
                'ent_fl_nm': DATA_DIR+'gnnentity/target_gnn.json',
                'intr_ent_fl_nm': DATA_DIR+'gnnentity/target_gnn_tmp.json', #intermediate file with timestamp ~ target_gnn_tmp.json
                'graph_fl_path': DATA_DIR+'gnnentity/entity_graph/',
                'op_embed_fl_nm': DATA_DIR+'gnnentity/target_gnn_meta.json'
            }
        ]
    }

    return conf
###############
def getDataParam():
    data_param_fl_nm = code_path + data_param_json
    with open(data_param_fl_nm) as f:
        data_param = json.load(f)
    return data_param
###############

def makeDir():
  fl_path=DATA_DIR+'gnnentity/entity_graph/'
  os.makedirs(fl_path, exist_ok=True)
  ###############
def loadFile(fl_nm):
    with open(fl_nm) as f:
        data = json.load(f)
    return data
##################
def getRawConfig(init_path, graph_path, epochs, dim, lr_rate):
  raw_config = dict(
    # I/O data
    init_path=init_path,
    entity_path=graph_path,
    edge_paths=[
        graph_path + 'edges_partitioned/',
    ],
    # Graph structure
    entities={
        "node": {"num_partitions": 1}
    },
    relations=[
        {
            "name": "self",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        },
        {
            "name": "parent",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        },
        {
            "name": "child",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        },
        {
            "name": "equivalent",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        },
        {
            "name": "disjoint",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        },
        {
            "name": "restriction",
            "lhs": "node",
            "rhs": "node",
            "operator": "complex_diagonal",
            "weight": 1.0
        }
    ],
    dynamic_relations=False,
    dimension=dim,  # output vector dimension of each node
    global_emb=False,
    comparator="dot",
    checkpoint_path=graph_path + 'chkpt/',
    checkpoint_preservation_interval=100,
    num_epochs=epochs,
    num_uniform_negs=1000,
    loss_fn="ranking",
    lr=lr_rate,
    regularization_coef=0.001,
    eval_fraction=0.,
    verbose=0,
  )

  return raw_config
##################
def populatePrimaryTrainFl(conf_val, data_param):
  graph_path=conf_val["graph_fl_path"]

  graph_fl_nm=graph_path+'node_edge.tsv'
  raw_config=getRawConfig("", graph_path, data_param["init_epoch"], data_param["vec_dim"], data_param["learning_rate"])

  setup_logging()
  config = parse_config(raw_config)

  subprocess_init = SubprocessInitializer()

  input_edge_paths = [Path(graph_fl_nm)]

  convert_input_data(
      config.entities,
      config.relations,
      config.entity_path,
      config.edge_paths,
      input_edge_paths,
      TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
      dynamic_relations=config.dynamic_relations,
  )

  train(config, subprocess_init=subprocess_init)

  return raw_config
########################
def addPreEmbedding(conf_val, NUMBER_OF_EPOCHS, raw_config, embeddings_dict):
  graph_path=conf_val["graph_fl_path"]
  nodes_path = graph_path + 'entity_names_node_0.json'
  nodes_emb_path = graph_path + "chkpt/" + "embeddings_node_0.v{NUMBER_OF_EPOCHS}.h5".format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

  with open(nodes_path,'r') as source:
    nodes = json.load(source)
  dist = {item:ind for ind,item in enumerate(nodes)}

  with h5py.File(nodes_emb_path,'r+') as source:
      for node in embeddings_dict:
        info=embeddings_dict[node]
        if node in nodes:
            source['embeddings'][dist[node]] = info['vector']

########################
def populateNewEmbedding(conf_val, data_param):
  graph_path=conf_val["graph_fl_path"]
  graph_fl_nm=graph_path+'node_edge.tsv'
  raw_config=getRawConfig(graph_path+"chkpt/", graph_path, data_param["total_epoch"], data_param["vec_dim"], data_param["learning_rate"])
  setup_logging()
  config = parse_config(raw_config)
  subprocess_init = SubprocessInitializer()
  input_edge_paths = [Path(graph_fl_nm)]

  convert_input_data(
      config.entities,
      config.relations,
      config.entity_path,
      config.edge_paths,
      input_edge_paths,
      TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
      dynamic_relations=config.dynamic_relations,
  )

  train(config, subprocess_init=subprocess_init)

  return raw_config
########################
def getNewEmbedding(conf_val, NUMBER_OF_EPOCHS, raw_config, entity_obj):
  nodes_path = conf_val["graph_fl_path"] + 'entity_names_node_0.json'
  nodes_emb_path = conf_val["graph_fl_path"] + "chkpt/" + "embeddings_node_0.v{NUMBER_OF_EPOCHS}.h5".format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

  with open(nodes_path, 'r') as f:
      nodes = json.load(f)

  with h5py.File(nodes_emb_path, 'r') as g:
      nodes_embeddings = g['embeddings'][:]

  nodes2embedding = dict(zip(nodes, nodes_embeddings))

  entity_id=entity_obj['iri']

  return nodes2embedding[entity_id]
########################
def crtEdgeFl(conf_val, entity_obj):
  file_nm=conf_val['graph_fl_path']+'node_edge.tsv'
  with open(file_nm, 'w') as f:
    for edge in entity_obj['graphEdges']:
        f.write('\t'.join(edge) + '\n')
####################
def delEdgeFl(conf_val):
  fl_path=conf_val['graph_fl_path']
  shutil.rmtree(fl_path)
  makeDir()
######################
def trainNewEmbed(conf_val, entity_obj, data_param):

  crtEdgeFl(conf_val, entity_obj)

  raw_config=populatePrimaryTrainFl(conf_val, data_param)

  addPreEmbedding(conf_val, data_param["init_epoch"], raw_config, entity_obj)

  raw_config=populateNewEmbedding(conf_val, data_param)

  embed=getNewEmbedding(conf_val, data_param["total_epoch"], raw_config, entity_obj)
  delEdgeFl(conf_val)

  return embed
########################
def populateGNNEntityHelper(conf_val, entity_info, data_param):
  i=0
  if(conf_val['ind']=='source'):
    for src_entity in entity_info:
      entity_obj=entity_info[src_entity]
      if (entity_obj['metaVector']): #is not None then do nothing
        continue
      embed = trainNewEmbed(conf_val, entity_obj, data_param)
      entity_obj['metaVector'] = embed.tolist()
      entity_info[src_entity]=entity_obj
      if (i%data_param["save_intermediate_node"] == 0):
        saveNewEmbeddingIntermediate(entity_info, DATA_DIR+'gnnentity/source_gnn_meta') #intermediate save

      i=i+1
  elif(conf_val['ind']=='target'):
    for trgt_entity in entity_info:
      entity_obj=entity_info[trgt_entity]
      if (entity_obj['metaVector']): #is not None then do nothing
        continue
      embed = trainNewEmbed(conf_val, entity_obj, data_param)
      entity_obj['metaVector'] = embed.tolist()
      entity_info[trgt_entity]=entity_obj
      if (i%data_param["save_intermediate_node"] == 0):
        saveNewEmbeddingIntermediate(entity_info, DATA_DIR+'gnnentity/target_gnn_meta') #intermediate save

      i=i+1
  return entity_info
########################
def saveNewEmbeddingIntermediate(entity_info, fl):
    ct = datetime.datetime.now()
    ct = ct.strftime("%m_%d_%Y_%H_%M_%S")
    fl_nm = fl+ct+'.json'
    with open(code_path+fl_nm, 'w') as outfile:
        json.dump(entity_info, outfile, indent=4)
#########################
def saveNewEmbedding(entity_info, conf):
    with open(code_path+conf['op_embed_fl_nm'], 'w') as outfile:
        json.dump(entity_info, outfile, indent=4)
########################

#######################
def populateGNNEntity(data_param):
    try:
        print("#################### populateGNNEntity START ####################")

        conf = assignVar()
        #####Mean of vectors
        conf_arr = conf["conf_arr"]

        makeDir()
        print("lenght conff_arr", len(conf_arr))
        num =0
        for conf_val in conf_arr:

          if (data_param["prev_embed"]==0):
            fl_nm = code_path+conf_val['ent_fl_nm']

            entity_info = loadFile(fl_nm)
            num+=1
            print("itteration",num)

          else:
            fl_nm = code_path+conf_val['intr_ent_fl_nm']
            entity_info = loadFile(fl_nm)

          entity_info=populateGNNEntityHelper(conf_val, entity_info, data_param)
          saveNewEmbedding(entity_info, conf_val) #final save
          time.sleep(wait_time)


    except Exception as exp:
        raise exp
    finally:
        print("#################### populateGNNEntity FINISH ####################")
###############

# 'prev_embed': 0, start from scratch or anything else  load previous embedding
# 'init_epoch' will be always 1 for creating initial file creation
# 'total_epoch' total epoch of training for each entity/node

if __name__=="__main__":
  data_param = getDataParam()
  populateGNNEntity(data_param['model'])
