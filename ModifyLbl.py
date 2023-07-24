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



##########
def assignVar():
    conf_1 = {
        'src_ip_fl_nm': DATA_DIR+'ip/source.json',
        'trgt_ip_fl_nm': DATA_DIR+'ip/target.json',
        'src_op_fl_nm': DATA_DIR+'modifylbl/source.json',
        'trgt_op_fl_nm': DATA_DIR+'modifylbl/target.json',
        'removed_utl_fl_nm': DATA_DIR+'util/removed.txt'
    }

    conf_arr = []
    conf_arr.append(conf_1)

    return conf_arr


conf_arr = assignVar()

############

def getDataParam():
    data_param_fl_nm = code_path + data_param_json
    with open(data_param_fl_nm) as f:
        data_param = json.load(f)
    return data_param
    
##############

def loadSourceTarget(conf):
    source_fl_nm = conf['src_ip_fl_nm']
    target_fl_nm = conf['trgt_ip_fl_nm']

    source_fl_nm = code_path + source_fl_nm
    with open(source_fl_nm) as f:
        source_data = json.load(f)

    target_fl_nm = code_path + target_fl_nm
    with open(target_fl_nm) as f:
        target_data = json.load(f)

    return source_data, target_data
    
###############

def modfWord(word):
    word = word.lower()
    wordnet_lemmatizer = WordNetLemmatizer()
    word = wordnet_lemmatizer.lemmatize(word)

    return word


def removeStopWords(word):
    stop_word_lst = ["of", "the", "system", "a", "all", "at", "or", "and", "to", "with"]
    if (word.lower() in stop_word_lst):
        return ""
    else:
        return word


def getEntityWords(entity):
    return_words = []
    entity_words = entity.replace("'", "").replace('_', ' ').replace('-', ' ').replace('/', ' ').replace('(',
                                                                                                         ' ').replace(
        ')', ' ').split()
    wordnet_lemmatizer = WordNetLemmatizer()
    for idx, w in enumerate(entity_words):
        word = entity_words[idx]
        word = word.lower()
        word = wordnet_lemmatizer.lemmatize(word)
        word = removeStopWords(word)
        if (word != ""):
            return_words.append(word)

    return return_words


def checkIfRomanNumeral(intVal):
    intVal_Tmp = intVal.upper()
    validRomanNumerals = ["M", "D", "C", "L", "X", "V", "I"]
    for letters in intVal_Tmp:
        if letters not in validRomanNumerals:
            return False

    return True


def chkAlphaNumeric(input):
    return bool(re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', input))


def int_to_roman(input):
    intVal_Tmp = input
    intVal_Tmp = intVal_Tmp.upper()
    nums = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    sum = 0
    for i in range(len(intVal_Tmp)):
        try:
            value = nums[intVal_Tmp[i]]
            # If the next place holds a larger number, this value is negative
            if (i + 1 < len(intVal_Tmp)) and (nums[intVal_Tmp[i + 1]] > value):
                sum -= value
            else:
                sum += value
        except KeyError:
            raise (ValueError, 'input is not a valid Roman numeral: %s' % intVal_Tmp)

    return sum


def crtAltLbl(conf, data, removed_fl):
    err_key = []

    for key in data.keys():
        alt_word = ""

        if (data[key]['lbl'] is not None):
            words = getEntityWords(data[key]['lbl'])  # get all the words separated
            for word in words:
                tmp_word = word

                # if the word is numeric
                if (tmp_word.isdigit()):
                    num = str(int(tmp_word))
                    alt_word = alt_word + " " + num
                    continue

                # if the word is alpha-numeric
                if (chkAlphaNumeric(tmp_word)):
                    num = "".join(re.findall('\d+', tmp_word))

                    if (len(num) > 0):
                        num = str(int(num))
                        alt_word = alt_word + " " + num

                    abbr = tmp_word.replace(num, "").lower()
                    if (abbr == "s"):
                        alt_word = alt_word + " " + modfWord("Sacral")
                    elif (abbr == "l"):
                        alt_word = alt_word + " " + modfWord("Lumbar")
                    elif (abbr == "t"):
                        alt_word = alt_word + " " + modfWord("Thoracic")
                    elif (abbr == "c"):
                        alt_word = alt_word + " " + modfWord("Cervical")
                    elif (abbr == "ca"):
                        alt_word = alt_word + " " + modfWord("Cornu") + " " + modfWord("Ammonis")
                    else:  # CD4,CD8
                        alt_word = alt_word + " " + modfWord(abbr)
                    continue

                # if the word is Roman Integer (alpha)
                if (tmp_word.isalpha() and checkIfRomanNumeral(tmp_word)):
                    roman_num = str(int_to_roman(tmp_word))
                    if (roman_num.isdigit()):
                        alt_word = alt_word + " " + roman_num
                    continue

                # if the word is simple Literal alpha)
                if (tmp_word.isalpha() and (len(tmp_word) >= 1)):
                    alt_word = alt_word + " " + tmp_word
                else:
                    removed_fl.write(tmp_word + '\n')  # these words are removed while modifying labels

        else:
            print("label is None" + key)
            err_key.append(key)

        alt_word = ' '.join(sorted(set(alt_word.split())))  # to remove repeat words
        data[key]['altLbl'] = alt_word.strip()

    for key in err_key:
        data.pop(key, None)
        removed_fl.write(key + ' :label is None \n')  # these labels are none

    return data
    
    
##############
def crtAltLblUtl(conf, source_data, target_data, data_src_nm):

    removed_fl = open(conf['removed_utl_fl_nm'], "w")
    source_data = crtAltLbl(conf, source_data, removed_fl)
    target_data = crtAltLbl(conf, target_data, removed_fl)
    removed_fl.close()

    return source_data, target_data
    
#############
def saveSrcTrgt(source_data, target_data, conf):
    with open(code_path + conf['src_op_fl_nm'], 'w') as outfile:
        json.dump(source_data, outfile, indent=4)

    with open(code_path + conf['trgt_op_fl_nm'], 'w') as outfile:
        json.dump(target_data, outfile, indent=4)
#############
def modifyLblMain(data_param):
    print("#################### ModifyLabel START ####################")
    try:
        conf_arr = assignVar()
        for conf in conf_arr:
            source_data, target_data = loadSourceTarget(conf)

            source_data, target_data = crtAltLblUtl(conf, source_data, target_data, data_param['db_nm'])

            saveSrcTrgt(source_data, target_data, conf)

            time.sleep(wait_time)

    except Exception as exp:
        raise exp
    finally:
        print("#################### ModifyLabel FINISH ####################")
    
###########
### MAIN FUNCTION
if __name__=="__main__":
  data_param = getDataParam()
  modifyLblMain(data_param['db'])
