import re
import torch
import preprocess
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from difflib import get_close_matches

LEMMATIZER = WordNetLemmatizer()
POS_PATTERN = re.compile(r"\[(?P<pos>[a,v,n,r])\]")
SENSE_NUMBER_PATTERN = re.compile(r"\[(?P<number>\d\d)\]")

def get_ws_nltk(word:str, is_prpn: bool, is_content: bool, sense_chars:list):
    if is_prpn:
        return {"\"tom\"": "\""+word.lower()+"\""}
    elif is_content:
        concept = []
        pos = ""
        ss_num = ""
        gold_senses =  [re.sub(r"\.s\.", ".a.", ss.name()) for ss in wn.synsets(word)]
        for c in sense_chars:
            if bool(re.match(POS_PATTERN, c)):
                pos = c[1]
            elif bool(re.match(SENSE_NUMBER_PATTERN, c)):
                ss_num = c[1]+c[2]
            elif pos == "" and ss_num == "" and not bool(re.match(POS_PATTERN, c)) and not bool(re.match(SENSE_NUMBER_PATTERN, c)):
                concept.append(c)

        prototype =  "".join(concept)+"."+pos+"."+ss_num
        if prototype in gold_senses:
            return {"work": "".join(concept).lower(),"\""+ pos +".00"+"\"": "\""+pos+"."+ss_num+"\"" }
        elif (not not concept) and (pos!="" or ss_num!=""):
            c_p_n_list = get_close_matches(prototype, gold_senses)[0].split(".")
            if not c_p_n_list:
                return {"work": c_p_n_list[0],"\""+ c_p_n_list[1] +".00"+"\"": "\""+c_p_n_list[1]+"."+c_p_n_list[2]+"\"" }
            else:
                "[ILLFORM]"
        else:
            return "[ILLFORM]"
    else:
        return {}
    

def get_ws_simple(word:str, is_prpname:bool,sense_chars:list):
    pass