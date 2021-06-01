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

def get_ws_nltk(word:str, is_prpn: bool, is_content: bool, sense_chars:list, frg:list):
    if is_prpn:
        if word !="":
            return {"\"tom\"": ["\""+word.lower()+"\""]}
        else:
            return "[ILLFORM]"
            
    elif not is_prpn and is_content:
        for cl in frg:
            if cl[2] == '"n.00"':
                frg_pos = "n"
            elif cl[2] == '"v.00"':
                frg_pos = "v"
            elif cl[2] == '"r.00"':
                frg_pos = "r"
            elif cl[2] == '"a.00"':
                frg_pos = "a"
        GOLD_SENSE_PATTERN = re.compile('^[a-z]+\.'+frg_pos+'\.\d\d$')
        concept = []
        pos = ""
        ss_num = ""
        all_senses =  [re.sub(r"\.s\.", ".a.", ss.name()) for ss in wn.synsets(word)]
        gold_senses = [ss for ss in all_senses if GOLD_SENSE_PATTERN.match(ss)]
        for c in sense_chars:
            if bool(re.match(POS_PATTERN, c)):
                pos = c[1]
            elif bool(re.match(SENSE_NUMBER_PATTERN, c)):
                ss_num = c[1]+c[2]
            elif pos == "" and ss_num == "" and not bool(re.match(POS_PATTERN, c)) and not bool(re.match(SENSE_NUMBER_PATTERN, c)):
                concept.append(c)

        prototype =  "".join(concept)+"."+pos+"."+ss_num
        if prototype in gold_senses:
            return {"work": ["".join(concept).lower()],"\""+ pos +".00"+"\"": ["\""+pos+"."+ss_num+"\""] }

        elif "".join(concept)!="" and (pos!="" or ss_num!=""):
            matches = get_close_matches(prototype, gold_senses)
            if not matches:
                return "[ILLFORM]"
            else:
                c_p_n_list = matches[0].split(".")
                return {"work": [c_p_n_list[0]],"\""+ c_p_n_list[1] +".00"+"\"": ["\""+c_p_n_list[1]+"."+c_p_n_list[2]+"\""]}
        else:
            return "[ILLFORM]"
    else:
        return {}
    

def get_ws_simple(word:str, is_prpn: bool, is_content: bool, sense_chars:list, frg:list):
    if is_prpn:
        if word !="":
            return {"\"tom\"": ["\""+word.lower()+"\""]}
        else:
            return "[ILLFORM]"
            
    elif not is_prpn and is_content:
        concept = []
        pos = ""
        ss_num = ""
        for c in sense_chars:
            if bool(re.match(POS_PATTERN, c)):
                pos = c[1]
            elif bool(re.match(SENSE_NUMBER_PATTERN, c)):
                ss_num = c[1]+c[2]
            elif pos == "" and ss_num == "" and not bool(re.match(POS_PATTERN, c)) and not bool(re.match(SENSE_NUMBER_PATTERN, c)) and len(c)==1:
                concept.append(c)

        if "".join(concept)!="" and pos!="" and ss_num!="":
            return {"work": ["".join(concept).lower()],"\""+ pos +".00"+"\"": ["\""+pos+"."+ss_num+"\""] }     
        else:
            return "[ILLFORM]"
    else:
        return {}