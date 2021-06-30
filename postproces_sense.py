import re
import torch
import preprocess
import nltk
#nltk.download('wordnet')

from difflib import get_close_matches
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
POS_PATTERN = re.compile(r"\[(?P<pos>[a,v,n,r])\]")
SENSE_NUMBER_PATTERN = re.compile(r"\[(?P<number>\d\d)\]")


def make_ss_label(concept:str,pos:str,psum:str):
    return {"work": [concept],"\""+ pos +".00"+"\"": ["\""+pos+"."+psum+"\""]}


def num_first(all_senses: list, word: str, concept: str, psum: str, pos:str):
    candidates = [ss.split(".")[0] for ss in all_senses if ss.split(".")[-1]==psum]
    lemma = LEMMATIZER.lemmatize(word, pos=pos).lower()
    if candidates:
        matches = get_close_matches(concept, candidates, cutoff=0.1)
        matches_l = get_close_matches(lemma, candidates, cutoff=0.1)
        if matches:
            return make_ss_label(matches[0], pos, psum)
        elif matches_l:
            return make_ss_label(matches_l[0], pos, psum)
        # else:
        #     return make_ss_label(candidates[0], pos, psum) # maybe not optimal
    else:
        return concept_first(all_senses, word, concept, psum, pos)


def concept_first(all_senses: list, word: str, concept: str, psum: str, pos:str):
    candidates = [ss.split(".")[0] for ss in all_senses]
    matches = get_close_matches(concept, candidates, cutoff=0.1)
    if matches:
        psum = all_senses[candidates.index(matches[0])].split(".")[-1]
        return make_ss_label(matches[0], pos, psum)

    return "[ILLFORM]" 


def choose_sense(sense_char: list, word:str, pos:str):
    all_senses =  [ss.name() for ss in wn.synsets(word) if ss.name().split(".")[-2]==pos]
    psnum = None
    concept_list =[]
    for c in sense_char:
        if bool(re.match(POS_PATTERN, c)):
            ppos = c[1]
        elif bool(re.match(SENSE_NUMBER_PATTERN, c)):
            psnum = c[1]+c[2]
        else:
            concept_list.append(c)
    concept = "".join(concept_list)
    if all_senses:
        if psnum:
            return num_first(all_senses, word, concept, psnum, pos)
        elif concept_list:
            return concept_first(all_senses, word, concept, psnum, pos)

    return "[ILLFORM]"

def get_ws_simple(word:str, ss_chars:list, frg:list):
    if ss_chars:
        for cl in frg:
            if cl[1]=="work":
                if cl[2] == '"n.00"':
                    pos = "n"
                elif cl[2] == '"v.00"':
                    pos = "v"
                elif cl[2] == '"r.00"':
                    pos = "r"
                elif cl[2] == '"a.00"':
                    pos = "a"
                return choose_sense(ss_chars, word, pos)

    return "[ILLFORM]"


    # def get_ws_nltk(word:str, is_prpn: bool, is_content: bool, sense_chars:list, frg:list):
    # if is_prpn:
    #     if word !="":
    #         return {"\"tom\"": ["\""+word.lower()+"\""]}
    #     else:
    #         return "[ILLFORM]"
            
    # elif not is_prpn and is_content:
    #     for cl in frg:
    #         if cl[2] == '"n.00"':
    #             frg_pos = "n"
    #         elif cl[2] == '"v.00"':
    #             frg_pos = "v"
    #         elif cl[2] == '"r.00"':
    #             frg_pos = "r"
    #         elif cl[2] == '"a.00"':
    #             frg_pos = "a"
    #     GOLD_SENSE_PATTERN = re.compile('^[a-z]+\.'+frg_pos+'\.\d\d$')
    #     concept = []
    #     pos = ""
    #     ss_num = ""
    #     #all_senses =  [re.sub(r"\.s\.", ".a.", ss.name()) for ss in wn.synsets(word)]
    #     all_senses =  [ss.name() for ss in wn.synsets(word)]
    #     gold_senses = [ss for ss in all_senses if GOLD_SENSE_PATTERN.match(ss)]
    #     for c in sense_chars:
    #         if bool(re.match(POS_PATTERN, c)):
    #             pos = c[1]
    #         elif bool(re.match(SENSE_NUMBER_PATTERN, c)):
    #             ss_num = c[1]+c[2]
    #         elif pos == "" and ss_num == "" and not bool(re.match(POS_PATTERN, c)) and not bool(re.match(SENSE_NUMBER_PATTERN, c)):
    #             concept.append(c)

    #     prototype =  "".join(concept)+"."+pos+"."+ss_num
    #     if prototype in gold_senses:
    #         return {"work": ["".join(concept).lower()],"\""+ pos +".00"+"\"": ["\""+pos+"."+ss_num+"\""] }

    #     elif "".join(concept)!="" and (pos!="" or ss_num!=""):
    #         matches = get_close_matches(prototype, gold_senses, cutoff=0.1)
    #         if not matches:
    #             return "[ILLFORM]"
    #         else:
    #             c_p_n_list = matches[0].split(".")
    #             return {"work": [c_p_n_list[0]],"\""+ c_p_n_list[1] +".00"+"\"": ["\""+c_p_n_list[1]+"."+c_p_n_list[2]+"\""]}
    #     else:
    #         return "[ILLFORM]"
    # else:
    #     return {}