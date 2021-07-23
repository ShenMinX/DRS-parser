import torch
import clf
import re
import drs
import string
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel



def find_unk(sent, tokenizer): 
    
    unk_char = set()
    for ch in "".join(sent).strip():
        try:
            if tokenizer.tokenize(ch)[0]=="[UNK]":
                unk_char.add(ch)
        except IndexError:
            unk_char.add(ch)

    return unk_char


def encode(lang:str, quality:str, unk_chars:set, quantity:set, tokenizer):

    data_file = open('Data\\'+lang+'\\'+quality+'\\train.txt',encoding = 'utf-8')
    for i, (sentence, fragments, _) in enumerate(clf.read(data_file), start=1):
        unk_chars = unk_chars.union(find_unk(sentence, tokenizer))
        quantity = quantity.union(find_quantity(sentence, fragments, quantity))
    return unk_chars, quantity

def find_quantity(sentence, fragments, quantity):
    for word, fragment in zip(sentence, fragments):
        for clause in fragment:
            if (clause[1] in ('Quantity', 'EQU') and not drs.is_constant(clause[3]) and not drs.is_ref(clause[3])):
                quantity.add((word, clause[3]))
    return quantity


def make_unk_list(tokenizer,lang, quality =["bronze", "silver"]):

    unk_chars_lang = set()
    quantity_lang = set()
    for qua in quality:
        unk_chars_lang, quantity_lang = encode(lang, qua, unk_chars_lang, quantity_lang, tokenizer)
    with open('Data\\'+lang+'\\all_unk.txt','w', encoding="utf-8") as f1:
        for idx, el in enumerate(unk_chars_lang):
            f1.write(str(idx)+"\t"+el+"\n")
        f1.close()
    with open('Data\\'+lang+'\\all_quantity.txt','w', encoding="utf-8") as f2:
        for idx, el in enumerate(quantity_lang):
            f2.write(str(idx)+"\t"+el[0]+"\t"+el[1]+"\n")
        f2.close()

 


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #en: "bert-base-cased"
    #nl: "Geotrend/bert-base-nl-cased"
    #de: "dbmdz/bert-base-german-cased"
    #it: "dbmdz/bert-base-italian-cased"

    languages = {"en": "bert-base-cased", "nl": "Geotrend/bert-base-nl-cased", "de": "dbmdz/bert-base-german-cased", "it": "dbmdz/bert-base-italian-cased"}
    for lang, model_name in languages.items():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        make_unk_list(tokenizer, lang)


    # sent = '" ẽ " is a ẽ.ẽ side ẽẽ letter in ẽ ẽ the the？ ẽ Guarani alphabet？'.split(" ")
    # trag = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22]

    # tokenized_sequence = tokenizer(" ".join(sent), add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)
    # wp = ["[CLS]"]+tokenizer.tokenize(" ".join(sent))+["[SEP]"]

    # idx = find_unk(sent, wp, tokenizer)
    
    # trag = trag + [len(wp)-1]

    # print(sent)
    # print(wp)
    # print(f'pred:{idx}')
    # print(f'trag:{trag}')
    # print(len(idx))
    # print(len(sent)+1)
        
    # bert_model = AutoModel.from_pretrained(model_name).to(device)
    # #bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
    # bert_model.config.output_hidden_states=True


    # input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, tokenizer, device)
    # print(input_ids)
    # print(valid_indices)
    # print(len(sent))