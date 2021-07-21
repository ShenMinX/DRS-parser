import torch
import clf
import re
import string
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel



def _valid_wordpiece_indexes(sent, wp_sent): 
    
    marker = ["[CLS]", "[SEP]", "[UNK]"]
    valid_idxs = []
    missing_chars = ""
    idx = 0
    assert wp_sent[-1]=="[SEP]"
    try:
        for wp_idx, wp in enumerate(wp_sent,0):
            if not wp in marker:
                if sent[idx].startswith(wp) and missing_chars == "":
                    valid_idxs.append(wp_idx)

                if missing_chars == "":
                    missing_chars = sent[idx][len(wp.replace("##","")):]
                else:
                    missing_chars = missing_chars[len(wp.replace("##","")):]
            
                if missing_chars == "":
                    idx+=1
            elif wp =="[UNK]":
                if missing_chars =="":
                    valid_idxs.append(wp_idx)
                    idx+=1
    except IndexError:
        print(sent)
        print(wp_sent)
        
    return valid_idxs+[len(wp_sent)-1]

def valid_tokenizing(sent, tokenizer, device):

    tokenized_sequence = tokenizer.encode(" ".join(sent))
    valid_idx = []

    input_ids = torch.LongTensor(tokenized_sequence.ids).to(device)
    token_type_ids = torch.LongTensor(tokenized_sequence.type_ids).to(device)
    attention_mask = torch.LongTensor(tokenized_sequence.attention_mask).to(device)

    valid_idx = _valid_wordpiece_indexes(sent, tokenized_sequence.tokens)
    valid_indices = torch.LongTensor(valid_idx).to(device)

    return input_ids, token_type_ids, attention_mask, valid_indices

def encode(lang:str, quality:str, unk_chars:set):
    alnum = re.compile(r'[^a-zA-Z0-9.]')
    data_file = open('Data\\'+lang+'\\'+quality+'\\train.txt',encoding = 'utf-8')
    for i, (sentence, fragments, unaligned) in enumerate(clf.read(data_file), start=1):
        for ch in "".join(sentence):
            if ch not in string.punctuation and bool(alnum.search(ch)):
                unk_chars.add(ch)
    print(i)
    return unk_chars

    

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = "bert-base-cased"
    #tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = BertWordPieceTokenizer("Data\\de\\bert-base-german-dbmdz-cased-vocab.txt", lowercase=False)
    
    sent = '" ẽ " is a ẽ.ẽ side ẽẽ letter in ẽ ẽ the the？ ẽ Guarani alphabet？'.split(" ")
    trag = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22]

    tokenized_sequence = tokenizer(" ".join(sent), add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)
    wp = ["[CLS]"]+tokenizer.tokenize(" ".join(sent))+["[SEP]"]

    idx = _valid_wordpiece_indexes(sent, wp)
    
    trag = trag + [len(wp)-1]

    print(sent)
    print(wp)
    print(f'pred:{idx}')
    print(f'trag:{trag}')
    print(len(idx))
    print(len(sent)+1)
    language = ["en","de", "it", "nl"]
    quality =["bronze", "silver"]
    unk_chars = set()
    for lang in language:
        unk_chars_lang = set()
        for qua in quality:
            unk_chars = encode(lang, qua, unk_chars)
            unk_chars_lang = encode(lang, qua, unk_chars_lang)
        with open('Data\\'+lang+'\\all_unk.txt','w', encoding="utf-8") as f1:
            for idx, el in enumerate(unk_chars_lang):
                f1.write(str(idx)+"\t"+el+"\n")
            f1.close()
    with open('Data\\all_unk.txt','w', encoding="utf-8") as f2:
        for idx, el in enumerate(unk_chars):
            f2.write(str(idx)+"\t"+el+"\n")
        f2.close()

    # bert_model = AutoModel.from_pretrained(model_name).to(device)
    # #bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
    # bert_model.config.output_hidden_states=True


    # input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, tokenizer, device)
    # print(input_ids)
    # print(valid_indices)
    # print(len(sent))