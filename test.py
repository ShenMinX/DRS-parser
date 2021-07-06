import torch
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel



def _valid_wordpiece_indexes(sent, wp_sent): 
    
    marker = ["[CLS]", "[SEP]"]
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

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = "dbmdz/bert-base-german-cased"
    #tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer = BertWordPieceTokenizer("Data\\de\\bert-base-german-dbmdz-cased-vocab.txt", lowercase=False)
    sent = ['Ich', 'hole', 'dich', 'um', '2.30', 'Uhr', 'ceegtebvtv','ab', '.']
    tokenized_sequence = tokenizer(" ".join(sent), add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)
    wp = tokenizer.tokenize(" ".join(sent))

    idx = _valid_wordpiece_indexes(sent, ["[CLS]"]+wp+["[SEP]"])
    print(wp)
    print(idx)

    bert_model = AutoModel.from_pretrained(model_name).to(device)
    #bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
    bert_model.config.output_hidden_states=True


    # input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, tokenizer, device)
    # print(input_ids)
    # print(valid_indices)
    # print(len(sent))