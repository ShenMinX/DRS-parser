import preprocess
import torch
import torch.utils.data as data
import re

import numpy as np

from torch.nn.utils.rnn import pad_sequence

from tokenizers import BertWordPieceTokenizer

def padding(seq, max_len): # outdated
    sen_pad = np.pad(seq,(0,max(0, max_len - len(seq))),'constant', constant_values = (0))[:max_len]
    return sen_pad


def _valid_wordpiece_indexes(sent, wp_sent): 
    
    marker = ["[CLS]", "[SEP]"]
    valid_idxs = []
    missing_chars = ""
    idx = 0

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
        
    return valid_idxs

def valid_tokenizing(sent, tokenizer, device):

    tokenized_sequence = tokenizer.encode(" ".join(sent))
    valid_idx = []

    input_ids = torch.LongTensor(tokenized_sequence.ids).to(device)
    token_type_ids = torch.LongTensor(tokenized_sequence.type_ids).to(device)
    attention_mask = torch.LongTensor(tokenized_sequence.attention_mask).to(device)

    valid_idx = _valid_wordpiece_indexes(sent, tokenized_sequence.tokens)
    valid_indices = torch.LongTensor(valid_idx).to(device)

    return input_ids, token_type_ids, attention_mask, valid_indices

class Dataset(data.Dataset):

    def __init__(self, sents, char_sents, targets, target_senses, max_sense_lens, word_to_ix, \
        char_to_ix, fragment_to_ix, itergration_to_ix, tokenizer, device): 
        'Initialization'
        self.sents = sents
        self.char_sents = char_sents
        self.targets = targets
        self.target_senses = target_senses
        self.max_sense_lens = max_sense_lens

        self.char_to_ix = char_to_ix
        self.fragment_to_ix = fragment_to_ix
        self.itergration_to_ix = itergration_to_ix
        self.tokenizer = tokenizer
        self.device = device

    
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        
        sent = self.sents[index]
        target = self.targets[index]

        words_len = preprocess.get_words_len(sent)

        break_token_idx = []
        stack = 0
        for ix in words_len[1:-1]:
            stack += ix
            break_token_idx.append(stack)
        
        target_s = [preprocess.tokens_to_ixs(self.char_to_ix, sense) for sense in self.target_senses[index]]

        char_sent, target_f, traget_i = list(map(lambda x: preprocess.tokens_to_ixs(x[0], x[1]),[(
            self.char_to_ix, self.char_sents[index]),(
                self.fragment_to_ix, [t[0] for t in target]), (
                    self.itergration_to_ix, [t[1] for t in target])]))
                        
        input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, self.tokenizer, self.device)

        return (input_ids, token_type_ids, attention_mask, valid_indices, char_sent, target_s, target_f, traget_i, words_len, break_token_idx, self.max_sense_lens[index])

    
def my_collate(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    input_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    attention_mask = [item[2] for item in batch]
    valid_indices = [item[3] for item in batch]
                
    valid_indices = pad_sequence(valid_indices, batch_first=True, padding_value=0.0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

    char_sent = [torch.LongTensor(item[4]).to(device) for item in batch]

    max_sense_len_batch = max([item[10] for item in batch])
    target_s = []
    for item in batch:
        sense_seq = []
        for sense in item[5]:
            sense_seq.append(padding(sense, max_sense_len_batch))
        target_s.append(torch.LongTensor(sense_seq).to(device))

    target_f = [torch.LongTensor(item[6]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[7]).to(device) for item in batch]

    words_lens = [torch.LongTensor(item[8]).to(device) for item in batch]

    break_token_idx = [torch.LongTensor(item[9]).to(device) for item in batch]

    return bert_input, valid_indices, char_sent, target_s, target_f, target_i, words_lens, break_token_idx

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    words, sent_char, fragment, integration_labels, sents, char_sents, targets, target_senses, max_sense_lens = preprocess.encode2()
    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt")

    my_data = Dataset(sents,char_sents,targets,target_senses, max_sense_lens, words.token_to_ix, sent_char.token_to_ix,\
         fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device)
    
    loader = data.DataLoader(dataset=my_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    
    for idx, (bert_input, valid_indices, char_sent, target_s, target_f, target_i, words_lens, break_token_idx) in enumerate(loader):
        print(target_s)

        print("-------------------------")

    


