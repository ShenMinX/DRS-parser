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
    assert wp_sent[-1]=="[SEP]"

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
        
    return valid_idxs+[len(wp_sent)-1]

# def valid_tokenizing(sent, tokenizer, device):

#     tokenized_sequence = tokenizer.encode(" ".join(sent))
#     valid_idx = []

#     input_ids = torch.LongTensor(tokenized_sequence.ids).to(device)
#     token_type_ids = torch.LongTensor(tokenized_sequence.type_ids).to(device)
#     attention_mask = torch.LongTensor(tokenized_sequence.attention_mask).to(device)

#     valid_idx = _valid_wordpiece_indexes(sent, tokenized_sequence.tokens)
#     valid_indices = torch.LongTensor(valid_idx).to(device)

#     return input_ids, token_type_ids, attention_mask, valid_indices

def valid_tokenizing(sent, tokenizer, device):

    tokenized_sequence = tokenizer(" ".join(sent), add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True)

    input_ids = torch.LongTensor(tokenized_sequence['input_ids']).to(device)
    token_type_ids = torch.LongTensor(tokenized_sequence['token_type_ids']).to(device)
    attention_mask = torch.LongTensor(tokenized_sequence['attention_mask']).to(device)
    wp = tokenizer.tokenize(" ".join(sent))

    valid_idx = _valid_wordpiece_indexes(sent, ["[CLS]"]+wp+["[SEP]"])
    valid_indices = torch.LongTensor(valid_idx).to(device)


    try:
        assert len(sent)+1 == valid_indices.shape[0]
    except AssertionError:
        print(valid_indices)
        print(len(sent)+1)
        print(tokenized_sequence.words())
        print(" ".join(sent))
        print(tokenizer.tokenize(" ".join(sent)))


    return input_ids, token_type_ids, attention_mask, valid_indices

class Dataset(data.Dataset):

    def __init__(self, sents, char_sents, targets, target_senses, max_sense_lens, word_to_ix, \
        char_to_ix, fragment_to_ix, itergration_to_ix, content_frg_idx, prpname_frg_idx, tokenizer, device): 
        'Initialization'
        self.sents = sents
        self.char_sents = char_sents
        self.targets = targets
        self.target_senses = target_senses
        self.max_sense_lens = max_sense_lens

        self.content_frg_idx = content_frg_idx
        self.prpname_frg_idx = prpname_frg_idx

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

        char_sent = [preprocess.tokens_to_ixs(self.char_to_ix, char_words) for char_words in self.char_sents[index]]
        target_s = [preprocess.tokens_to_ixs(self.char_to_ix, sense) for sense in self.target_senses[index]]

        target_f, traget_i = list(map(lambda x: preprocess.tokens_to_ixs(x[0], x[1]),[(
                self.fragment_to_ix, [t[0] for t in target]), (
                    self.itergration_to_ix, [t[1] for t in target])]))
                        
        input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, self.tokenizer, self.device)

        return (input_ids, token_type_ids, attention_mask, valid_indices, char_sent, target_s, target_f, traget_i, words_len, self.max_sense_lens[index], sent)

    
def my_collate(batch):
    
    input_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    attention_mask = [item[2] for item in batch]
    valid_indices = [item[3] for item in batch]
                
    valid_indices = pad_sequence(valid_indices, batch_first=True, padding_value=0.0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

    words_lens = [torch.LongTensor(item[8]).to(device) for item in batch]

    max_word_len_batch = max([max(item[8]) for item in batch])
    max_sense_len_batch = max([item[9] for item in batch])

    sentences = [item[10] for item in batch]

    char_sent = []
    target_s = []
    for item in batch:
        char_seq = []
        sense_seq = []
        for word, sense in zip(item[4],item[5]):
            char_seq.append(padding(word, max_word_len_batch))
            sense_seq.append(padding(sense, max_sense_len_batch))
        char_sent.append(torch.LongTensor(char_seq).to(device))
        target_s.append(torch.LongTensor(sense_seq).to(device))
    
    char_sent_len = [s.shape[0] for s in char_sent]
    padded_char_input = pad_sequence(char_sent, batch_first=True, padding_value=0.0)

    target_f = [torch.LongTensor(item[6]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[7]).to(device) for item in batch]

    return bert_input, valid_indices, padded_char_input, target_s, target_f, target_i, words_lens, sentences


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    words, chars, fragments, integration_labels, content_frg_idx, prpname_frg_idx, sents, char_sents, targets, target_senses, max_sense_lens = preprocess.encode2()
    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt")

    my_data = Dataset(sents,char_sents,targets,target_senses, max_sense_lens, words.token_to_ix, chars.token_to_ix,\
         fragments.token_to_ix, integration_labels.token_to_ix, content_frg_idx, prpname_frg_idx, tokenizer, device)
    
    loader = data.DataLoader(dataset=my_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    
    for idx, (bert_input, valid_indices, padded_char_input, target_s, target_f, target_i, words_lens, sentences) in enumerate(loader):
        print(words_lens)

        print("-------------------------")

    


