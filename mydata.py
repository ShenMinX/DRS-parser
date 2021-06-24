import preprocess
import torch
import torch.utils.data as data
import re

from tokenizers import BertWordPieceTokenizer

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

    def __init__(self, sents, targets, word_to_ix, sense_to_ix, fragment_to_ix, itergration_to_ix, tokenizer, device, content_frg_idx, sents2 = None, targets2 = None): 
        'Initialization'
        self.sents = sents
        self.sents2 = sents2
        self.targets = targets
        self.targets2 = targets2
        self.primary_size = len(sents)
        self.content_frg_idx = content_frg_idx
        #self.word_to_ix = word_to_ix
        self.sense_to_ix = sense_to_ix
        self.fragment_to_ix = fragment_to_ix
        self.itergration_to_ix = itergration_to_ix
        self.tokenizer = tokenizer
        self.device = device

    
    def __len__(self):
        if self.sents2 == None:
            return len(self.sents)
        else:
            return len(self.sents) + len(self.sents2)

    def __getitem__(self, index):
        
        if index >= self.primary_size:
            sent = self.sents2[index-self.primary_size]
            target = self.targets2[index-self.primary_size]
        else:
            sent = self.sents[index]
            target = self.targets[index]

        target_s, target_f, traget_i = list(map(lambda x: preprocess.tokens_to_ixs(x[0], x[1]),[(
                self.sense_to_ix, [t[0] for t in target]), (
                    self.fragment_to_ix, [t[1] for t in target]), (
                        self.itergration_to_ix, [t[2] for t in target])]))
                        
        input_ids, token_type_ids, attention_mask, valid_indices = valid_tokenizing(sent, self.tokenizer, self.device)

        return (input_ids, token_type_ids, attention_mask, valid_indices, target_s, target_f, traget_i, sent)

def my_collate(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    input_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    attention_mask = [item[2] for item in batch]
    valid_indices = [item[3] for item in batch]
    target_s = [torch.LongTensor(item[4]).to(device) for item in batch]
    target_f = [torch.LongTensor(item[5]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[6]).to(device) for item in batch]
    sent = [item[7] for item in batch]

    return [input_ids, token_type_ids, attention_mask, valid_indices, target_s, target_f, target_i, sent]

if __name__ == "__main__":

    a,b,c,d,e,f = preprocess.encode2()
    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt")
    my_data = Dataset(e,f, a.token_to_ix, b.token_to_ix, c.token_to_ix, d.token_to_ix, tokenizer)
    loader = data.DataLoader(dataset=my_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    
    for idx, item in enumerate(loader):
        input_ids, token_type_ids, attention_mask, valid_indices, sense, frg, inter = [i for i in item]
        for sids, tids, mask, vids, ss, f, i in zip(input_ids, token_type_ids, attention_mask, valid_indices, sense, frg, inter):
            print(sids.shape, tids.shape, mask.shape, vids.shape, ss.shape, f.shape, i.shape)

        print("-------------------------")

    


