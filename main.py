import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

import re
import numpy as np

from tokenizers import BertWordPieceTokenizer
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

import mydata
import preprocess
from postprocess import decode, tuple_to_dictlist, tuple_to_list, tuple_to_iterlabels
from models import Linear_classifiers

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def my_collate(batch):

    input_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    attention_mask = [item[2] for item in batch]
    valid_indices = [item[3] for item in batch]
                
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

    target_s = [torch.LongTensor(item[4]).to(device) for item in batch]
    target_f = [torch.LongTensor(item[5]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[6]).to(device) for item in batch]
    sent = [item[7] for item in batch]

    return bert_input, valid_indices, target_s, target_f, target_i, sent


def average_word_emb(emb, valid):
    embs = []
    for i in range(len(valid)-1):
        embs.append(emb[torch.LongTensor([idx for idx in range(valid[i], valid[i+1])]).to(device)].mean(0).unsqueeze(0))
    return torch.cat(embs, 0)

if __name__ == '__main__':

    #train
    hyper_batch_size = 24

    learning_rate = 0.0015

    epochs = 15

    bert_embed_size = 768

    fine_tune  = True

    
    words, senses, fragment, integration_labels, tr_sents, tr_targets = preprocess.encode2(data_file = open('Data\\en\\gold\\train.txt', encoding = 'utf-8'))

    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)

    train_dataset = mydata.Dataset(tr_sents, tr_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device)


    #bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
    bert_model.config.output_hidden_states=True

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=hyper_batch_size, shuffle=False, collate_fn=my_collate)

    lossfunc = nn.CrossEntropyLoss()

    tagging_model = Linear_classifiers(
        embed_size = bert_embed_size, 
        syms_size = len(senses.token_to_ix), 
        frgs_size = len(fragment.token_to_ix), 
        intergs_size = len(integration_labels.token_to_ix),
        dropout_rate = 0.2
        ).to(device)

    optimizer = torch.optim.Adam(tagging_model.parameters(),lr=learning_rate)

    if fine_tune == True:
        bert_optimizer = AdamW(bert_model.parameters(), lr=1e-5)

    for e in range(epochs):
        total_loss = 0.0

        for idx, (bert_input, valid_indices, target_s, target_f, target_i, sent) in enumerate(train_loader):

            if fine_tune == False:
                with torch.no_grad():
                    bert_outputs = bert_model(**bert_input)

            else:
                bert_outputs = bert_model(**bert_input)
                
            embeddings = bert_outputs.hidden_states[7]

            mask = pad_sequence([torch.ones(len(s), dtype=torch.long).to(device) for s in sent], batch_first=True, padding_value=0)

            valid_embeds = [
                average_word_emb(embeds, valid)
                for embeds, valid in zip(embeddings, valid_indices)]

            # valid_embeds = [
            #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
            #     for embeds, valid in zip(embeddings, valid_indices)]
                
            batch_size = len(valid_embeds)

            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
            padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

            #print(padded_input.shape, padded_sense.shape,padded_frg.shape, padded_inter.shape )

            sense_out, frg_out, inter_out = tagging_model(padded_input)

            batch_loss = 0.0

            for i in range(padded_input.shape[1]): 
                sense_loss = lossfunc(sense_out[:,i,:]*mask[:, i].view(-1, 1), padded_sense[:,i]*mask[:, i])
                frg_loss = lossfunc(frg_out[:,i,:]*mask[:, i].view(-1, 1), padded_frg[:,i]*mask[:, i])
                inter_loss = lossfunc(inter_out[:,i,:]*mask[:, i].view(-1, 1), padded_inter[:,i]*mask[:, i])

                batch_loss = batch_loss + sense_loss + frg_loss + inter_loss

            optimizer.zero_grad()
            if fine_tune == True:
                bert_optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if fine_tune == True:
                bert_optimizer.step()

        with torch.no_grad():
            total_loss += float(batch_loss)

        print(e+1, ". total loss:", total_loss)

        e+=1



    #eval:

    with torch.no_grad():

        correct_s = 0
        correct_f = 0
        correct_i = 0
        n_of_t = 0
        count = 1
        _, _, _, _, te_sents, te_targets = preprocess.encode2(data_file = open('Data\\en\\gold\\dev.txt', encoding = 'utf-8'))
        pred_file = open('Data\\en\\gold\\prediction.clf', 'w', encoding="utf-8")

        test_dataset = mydata.Dataset(te_sents,te_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device)

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=hyper_batch_size, shuffle=False, collate_fn=my_collate)

        for idx, (bert_input, valid_indices, target_s, target_f, target_i, sent) in enumerate(test_loader):

            bert_outputs = bert_model(**bert_input)
            embeddings = bert_outputs.hidden_states[7]

            # valid_embeds = [
            #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
            #     for embeds, valid in zip(embeddings, valid_indices)]

            valid_embeds = [
                average_word_emb(embeds, valid)
                for embeds, valid in zip(embeddings, valid_indices)]

            batch_size = len(valid_embeds)

            seq_len = [len(s) for s in sent]

            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
            padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

            sense_out, frg_out, inter_out = tagging_model(padded_input)

            sense_max = torch.argmax(sense_out, 2)
            frg_max = torch.argmax(frg_out, 2)
            inter_max = torch.argmax(inter_out, 2)

            unpad_sense = [sense_max[i,:l].tolist() for i, l in enumerate(seq_len)]
            unpad_frg = [frg_max[i,:l].tolist() for i, l in enumerate(seq_len)]
            unpad_inter = [inter_max[i,:l].tolist() for i, l in enumerate(seq_len)]

            sense_pred = [preprocess.ixs_to_tokens(senses.ix_to_token, seq) for seq in unpad_sense]
            frg_pred = [preprocess.ixs_to_tokens(fragment.ix_to_token, seq) for seq in unpad_frg]
            inter_pred = [preprocess.ixs_to_tokens(integration_labels.ix_to_token, seq) for seq in unpad_inter]

            for ts, tf, ti, ps, pf, pi in zip(target_s, target_f, target_i, unpad_sense, unpad_frg, unpad_inter):
                for s_idx in range(len(ps)):
                    if ts[s_idx]==ps[s_idx]:
                        correct_s +=1
                    if tf[s_idx]==pf[s_idx]:
                        correct_f +=1
                    if ti[s_idx]==pi[s_idx]:
                        correct_i +=1
                n_of_t += ts.shape[0]

     

        #python counter.py -f1 prediction.clf -f2 dev.txt -prin -g clf_signature.yaml

            for sen, tar_s, tar_f, tar_i in zip(sent,sense_pred,frg_pred,inter_pred):
                #decode(sen[1: -1], [tuple_to_dictlist(t_s) for t_s in tar_s[1:-1]], [tuple_to_list(t_f) for t_f in tar_f[1:-1]], [tuple_to_iterlabels(t_i) for t_i in tar_i[1:-1]], i+1, pred_file)
                decode(sen, [tuple_to_dictlist(t_s) for t_s in tar_s], [tuple_to_list(t_f) for t_f in tar_f], [tuple_to_iterlabels(t_i) for t_i in tar_i], words.token_to_ix, count, pred_file)
                count+=1
        pred_file.close()

        print("Sense Accurancy: ", correct_s/n_of_t)
        print("Fragment Accurancy: ", correct_f/n_of_t)
        print("intergration label Accurancy: ", correct_i/n_of_t)







        
