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
from models import Linear_classifiers

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    target_s = [torch.LongTensor(item[4]).to(device) for item in batch]
    target_f = [torch.LongTensor(item[5]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[6]).to(device) for item in batch]

    return bert_input, valid_indices, target_s, target_f, target_i

if __name__ == '__main__':

    #train

    learning_rate = 0.0015

    epochs = 1

    bert_embed_size = 768

    fine_tune  = False

    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 33
    
    words, senses, fragment, integration_labels, tr_sents, tr_targets = preprocess.encode2(data_file = open('Data\\toy\\dev.txt', encoding = 'utf-8'))

    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)

    dataset = mydata.Dataset(tr_sents,tr_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = data.sampler.SubsetRandomSampler(val_indices)

    #bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

    train_loader = data.DataLoader(dataset=dataset, batch_size=48, sampler=train_sampler, shuffle=False, collate_fn=my_collate)

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

        for idx, (bert_input, valid_indices, target_s, target_f, target_i) in enumerate(train_loader):

            if fine_tune == False:
                with torch.no_grad():
                    bert_outputs = bert_model(**bert_input)

            else:
                bert_outputs = bert_model(**bert_input)
                
            embeddings = bert_outputs.last_hidden_state

            valid_embeds = [
                embeds[torch.nonzero(valid).squeeze(1)]
                for embeds, valid in zip(embeddings, valid_indices)]
                
            batch_size = len(valid_embeds)

            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
            padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

            #print(padded_input.shape, padded_sense.shape,padded_frg.shape, padded_inter.shape )

            sense_out, frg_out, inter_out = tagging_model(padded_input)

            batch_loss = 0.0
            max_length = padded_input.shape[1]

            for i in range(padded_input.shape[1]): 
                sense_loss = lossfunc(sense_out[:,i,:], padded_sense[:,i])
                frg_loss = lossfunc(frg_out[:,i,:], padded_frg[:,i])
                inter_loss = lossfunc(inter_out[:,i,:], padded_inter[:,i])

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

        print(e, ". total loss:", total_loss)

        e+=1



    #eval:

    with torch.no_grad():

        correct = 0
        n_of_t = 0

        test_loader = data.DataLoader(dataset=dataset, batch_size=48, sampler=valid_sampler, shuffle=False, collate_fn=my_collate)

        for idx, (bert_input, valid_indices, target_s, target_f, target_i) in enumerate(test_loader):

            bert_outputs = bert_model(**bert_input)
            embeddings = bert_outputs.last_hidden_state

            valid_embeds = [
                embeds[torch.nonzero(valid).squeeze(1)]
                for embeds, valid in zip(embeddings, valid_indices)]

            batch_size = len(valid_embeds)

            seq_len = [s.shape[0] for s in target_s]

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

            for ts, ps in zip(target_s, unpad_sense):
                for s_idx in range(len(ps)):
                    if ts[s_idx]==ps[s_idx]:
                        correct +=1
                n_of_t += ts.shape[0]

    print("Accurancy: ", correct/n_of_t)






        
