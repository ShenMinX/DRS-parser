import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

import re

from tokenizers import BertWordPieceTokenizer
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

import mydata
import preprocess
from models import Linear_classifiers


if __name__ == '__main__':

    #train

    learning_rate = 0.0015

    epochs = 15

    bert_embed_size = 768

    fine_tune  = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    words, senses, clauses, integration_labels, tr_sents, tr_targets = preprocess.encode2()
    _, _, _, _, te_sents, te_targets = preprocess.encode2()

    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)

    train_set = mydata.Dataset(tr_sents,tr_targets, words.token_to_ix, senses.token_to_ix, clauses.token_to_ix, integration_labels.token_to_ix, tokenizer, device)

    #bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

    loader = data.DataLoader(dataset=train_set, batch_size=48, shuffle=False, collate_fn=mydata.my_collate)

    lossfunc = nn.CrossEntropyLoss()

    tagging_model = Linear_classifiers(
        embed_size = bert_embed_size, 
        syms_size = len(senses.token_to_ix), 
        frgs_size = len(clauses.token_to_ix), 
        intergs_size = len(integration_labels.token_to_ix),
        dropout_rate = 0.2
        ).to(device)

    optimizer = torch.optim.Adam(tagging_model.parameters(),lr=learning_rate)

    if fine_tune == True:
        bert_optimizer = AdamW(bert_model.parameters(), lr=1e-5)

    for e in range(epochs):
        total_loss = 0.0

        for idx, item in enumerate(loader):
            input_ids, token_type_ids, attention_mask, valid_indices, sense_batch, frg_batch, inter_batch = [i for i in item]

            if fine_tune == False:
                with torch.no_grad():
                    valid_indices = pad_sequence(valid_indices, batch_first=True, padding_value=0.0)
                    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
                    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
                    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

                    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

                    bert_outputs = bert_model(**bert_input)
                    embeddings = bert_outputs.last_hidden_state

                    valid_embeds = [
                        embeds[torch.nonzero(valid).squeeze(1)]
                        for embeds, valid in zip(embeddings, valid_indices)]

            else:
                valid_indices = pad_sequence(valid_indices, batch_first=True, padding_value=0.0)
                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
                token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
                attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

                bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}
                bert_outputs = bert_model(**bert_input)
                embeddings = bert_outputs.last_hidden_state

                valid_embeds = [
                    embeds[torch.nonzero(valid).squeeze(1)]
                    for embeds, valid in zip(embeddings, valid_indices)]
                
            batch_size = len(valid_embeds)

            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
            padded_sense = pad_sequence(sense_batch, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(frg_batch, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(inter_batch, batch_first=True, padding_value=0.0)

            #print(padded_input.shape, padded_sense.shape,padded_frg.shape, padded_inter.shape )

            sense_pred, frg_pred, inter_pred = tagging_model(padded_input)

            batch_loss = 0.0
            max_length = padded_input.shape[1]

            for i in range(padded_input.shape[1]): 
                sense_loss = lossfunc(sense_pred[:,i,:], padded_sense[:,i])
                frg_loss = lossfunc(frg_pred[:,i,:], padded_frg[:,i])
                inter_loss = lossfunc(frg_pred[:,i,:], padded_inter[:,i])

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

        
