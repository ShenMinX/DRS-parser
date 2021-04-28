import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

import re
import random

from tokenizers import BertWordPieceTokenizer
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

import mydata
import preprocess
from models import Linear_classifiers, Encoder, Decoder

from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_char_context(valid_embeds, words_lens, break_token_idx):
    return [ve.repeat_interleave(wl, dim = 0).index_fill_(0, br, 0) for ve, wl, br in zip(valid_embeds, words_lens, break_token_idx)]


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

    char_sent = [torch.LongTensor(item[4]).to(device) for item in batch]
    char_sent_len = [s.shape[0] for s in char_sent]
    padded_char_input = pad_sequence(char_sent, batch_first=True, padding_value=0.0)

    max_sense_len_batch = max([item[10] for item in batch])
    target_s = []
    for item in batch:
        sense_seq = []
        for sense in item[5]:
            sense_seq.append(mydata.padding(sense, max_sense_len_batch))
        target_s.append(torch.LongTensor(sense_seq).to(device))

    target_f = [torch.LongTensor(item[6]).to(device) for item in batch]
    target_i = [torch.LongTensor(item[7]).to(device) for item in batch]

    words_lens = [torch.LongTensor(item[8]).to(device) for item in batch]

    break_token_idx = [torch.LongTensor(item[9]).to(device) for item in batch]

    return bert_input, valid_indices, padded_char_input, char_sent_len, target_s, target_f, target_i, words_lens, break_token_idx

if __name__ == '__main__':

    #train
    hyper_batch_size = 6

    learning_rate = 0.0015

    epochs = 10

    bert_embed_size = 768

    enc_embed_size = 200

    enc_hid_size = 300

    dec_embed_size = 200

    dec_hid_size = 300

    eps=1e-7

    fine_tune  = True
    
    words, chars, fragments, integration_labels, sents, char_sents, targets, target_senses, max_sense_lens = preprocess.encode2(data_file = open('Data\\mergedata\\gold\\gold.clf', encoding = 'utf-8'))

    tokenizer = BertWordPieceTokenizer("bert-base-cased-vocab.txt", lowercase=False)

    train_set = mydata.Dataset(sents, char_sents, targets, target_senses, max_sense_lens, words.token_to_ix, chars.token_to_ix,\
         fragments.token_to_ix, integration_labels.token_to_ix, tokenizer, device)

    #bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased').to(device)

    train_loader = data.DataLoader(dataset=train_set, batch_size=hyper_batch_size, shuffle=False, collate_fn=my_collate)

    lossfunc = nn.CrossEntropyLoss()

    tagging_model = Linear_classifiers(
        embed_size = bert_embed_size, 
        frgs_size = len(fragments.token_to_ix), 
        intergs_size = len(integration_labels.token_to_ix),
        dropout_rate = 0.2
        ).to(device)

    model_encoder = Encoder(
                  vocab=chars.token_to_ix, 
                  hidden_size=enc_hid_size, 
                  embed_size=enc_embed_size).to(device)

    model_decoder = Decoder(
                      vocab=chars.token_to_ix, 
                      encode_size=enc_hid_size*2 + 768, 
                      hidden_size=dec_hid_size, 
                      embed_size=dec_embed_size,
                      device=device
                     ).to(device)

    criterion = nn.NLLLoss()

    enc_optimizer = torch.optim.Adam(model_encoder.parameters(),lr=learning_rate)
    dec_optimizer = torch.optim.Adam(model_decoder.parameters(),lr=learning_rate)

    optimizer = torch.optim.Adam(tagging_model.parameters(),lr=learning_rate)

    if fine_tune == True:
        bert_optimizer = AdamW(bert_model.parameters(), lr=1e-5)

    for e in range(epochs):
        total_loss = 0.0

        for idx, (bert_input, valid_indices, padded_char_input, char_sent_len, target_s, target_f, target_i, words_lens, break_token_idx) in enumerate(train_loader):

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

            chars_contexts = get_char_context(valid_embeds, words_lens, break_token_idx)
            
            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)

            padded_chars_context = pad_sequence(chars_contexts, batch_first=True, padding_value=0.0)

            assert padded_chars_context.shape[0] == padded_char_input.shape[0]
            assert padded_chars_context.shape[1] == padded_char_input.shape[1]
            

            padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

            #print(padded_input.shape, padded_sense.shape,padded_frg.shape, padded_inter.shape )

            frg_out, inter_out = tagging_model(padded_input)

            batch_loss = 0.0
            max_length = padded_input.shape[1]
            max_tl = padded_sense.shape[1]

            max_sense_len = padded_sense.shape[2]

            for i in range(padded_input.shape[1]): 
                frg_loss = lossfunc(frg_out[:,i,:], padded_frg[:,i])
                inter_loss = lossfunc(inter_out[:,i,:], padded_inter[:,i])

                batch_loss = batch_loss + frg_loss + inter_loss
            
            enc_out, enc_hidden = model_encoder(padded_char_input, char_sent_len)

            expanded_enc_out = torch.cat((enc_out, padded_chars_context), 2)

            dec_input = torch.tensor([chars.token_to_ix["-BOS-"]]*batch_size, dtype=torch.long).to(device).view(batch_size, 1)

            with torch.no_grad():
                rnn_hid = (torch.zeros(batch_size,dec_hid_size).to(device),torch.zeros(batch_size,dec_hid_size).to(device))
            
            for i in range(max_tl):
                for j in range(max_sense_len):
                    output, rnn_hid = model_decoder(expanded_enc_out, rnn_hid, dec_input, padded_char_input) # batch x vocab_size
                
                    _, dec_pred = torch.max(output, 1) # batch_size vector

                    if random.randint(0, 11) > 5:          
                        dec_input = padded_sense[:,i,j].view(batch_size, 1)
                    else:
                        dec_input = dec_pred.view(batch_size, 1)

                    p_step_loss = criterion(torch.log(output + eps), padded_sense[:,i, j])

                    batch_loss = batch_loss + p_step_loss

            optimizer.zero_grad()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            if fine_tune == True:
                bert_optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            enc_optimizer.step()
            dec_optimizer.step()
            if fine_tune == True:
                bert_optimizer.step()

        with torch.no_grad():
            total_loss += float(batch_loss)

        print(e+1, ". total loss:", total_loss)

        e+=1



    #eval:

    _, _, _, _, te_sents, te_char_sents, te_targets, te_target_senses, te_max_sense_lens = preprocess.encode2(data_file = open('Data\\mergedata\\gold\\gold.clf', encoding = 'utf-8'))

    test_set = mydata.Dataset(te_sents, te_char_sents, te_targets, te_target_senses,te_max_sense_lens, words.token_to_ix, chars.token_to_ix,\
         fragments.token_to_ix, integration_labels.token_to_ix, tokenizer, device)

    with torch.no_grad():
        
        n_of_t = 0
        correct = 0

        test_loader = data.DataLoader(dataset=test_set, batch_size=hyper_batch_size, shuffle=False, collate_fn=my_collate)

        for idx, (bert_input, valid_indices, padded_char_input, char_sent_len, target_s, target_f, target_i, words_lens, break_token_idx) in enumerate(test_loader):



            bert_outputs = bert_model(**bert_input)
            embeddings = bert_outputs.last_hidden_state

            valid_embeds = [
                embeds[torch.nonzero(valid).squeeze(1)]
                for embeds, valid in zip(embeddings, valid_indices)]

            batch_size = len(valid_embeds)

            seq_len = [s.shape[0] for s in target_s]

            chars_contexts = get_char_context(valid_embeds, words_lens, break_token_idx)

            padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)

            padded_chars_context = pad_sequence(chars_contexts, batch_first=True, padding_value=0.0)

            padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
            padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
            padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

            frg_out, inter_out = tagging_model(padded_input)

            frg_max = torch.argmax(frg_out, 2)
            inter_max = torch.argmax(inter_out, 2)

            max_tl = padded_sense.shape[1]     # max target length

            max_sense_len = padded_sense.shape[2]

            enc_out, enc_hidden = model_encoder(padded_char_input, char_sent_len)

            expanded_enc_out = torch.cat((enc_out, padded_chars_context), 2)

            dec_input = torch.tensor([chars.token_to_ix["-BOS-"]]*batch_size, dtype=torch.long).to(device).view(batch_size, 1)

            rnn_hid = (torch.zeros(batch_size,dec_hid_size).to(device),torch.zeros(batch_size,dec_hid_size).to(device)) # default init_hidden_value
             
            for i in range(max_tl):
                #pred = torch.tensor([],dtype=torch.long).to(device)
                for j in range(max_sense_len):
    
                    output, rnn_hid = model_decoder(expanded_enc_out, rnn_hid, dec_input, padded_char_input) # batch x vocab_size
                
                    _, dec_pred = torch.max(output, 1) # batch_size vector

                    #pred = torch.cat([pred, dec_pred.view(batch_size, 1)], dim = 1)

                    dec_input = dec_pred.view(batch_size, 1)

                    for b in range(batch_size):
                        if padded_sense[b, i, j]==dec_pred[b] and dec_pred[b]!=chars.token_to_ix['[PAD]']:
                            correct +=1
                        if padded_sense[b, i, j]!=chars.token_to_ix['[PAD]']:
                            n_of_t +=1

                        
            
            # unpad for evaluation
            # for b in range(batch_size):
            #     final_pred = pred[b,:][pred[b,:]!=chars.token_to_ix['[PAD]']].tolist()

            #     final_preds.append(final_pred)

            unpad_frg = [frg_max[i,:l].tolist() for i, l in enumerate(seq_len)]
            unpad_inter = [inter_max[i,:l].tolist() for i, l in enumerate(seq_len)]

            frg_pred = [preprocess.ixs_to_tokens(fragments.ix_to_token, seq) for seq in unpad_frg]
            inter_pred = [preprocess.ixs_to_tokens(integration_labels.ix_to_token, seq) for seq in unpad_inter]
        
        # for p, t in zip(final_preds, te_target_senses):
        #     print(p, t)

        # n_of_t = 0
        # correct = 0
        # rouge_target = []
        # for predic, targ in zip(final_preds, te_target_senses):
        #     targ_list = preprocess.tokens_to_ixs(chars.token_to_ix, targ)
        #     rouge_target.append(targ_list)
        #     min_length = min(len(targ_list), len(predic)) 
        #     for i in range(min_length):
        #         if predic[i]==targ_list[i]:
        #             correct += 1
        #     n_of_t += len(targ_list)
        
        print("Accurancy: ", correct/n_of_t)


        # _, _, rouge_1 = rouge_n_summary_level(final_preds, rouge_target, 1)
        # print('ROUGE-1: %f' % rouge_1)

        # _, _, rouge_2 = rouge_n_summary_level(final_preds, rouge_target, 2)
        # print('ROUGE-2: %f' % rouge_2)
        
        # _, _, rouge_l = rouge_l_summary_level(final_preds, rouge_target) # extremely time consuming...
        # print('ROUGE-L: %f' % rouge_l)







        
