import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data


from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

import mydata
import preprocess
from postprocess import decode, tuple_to_dictlist, tuple_to_list, tuple_to_iterlabels
from error_eval import ana_metrics

from models import Linear_classifiers

torch.manual_seed(33)

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
    orgn_sent = [item[7] for item in batch]

    return bert_input, valid_indices, target_s, target_f, target_i, orgn_sent

# def my_collate(batch): #version that excludes unaligned sentences

#     input_ids = []
#     token_type_ids = []
#     attention_mask = []
#     valid_indices = []
#     target_s = []
#     target_f = []
#     target_i = []
#     sent = []

#     for item in batch: 
#         if len(item[3])==len(item[7])+1:
#             input_ids.append(item[0])
#             token_type_ids.append(item[1])
#             attention_mask.append(item[2])
#             valid_indices.append(item[3])
#             target_s.append(torch.LongTensor(item[4]).to(device))
#             target_f.append(torch.LongTensor(item[5]).to(device))
#             target_i.append(torch.LongTensor(item[6]).to(device))
#             sent.append(item[7])
                
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
#     token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
#     attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

#     bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}

#     return bert_input, valid_indices, target_s, target_f, target_i, sent


def average_word_emb(emb, valid):
    embs = []
    for i in range(len(valid)-1):
        embs.append(emb[torch.LongTensor([idx for idx in range(valid[i], valid[i+1])]).to(device)].mean(0).unsqueeze(0))
    return torch.cat(embs, 0)

if __name__ == '__main__':

    #train
    lang = "de"

    train = False

    save_checkpoint = False

    hyper_batch_size = 16

    num_warmup_steps = 0

    learning_rate = 0.00015

    epochs = 10
    middle_epoch = 5
    old_epoch = 0
    if epochs < middle_epoch:
        middle_epoch = epochs 

    bert_embed_size = 768

    fine_tune  = True

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    
    # for en, de:
    words, senses, fragment, integration_labels, tr_sents, tr_targets, content_frg_idx, orgn_sents, sents2, targets2 = preprocess.encode2(primary_file ='Data\\'+lang+'\\gold\\train.txt', optional_file='Data\\'+lang+'\\silver\\train.txt', optional_file2='Data\\'+lang+'\\bronze\\train.txt', language=lang)
    # for it, nl:
    #words, senses, fragment, integration_labels, tr_sents, tr_targets, content_frg_idx, orgn_sents, sents2, targets2 = preprocess.encode2(primary_file ='Data\\'+lang+'\\silver\\train.txt', optional_file='Data\\'+lang+'\\bronze\\train.txt', optional_file2=None, language=lang)
    
    bert_models = {"en": "bert-base-cased","nl": "Geotrend/bert-base-nl-cased", "de": "dbmdz/bert-base-german-cased", "it": "dbmdz/bert-base-italian-cased"}

    model_path = 'Data\\'+lang+'\\model_paremeters.pth' # path of checkpoint of finetuned model

    model_name = bert_models[lang]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = mydata.Dataset(tr_sents, tr_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device, content_frg_idx, orgn_sents, sents2, targets2)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    primary_indices, optional_indices = indices[:train_dataset.primary_size], indices

    primary_sampler = data.sampler.SubsetRandomSampler(primary_indices)
    if len(train_dataset)>train_dataset.primary_size:
        optional_sampler = data.sampler.SubsetRandomSampler(optional_indices)


    bert_model = AutoModel.from_pretrained(model_name).to(device)

    bert_model.config.output_hidden_states=True

    
    primary_loader = data.DataLoader(dataset=train_dataset, batch_size=hyper_batch_size, sampler=primary_sampler, shuffle=False, collate_fn=my_collate)
    if len(train_dataset)>train_dataset.primary_size:
        optional_loader = data.DataLoader(dataset=train_dataset, batch_size=hyper_batch_size, sampler=optional_sampler, shuffle=False, collate_fn=my_collate)

    lossfunc = nn.CrossEntropyLoss()

    ############ Rebalancing classes in loss function for empty labels ############################## 
    weighted_label = integration_labels.token_to_ix[preprocess.dictlist_to_tuple({"b": [], "e": [], "n": [], "p": [], "s": [], "t": [], "x": []})]
    label_base = (torch.tensor(list(range(len(integration_labels.token_to_ix))))!=weighted_label).type(torch.float32).to(device)
    loss_weight = torch.where(label_base==0, torch.tensor(0.5, dtype=torch.float32).to(device), label_base)
    lossfunc2 = nn.CrossEntropyLoss(weight=loss_weight)

    weighted_sense = senses.token_to_ix[preprocess.dictlist_to_tuple({})]
    sense_base = (torch.tensor(list(range(len(senses.token_to_ix))))!=weighted_sense).type(torch.float32).to(device)
    loss_weight_s = torch.where(sense_base==0, torch.tensor(0.5, dtype=torch.float32).to(device), sense_base)
    lossfunc3 = nn.CrossEntropyLoss(weight=loss_weight_s)
    ##################################################################################################

    tagging_model = Linear_classifiers(
        embed_size = bert_embed_size, 
        syms_size = len(senses.token_to_ix), 
        frgs_size = len(fragment.token_to_ix), 
        intergs_size = len(integration_labels.token_to_ix),
        dropout_rate = 0.2
        ).to(device)

    optimizer = torch.optim.Adam(tagging_model.parameters(),lr=learning_rate)

    bert_optimizer = AdamW(bert_model.parameters(), lr=learning_rate)

    ################ Load Checkpoints ###############################################################
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if fine_tune == True:
            bert_model.load_state_dict(checkpoint['bert_model_state_dict'])
            bert_optimizer.load_state_dict(checkpoint['bert_optimizer_state_dict'])
        tagging_model.load_state_dict(checkpoint['tagging_model_state_dict'])       
        optimizer.load_state_dict(checkpoint['tagging_optimizer_state_dict'])
        old_epoch = checkpoint['epoch']

    
    except FileNotFoundError:
        train = True
    #################################################################################################

    if train:
        scheduler1 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, len(train_dataset)*middle_epoch)
        scheduler3 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, train_dataset.primary_size*(epochs - middle_epoch))
        if fine_tune == True:
            scheduler2 = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps, len(train_dataset)*middle_epoch)
            scheduler4 = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps, train_dataset.primary_size*(epochs - middle_epoch))

        with torch.no_grad():
            content_set = torch.LongTensor(list(train_dataset.content_frg_idx)).to(device)

        for e in range(epochs):
            total_loss = 0.0
            if e >= middle_epoch or len(train_dataset)==train_dataset.primary_size or epochs==middle_epoch:
                train_loader = primary_loader
            else:
                train_loader = optional_loader

            for idx, (bert_input, valid_indices, target_s, target_f, target_i, og_sents) in enumerate(train_loader):

                if fine_tune == False:
                    with torch.no_grad():
                        bert_outputs = bert_model(**bert_input)

                else:
                    bert_outputs = bert_model(**bert_input)
                    
                embeddings = bert_outputs.hidden_states[7]

                mask = pad_sequence([torch.ones(t.shape[0], dtype=torch.long).to(device) for t in target_s], batch_first=True, padding_value=0)

                ###  average wordpiece embedding
                valid_embeds = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings, valid_indices)]

                ###  initial wordpiece embedding
                # valid_embeds = [
                #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
                #     for embeds, valid in zip(embeddings, valid_indices)]
                    
                batch_size = len(valid_embeds)

                padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
                padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
                padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
                padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)



                sense_out, frg_out, inter_out = tagging_model(padded_input)

                batch_loss = 0.0

                for i in range(padded_input.shape[1]): 
                    sense_loss = lossfunc3(sense_out[:,i,:]*mask[:, i].view(-1, 1), padded_sense[:,i]*mask[:, i])
                    frg_loss = lossfunc(frg_out[:,i,:]*mask[:, i].view(-1, 1), padded_frg[:,i]*mask[:, i])
                    inter_loss = lossfunc2(inter_out[:,i,:]*mask[:, i].view(-1, 1), padded_inter[:,i]*mask[:, i])

                    batch_loss = batch_loss + sense_loss + frg_loss + inter_loss

                optimizer.zero_grad()
                if fine_tune == True:
                    bert_optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if e <middle_epoch:
                    scheduler1.step()
                else:
                    scheduler3.step()
                if fine_tune == True:
                    bert_optimizer.step()
                    if e <middle_epoch:
                        scheduler2.step()
                    else:
                        scheduler4.step()

            with torch.no_grad():
                total_loss += float(batch_loss)

            print(e+old_epoch+1, ". total loss:", total_loss)

            e+=1

        if save_checkpoint:
            torch.save({
            'epoch': old_epoch+e+1,
            'bert_model_state_dict': bert_model.state_dict(),
            'tagging_model_state_dict': tagging_model.state_dict(),
            'bert_optimizer_state_dict': bert_optimizer.state_dict(),
            'tagging_optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

    end.record()
    torch.cuda.synchronize()
    print("Train time: ",start.elapsed_time(end))  

    #eval:
    in_files = ['Data\\'+lang+'\\gold\\dev.txt', 'Data\\'+lang+'\\gold\\test.txt']
    out_files = ['Data\\'+lang+'\\gold\\prediction_dev.txt', 'Data\\'+lang+'\\gold\\prediction_test.txt']
    out_files2 = ['Data\\'+lang+'\\gold\\sen_prpty_dev.txt', 'Data\\'+lang+'\\gold\\sen_prpty_test.txt']
    for in_f, out_f, out_f2 in zip(in_files, out_files, out_files2):
        with torch.no_grad():

            print("Encoder Parameters:",sum([param.nelement() for param in bert_model.parameters()]))
            print("Decoder Parameters:",sum([param.nelement() for param in tagging_model.parameters()]))

            correct_s = 0
            correct_f = 0
            correct_i = 0
            n_of_t = 0
            count = 1
            _, _, _, _, te_sents, te_targets, _, orgn_sents, _, _ = preprocess.encode2(primary_file = in_f, language = lang)
            pred_file = open( out_f, 'w', encoding="utf-8")
            sen_prpty_file = open( out_f2, 'w', encoding="utf-8")

            test_dataset = mydata.Dataset(te_sents,te_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device, content_frg_idx, orgn_sents)

            test_loader = data.DataLoader(dataset=test_dataset, batch_size=hyper_batch_size, shuffle=False, collate_fn=my_collate)

            for idx, (bert_input, valid_indices, target_s, target_f, target_i, og_sents) in enumerate(test_loader):

                bert_outputs = bert_model(**bert_input)
                embeddings = bert_outputs.hidden_states[7]

                # valid_embeds = [
                #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
                #     for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings, valid_indices)]

                batch_size = len(valid_embeds)

                seq_len = [t.shape[0] for t in target_s]

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

        
            #cd DRS_parsing_3\evaluation
            #python counter.py -f1 prediction_dev.txt -f2 dev.txt -prin -ms_file result_dev.txt -g clf_signature.yaml
            #python counter.py -f1 prediction_test.txt -f2 test.txt -prin -ms_file result_test.txt -g clf_signature.yaml

                for sen, tar_s, tar_f, tar_i in zip(og_sents,sense_pred,frg_pred,inter_pred):
                    ana_clauses = decode(sen, [tuple_to_dictlist(t_s) for t_s in tar_s], [tuple_to_list(t_f) for t_f in tar_f], [tuple_to_iterlabels(t_i) for t_i in tar_i], words.token_to_ix, count, pred_file, lang)
                    #sen_prpty_file.write(str(count)+"\t"+str(len(sen))+"\n") # for relation between number of words and fscore
                    sen_prpty_file.write(ana_metrics(ana_clauses, count))  # for eval fscore of drs contains certain clause element                 
                    count+=1
            pred_file.close()
            sen_prpty_file.close()

            print("Sense Accurancy: ", correct_s/n_of_t)
            print("Fragment Accurancy: ", correct_f/n_of_t)
            print("intergration label Accurancy: ", correct_i/n_of_t)







        
