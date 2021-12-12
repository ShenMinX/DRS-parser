import os
import torch
import click
import json
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

############################################################################################################################################################################################################################################################
# Train: 
# python main.py train -l en -t -e 10 -fi -fd Data\\en\\gold\\dev.txt -f Data\\en\\gold\\test.txt -ft Data\\en\\gold\\train.txt
# python main.py train -l en -b 12 -t -e 10 -w 0 -lr 0.00015 -fi -s -fm Data\\en\\model_paremeters.pth -fd Data\\en\\gold\\dev.txt -f Data\\en\\gold\\test.txt -ft Data\\en\\gold\\train.txt -f3 Data\\en\\silver\\train.txt -f4 Data\\en\\bronze\\train.txt
# Test: 
# python main.py train -l en -fi -fm Data\\en\\model_paremeters.pth -fd Data\\en\\gold\\dev.txt -f Data\\en\\gold\\test.txt -ft Data\\en\\gold\\train.txt -f3 Data\\en\\silver\\train.txt -f4 Data\\en\\bronze\\train.txt
# python main.py test -l en -fm Data\\en\\model_paremeters.pth -fd Data\\en\\gold\\dev.txt -f Data\\en\\gold\\test.txt
############################################################################################################################################################################################################################################################

def save_labels(path, senses, fragments, integration_labels):
    marks = ["[PAD]","-EOS-","-BOS-","[UNK]"]
    labels = {}
    senses_token_to_ix = {}
    senses_ix_to_token = {}
    for s1, v1, v2, s2 in zip(senses.token_to_ix, senses.ix_to_token):
        if s1 not in marks:
            senses_token_to_ix[tuple_to_dictlist(s1)] = v1
        if s2 not in marks:
            senses_ix_to_token[v2] = tuple_to_dictlist(s2)
    labels["senses_token_to_ix"] = senses_token_to_ix
    labels["senses_ix_to_token"] = senses_ix_to_token

    fragments_token_to_ix = {}
    fragments_ix_to_token = {}
    for f1, v1, v2, f2 in zip(fragments.token_to_ix, fragments.ix_to_token):
        if f1 not in marks:
            fragments_token_to_ix[tuple_to_list(f1)] = v1
        if f2 not in marks:
            fragments_ix_to_token[v2] = tuple_to_list(f2)
    labels["fragments_token_to_ix"] = fragments_token_to_ix
    labels["fragments_ix_to_token"] = fragments_ix_to_token

    integration_labels_token_to_ix = {}
    integration_labels_ix_to_token = {}
    for i1, v1, v2, i2 in zip(integration_labels.token_to_ix, integration_labels.ix_to_token):
        if i1 not in marks:
            integration_labels_token_to_ix[tuple_to_dictlist(i1)] = v1
        if i2 not in marks:
            integration_labels_ix_to_token[v2] = tuple_to_dictlist(i2)
    labels["integration_labels_token_to_ix"] = integration_labels_token_to_ix
    labels["integration_labels_ix_to_token"] = integration_labels_ix_to_token

    out = json.dumps(labels)

    with open(path, 'w') as labels_dict:
        json.dump(out, labels_dict)

def load_labels(path):
    with open(path, 'r') as labels_dict:
        labels_in = json.load(labels_dict)
        labels = json.loads(labels_in)
        senses_token_to_ix = {}
        senses_ix_to_token = {}
        for s1, v1, v2, s2 in zip(labels["senses_token_to_ix"], labels["senses_ix_to_token"]):
            senses_token_to_ix[preprocess.dictlist_to_tuple(s1)] = v1
            senses_ix_to_token[v2] = preprocess.dictlist_to_tuple(s2)
        senses = preprocess.dictionary(senses_token_to_ix, senses_ix_to_token)

        fragments_token_to_ix = {}
        fragments_ix_to_token = {}
        for f1, v1, v2, f2 in zip(labels["fragments_token_to_ix"], labels["fragments_ix_to_token"]):
            fragments_token_to_ix[tuple(f1)] = v1
            fragments_ix_to_token[v2] = tuple(f2)
        fragments = preprocess.dictionary(fragments_token_to_ix, fragments_ix_to_token)

        integration_labels_token_to_ix = {}
        integration_labels_ix_to_token = {}
        for i1, v1, v2, i2 in zip(labels["integration_labels_token_to_ix"], labels["integration_labels_ix_to_token"]):
            integration_labels_token_to_ix[preprocess.dictlist_to_tuple(i1)] = v1
            integration_labels_ix_to_token[v2] = preprocess.dictlist_to_tuple(i2)
        integration_labels = preprocess.dictionary(integration_labels_token_to_ix, integration_labels_ix_to_token)
    
    return senses, fragments, integration_labels
    

@click.group()
def main():
    pass

@main.command()
@click.option("-l", "--language",type=click.Choice(['en', 'de','it', 'nl'], case_sensitive=False), help="en: English, de: German, it: Italian, nl: Dutch")
@click.option('-b','--batch', type=int, default=16, help="Batch size")
@click.option('-t','--train', is_flag=True, help='Train: true | Test: false')
@click.option('-e','--epoch', type=int, default=10, help="Training + Testing, or Test only")
@click.option('-w','--num_warmup_steps', type=int, default=0, help="Number of warm_up steps in training for linear scheduler")
@click.option('-lr','--learning_rate', type=float, default=0.00015, help="Learning rate of training")
@click.option('-fi','--finetuning', is_flag=True, help="Train BERT parameters or not")
@click.option('-s','--save_checkpoint', is_flag=True, help='True: save, False: no save')
@click.option('-fm','--model_file', type=click.Path(exists=True), required=False, help="Provide model file path, and choose whether to save model. ")
@click.option('-fd','--dev_file', type=click.Path(exists=True), help="Dev file path (optional)")
@click.option('-f','--test_file', type=click.Path(exists=True), required=False, default=None, help="Test file path")
@click.option('-ft','--train_file', type=click.Path(exists=True), help="Train file path")
@click.option('-f3','--train_file_op1', type=click.Path(exists=True), required=False, default=None, help="Optional 2nd train set path for train & finetuning")
@click.option('-f4','--train_file_op2', type=click.Path(exists=True), required=False, default= None, help="Optional 3rd train set path for train & finetuning")
def train(language, batch, train, epoch, num_warmup_steps, learning_rate, finetuning, save_checkpoint, model_file, dev_file, test_file, train_file, train_file_op1, train_file_op2):
    """Train and Test"""
    middle_epoch = epoch/2
    old_epoch = 0

    bert_embed_size = 768

    if train_file_op1==None and train_file_op2==None:
        middle_epoch = epoch
 
    f = open(dev_file)
    root_dir=os.path.realpath(f.name)
    f.close()
    in_files = [dev_file]
    out_files = [root_dir+'prediction_dev.txt']
    out_files2 = [root_dir+'sen_prpty_dev.txt', ]
    
    if dev_file != None:
        in_files.append(test_file)
        out_files.append(root_dir+'prediction_test.txt')
        out_files2.append(root_dir+'sen_prpty_test.txt')



    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    

    words, senses, fragment, integration_labels, tr_sents, tr_targets, orgn_sents, sents2, targets2 = preprocess.encode2(primary_file = train_file, optional_file = train_file_op1, optional_file2 = train_file_op2, language=language)
    if save_checkpoint:
        save_labels(root_dir+'label_dict.txt')
    with open(root_dir+'word_dict.txt', 'w') as word_dict:
        json.dump(words.__dict__, word_dict)
    # with open(root_dir+'senses_dict.txt', 'w') as senses_dict:
    #     json.dump(senses.__dict__, senses_dict)
    # with open(root_dir+'fragment_dict.txt', 'w') as fragment_dict:
    #     json.dump(fragment.__dict__, fragment_dict)
    # with open(root_dir+'integration_labels_dict.txt', 'w') as integration_labels_dict:
    #     json.dump(integration_labels.__dict__, integration_labels_dict)

    bert_models = {"en": "bert-base-cased","nl": "Geotrend/bert-base-nl-cased", "de": "dbmdz/bert-base-german-cased", "it": "dbmdz/bert-base-italian-cased"}

    model_name = bert_models[language]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = mydata.Dataset(tr_sents, tr_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device, orgn_sents, sents2, targets2)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    primary_indices, optional_indices = indices[:train_dataset.primary_size], indices

    primary_sampler = data.sampler.SubsetRandomSampler(primary_indices)
    if len(train_dataset)>train_dataset.primary_size:
        optional_sampler = data.sampler.SubsetRandomSampler(optional_indices)


    bert_model = AutoModel.from_pretrained(model_name).to(device)

    bert_model.config.output_hidden_states=True

    
    primary_loader = data.DataLoader(dataset=train_dataset, batch_size=batch, sampler=primary_sampler, shuffle=False, collate_fn=my_collate)
    if len(train_dataset)>train_dataset.primary_size:
        optional_loader = data.DataLoader(dataset=train_dataset, batch_size=batch, sampler=optional_sampler, shuffle=False, collate_fn=my_collate)

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
    if (train == True and save_checkpoint == True) or (train == False and save_checkpoint == False):
        try:
            checkpoint = torch.load(model_file, map_location=device)
            if finetuning == True:
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
        scheduler3 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, train_dataset.primary_size*(epoch - middle_epoch))
        if finetuning == True:
            scheduler2 = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps, len(train_dataset)*middle_epoch)
            scheduler4 = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps, train_dataset.primary_size*(epoch - middle_epoch))

        for e in range(epoch):
            total_loss = 0.0
            if e >= middle_epoch or len(train_dataset)==train_dataset.primary_size or epoch==middle_epoch:
                train_loader = primary_loader
            else:
                train_loader = optional_loader

            for idx, (bert_input, valid_indices, target_s, target_f, target_i, og_sents) in enumerate(train_loader):

                if finetuning == False:
                    with torch.no_grad():
                        bert_outputs = bert_model(**bert_input)

                else:
                    bert_outputs = bert_model(**bert_input)
                    
                embeddings = bert_outputs.hidden_states[7]

                embeddings2 = bert_outputs.last_hidden_state

                mask = pad_sequence([torch.ones(t.shape[0], dtype=torch.long).to(device) for t in target_s], batch_first=True, padding_value=0)

                ###  average wordpiece embedding
                valid_embeds = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds2 = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings2, valid_indices)]

                ###  initial wordpiece embedding
                # valid_embeds = [
                #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
                #     for embeds, valid in zip(embeddings, valid_indices)]
                    
                batch_size = len(valid_embeds)

                padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
                padded_input2 = pad_sequence(valid_embeds2, batch_first=True, padding_value=0.0)
                padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
                padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
                padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)



                sense_out, frg_out, inter_out = tagging_model(padded_input, padded_input2)

                batch_loss = 0.0

                for i in range(padded_input.shape[1]): 
                    sense_loss = lossfunc3(sense_out[:,i,:]*mask[:, i].view(-1, 1), padded_sense[:,i]*mask[:, i])
                    frg_loss = lossfunc(frg_out[:,i,:]*mask[:, i].view(-1, 1), padded_frg[:,i]*mask[:, i])
                    inter_loss = lossfunc2(inter_out[:,i,:]*mask[:, i].view(-1, 1), padded_inter[:,i]*mask[:, i])

                    batch_loss = batch_loss + sense_loss + frg_loss + inter_loss

                optimizer.zero_grad()
                if finetuning == True:
                    bert_optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if e <middle_epoch:
                    scheduler1.step()
                else:
                    scheduler3.step()
                if finetuning == True:
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
            }, model_file)

    end.record()
    torch.cuda.synchronize()
    print("Train time: ",start.elapsed_time(end))  

    #eval:
    for in_f, out_f, out_f2 in zip(in_files, out_files, out_files2):
        with torch.no_grad():

            print("Encoder Parameters:",sum([param.nelement() for param in bert_model.parameters()]))
            print("Decoder Parameters:",sum([param.nelement() for param in tagging_model.parameters()]))

            correct_s = 0
            correct_f = 0
            correct_i = 0
            n_of_t = 0
            count = 1
            _, _, _, _, te_sents, te_targets, orgn_sents, _, _ = preprocess.encode2(primary_file = in_f, language = language)
            pred_file = open( out_f, 'w', encoding="utf-8")
            sen_prpty_file = open( out_f2, 'w', encoding="utf-8")

            test_dataset = mydata.Dataset(te_sents,te_targets, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device, orgn_sents)

            test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False, collate_fn=my_collate)

            for idx, (bert_input, valid_indices, target_s, target_f, target_i, og_sents) in enumerate(test_loader):

                bert_outputs = bert_model(**bert_input)
                embeddings = bert_outputs.hidden_states[7]
                embeddings2 = bert_outputs.last_hidden_state

                # valid_embeds = [
                #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
                #     for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds2 = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings2, valid_indices)]

                batch_size = len(valid_embeds)

                seq_len = [t.shape[0] for t in target_s]

                padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
                padded_input2 = pad_sequence(valid_embeds2, batch_first=True, padding_value=0.0)
                padded_sense = pad_sequence(target_s, batch_first=True, padding_value=0.0)
                padded_frg = pad_sequence(target_f, batch_first=True, padding_value=0.0)
                padded_inter = pad_sequence(target_i, batch_first=True, padding_value=0.0)

                sense_out, frg_out, inter_out = tagging_model(padded_input, padded_input2)

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
                    ana_clauses = decode(sen, [tuple_to_dictlist(t_s) for t_s in tar_s], [tuple_to_list(t_f) for t_f in tar_f], [tuple_to_iterlabels(t_i) for t_i in tar_i], words.token_to_ix, count, pred_file, language)
                    #sen_prpty_file.write(str(count)+"\t"+str(len(sen))+"\n") # for relation between number of words and fscore
                    sen_prpty_file.write(ana_metrics(ana_clauses, count))  # for eval fscore of drs contains certain clause element                 
                    count+=1
            pred_file.close()
            sen_prpty_file.close()

            print("Sense Accurancy: ", correct_s/n_of_t)
            print("Fragment Accurancy: ", correct_f/n_of_t)
            print("intergration label Accurancy: ", correct_i/n_of_t)
  
@main.command()
@click.option("-l", "--language",type=click.Choice(['en', 'de','it', 'nl'], case_sensitive=False), help="en: English, de: German, it: Italian, nl: Dutch")
@click.option('-b','--batch', type=int, default=16, help="Batch size")
@click.option('-fm','--model_file', type=click.Path(exists=True), required=False, help="Provide model file path, and choose whether to save model. ")
@click.option('-fd','--dev_file', type=click.Path(exists=True), help="Dev file path (optional)")
@click.option('-f','--test_file', type=click.Path(exists=True), required=False, default=None, help="Test file path")
def test(language, batch, model_file, dev_file, test_file):

    f = open(dev_file)
    root_dir=os.path.realpath(f.name)
    f.close()
    in_files = [dev_file]
    out_files = [root_dir+'prediction_dev.txt']
    out_files2 = [root_dir+'sen_prpty_dev.txt']
    
    if dev_file != None:
        in_files.append(test_file)
        out_files.append(root_dir+'prediction_test.txt')
        out_files2.append(root_dir+'sen_prpty_test.txt')

    def object_decoder(obj):
        if '__type__' in obj and obj['__type__'] == 'dictionary':
            return preprocess.dictionary(obj['token_to_ix'], obj['ix_to_token'])
        return obj
    
    with open(root_dir+'word_dict.txt', 'r') as word_dict:
        words = json.loads(word_dict, object_hook=object_decoder)

    senses, fragment, integration_labels = load_labels(root_dir+'label_dict.txt')

    bert_models = {"en": "bert-base-cased","nl": "Geotrend/bert-base-nl-cased", "de": "dbmdz/bert-base-german-cased", "it": "dbmdz/bert-base-italian-cased"}

    model_name = bert_models[language]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bert_model = AutoModel.from_pretrained(model_name).to(device)

    bert_model.config.output_hidden_states=True

    tagging_model = Linear_classifiers(
        embed_size = 768, 
        syms_size = len(senses.token_to_ix), 
        frgs_size = len(fragment.token_to_ix), 
        intergs_size = len(integration_labels.token_to_ix),
        dropout_rate = 0.2
        ).to(device)

    checkpoint = torch.load(model_file, map_location=device)

    bert_model.load_state_dict(checkpoint['bert_model_state_dict'])

    tagging_model.load_state_dict(checkpoint['tagging_model_state_dict'])       



    for in_f, out_f, out_f2 in zip(in_files, out_files, out_files2):
        with torch.no_grad():

            print("Encoder Parameters:",sum([param.nelement() for param in bert_model.parameters()]))
            print("Decoder Parameters:",sum([param.nelement() for param in tagging_model.parameters()]))

 
            count = 1
            _, _, _, _, te_sents, _, _, orgn_sents, _, _ = preprocess.encode2(primary_file = in_f, language = language)
            pred_file = open( out_f, 'w', encoding="utf-8")
            sen_prpty_file = open( out_f2, 'w', encoding="utf-8")

            test_dataset = mydata.Dataset(te_sents,None, words.token_to_ix, senses.token_to_ix, fragment.token_to_ix, integration_labels.token_to_ix, tokenizer, device, orgn_sents)

            test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False, collate_fn=my_collate2)

            for idx, (bert_input, valid_indices, og_sents) in enumerate(test_loader):

                bert_outputs = bert_model(**bert_input)
                embeddings = bert_outputs.hidden_states[7]
                embeddings2 = bert_outputs.last_hidden_state

                # valid_embeds = [
                #     embeds[valid[:-1]]    #valid: idx(w[0])...idx([SEP])
                #     for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings, valid_indices)]

                valid_embeds2 = [
                    average_word_emb(embeds, valid)
                    for embeds, valid in zip(embeddings2, valid_indices)]

                batch_size = len(valid_embeds)

                seq_len = [ve.shape[0] for ve in valid_embeds]

                padded_input = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)
                padded_input2 = pad_sequence(valid_embeds2, batch_first=True, padding_value=0.0)

                sense_out, frg_out, inter_out = tagging_model(padded_input, padded_input2)

                sense_max = torch.argmax(sense_out, 2)
                frg_max = torch.argmax(frg_out, 2)
                inter_max = torch.argmax(inter_out, 2)

                unpad_sense = [sense_max[i,:l].tolist() for i, l in enumerate(seq_len)]
                unpad_frg = [frg_max[i,:l].tolist() for i, l in enumerate(seq_len)]
                unpad_inter = [inter_max[i,:l].tolist() for i, l in enumerate(seq_len)]

                sense_pred = [preprocess.ixs_to_tokens(senses.ix_to_token, seq) for seq in unpad_sense]
                frg_pred = [preprocess.ixs_to_tokens(fragment.ix_to_token, seq) for seq in unpad_frg]
                inter_pred = [preprocess.ixs_to_tokens(integration_labels.ix_to_token, seq) for seq in unpad_inter]

        
            #cd DRS_parsing_3\evaluation
            #python counter.py -f1 prediction_dev.txt -f2 dev.txt -prin -ms_file result_dev.txt -g clf_signature.yaml
            #python counter.py -f1 prediction_test.txt -f2 test.txt -prin -ms_file result_test.txt -g clf_signature.yaml

                for sen, tar_s, tar_f, tar_i in zip(og_sents,sense_pred,frg_pred,inter_pred):
                    ana_clauses = decode(sen, [tuple_to_dictlist(t_s) for t_s in tar_s], [tuple_to_list(t_f) for t_f in tar_f], [tuple_to_iterlabels(t_i) for t_i in tar_i], words.token_to_ix, count, pred_file, language)
                    #sen_prpty_file.write(str(count)+"\t"+str(len(sen))+"\n") # for relation between number of words and fscore
                    sen_prpty_file.write(ana_metrics(ana_clauses, count))  # for eval fscore of drs contains certain clause element                 
                    count+=1
            pred_file.close()
            sen_prpty_file.close()
    

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

def my_collate2(batch):

    input_ids = [item[0] for item in batch]
    token_type_ids = [item[1] for item in batch]
    attention_mask = [item[2] for item in batch]
    valid_indices = [item[3] for item in batch]
                
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0.0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)

    bert_input = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}


    orgn_sent = [item[4] for item in batch]

    return bert_input, valid_indices, orgn_sent




def average_word_emb(emb, valid):
    embs = []
    for i in range(len(valid)-1):
        embs.append(emb[torch.LongTensor([idx for idx in range(valid[i], valid[i+1])]).to(device)].mean(0).unsqueeze(0))
    return torch.cat(embs, 0)

if __name__ == '__main__':

    main()








        
