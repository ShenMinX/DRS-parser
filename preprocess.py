import address
import alignment
import clf
import constants
import click
import drs
import json
import mask
import symbols
import sys
import re

class dictionary():

    def __init__(self):
        self.token_to_ix = {}
        self.ix_to_token = {}
        self.insert_mark()
    
    def insert(self, token):
        if token not in self.token_to_ix:
            ix = len(self.token_to_ix)
            self.token_to_ix[token] = ix
            self.ix_to_token[ix] = token 

    def insert_mark(self):
        marks = ["[PAD]","-EOS-","-BOS-","[UNK]"]
        for m in marks:
            self.insert(m)   

def tokens_to_ixs(token_to_ix, tokens):
    out = []
    for w in tokens:
        if w in token_to_ix:
            out.append(token_to_ix[w])
        else:
            out.append(token_to_ix["[UNK]"])
    return out

def ixs_to_tokens(ix_to_token, ixs):
    out = []
    for w in ixs:
        out.append(ix_to_token[w])
    return out

def dictlist_to_tuple(dict):
    return tuple((x, tuple(z for z in y)) for x, y in dict.items())

def encode2(encoding='ret-int', data_file = open('Data\\toy\\train.txt', encoding = 'utf-8')):
    words = dictionary()
    sent_char = dictionary()
    senses = dictionary()
    clauses = dictionary()
    integration_labels = dictionary()

    sent_char.insert("[break]")

    senses.insert("[none]")
    senses.insert("[break]")

    SENSE_STRING_PATTERN = re.compile(r'"(?P<pos>[nvar])\.(?P<number>\d\d?)"')

    sents = []
    char_sents = []
    
    target_senses = []
    targets = []
    max_seq_len = 0
    for i, (sentence, fragments, unaligned) in enumerate(
            clf.read(data_file), start=1):
        max_seq_len = max(max_seq_len, len(sentence))
        #alignment.align(unaligned, fragments, i)
        syms = tuple(symbols.guess_symbol(w, 'en') for w in sentence)
        fragments = constants.add_constant_clauses(syms, fragments)
        fragments = constants.replace_constants(fragments)
        fragments = tuple(drs.sorted(f) for f in fragments)
        fragments = address.debruijnify(fragments)

        sent = ["-BOS-"]
        char_sent = ["-BOS-"]
        target = [("-BOS-", "-BOS-")]
        sense_seq = ["-BOS-"]

        for word, fragment in zip(sentence, fragments):
            fragment, syms = mask.mask_fragment(fragment)
            if encoding == 'ret-int':
                fragment, integration_label = address.abstract(fragment)
            else:
                integration_label = {}

            words.insert(word)
            for ch in word:
                sent_char.insert(ch)
                char_sent.append(ch)
            char_sent.append("break")

            if "work" in syms and "\"v.00\"" in syms:
                for ch in syms["work"]:
                    senses.insert(ch)
                    sense_seq.append(ch)
                match = SENSE_STRING_PATTERN.search(syms["\"v.00\""])
                senses.insert("["+match.group('pos')+"]")
                sense_seq.append("["+match.group('pos')+"]")
                senses.insert("["+match.group('number')+"]")
                sense_seq.append("["+match.group('number')+"]")
                sense_seq.append("[break]")
            
            elif "\"tom\"" in syms:
                for ch in syms["\"tom\""]:
                    if ch !="\"":
                        senses.insert(ch)
                        sense_seq.append(ch)
                sense_seq.append("[break]")

            else:
                sense_seq.append("[none]")
                sense_seq.append("[break]")
        
            clauses.insert(tuple(fragment))
            integration_labels.insert(dictlist_to_tuple(integration_label))
            
            sent.append(word)
            target.append((tuple(fragment), dictlist_to_tuple(integration_label)))

        del sense_seq[-1]
        sense_seq.append("-EOS-")

        del char_sent[-1]
        char_sent.append("-EOS-")

        sent.append("-EOS-")
        target.append(("-EOS-", "-EOS-"))

        sents.append(sent)
        char_sents.append(char_sent)
        targets.append(target)
        target_senses.append(sense_seq)

    print(f"max sequence length: {max_seq_len}", file=sys.stderr)

    return words, sent_char, senses, clauses, integration_labels, sents, char_sents, targets, target_senses


def encode(encoding='ret-int', data_file = open('Data\\mergedata\\gold\\gold.clf', encoding = 'utf-8')):
    retrieval_labels = set()
    integration_labels = []
    max_seq_len = 0
    for i, (sentence, fragments, unaligned) in enumerate(
            clf.read(data_file), start=1):
        max_seq_len = max(max_seq_len, len(sentence))
        #alignment.align(unaligned, fragments, i)
        syms = tuple(symbols.guess_symbol(w, 'en') for w in sentence)
        fragments = constants.add_constant_clauses(syms, fragments)
        fragments = constants.replace_constants(fragments)
        fragments = tuple(drs.sorted(f) for f in fragments)
        fragments = address.debruijnify(fragments)
        column_count = 4
        print('\t'.join(('-BOS-',) * column_count))
        for word, fragment in zip(sentence, fragments):
            fragment, syms = mask.mask_fragment(fragment)
            if encoding == 'ret-int':
                fragment, integration_label = address.abstract(fragment)
            else:
                integration_label = {}
            integration_labels.append(integration_label)
            retrieval_labels.add(fragment)
            fields = [
                word,
                json.dumps(syms),
                json.dumps(fragment),
                json.dumps(integration_label)
            ]
            print(*fields, sep='\t')
        print('\t'.join(('-EOS-',) * column_count))
        print()
    print(f'# of retrieval labels: {len(retrieval_labels)}', file=sys.stderr)
    if encoding == 'ret-int':
        maxxref = max(
            len(v)
            for l in integration_labels
            for v in l.values()
        )
        integration_labels = set(json.dumps(l) for l in integration_labels)
        print(f'# of integration labels: {len(integration_labels)}', file=sys.stderr)
    print(f"max sequence length: {max_seq_len}", file=sys.stderr)



if __name__ == '__main__':
    #encode()
    words, sent_char, senses, clauses, integration_labels, sents, char_sents, targets, target_senses =encode2()

    # for seq in target_senses:
    #     print(seq)
    for sen in char_sents:
        print(sen)