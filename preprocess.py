import address
import alignment
import clf
import constants
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
        marks = set(["[PAD]","-EOS-","-BOS-","[UNK]"])
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

def ixs_to_tokens_no_mark(ix_to_token, ixs, zero_token):
    out = []
    for w in ixs:
        if ix_to_token[w] != "[PAD]" and ix_to_token[w] != "-EOS-" and ix_to_token[w] != "-BOS-":
            out.append(ix_to_token[w])
        else:
            out.append(zero_token)
    return out

def dictlist_to_tuple(dict):
    return tuple((x, tuple(z for z in y)) for x, y in dict.items())

def get_words_len(sent):
    return [len(w)+2 for w in sent]

def encode2(encoding='ret-int', data_file = open('Data\\mergedata\\gold\\gold.clf', encoding = 'utf-8')):
    words = dictionary()
    chars = dictionary()
    clauses = dictionary()
    integration_labels = dictionary()

    content_frg_idx = set([])
    prpname_frg_idx = set([])

    SENSE_STRING_PATTERN = re.compile(r'"(?P<pos>[nvar])\.(?P<number>\d\d?)"')

    sents = []
    char_sents = []
    
    target_senses = []
    targets = []
    max_seq_len = 0

    max_sense_lens = []
    for i, (sentence, fragments, unaligned) in enumerate(
            clf.read(data_file), start=1):
        max_seq_len = max(max_seq_len, len(sentence))
        #alignment.align(unaligned, fragments, i)
        syms = tuple(symbols.guess_symbol(w, 'en') for w in sentence)
        fragments = constants.add_constant_clauses(syms, fragments)
        fragments = constants.replace_constants(fragments)
        fragments = tuple(drs.sorted(f) for f in fragments)
        fragments = address.debruijnify(fragments)

        # sent = ["-BOS-"]
        sent = []
        char_sent = []
        # target = [("-BOS-", "-BOS-")]
        target = []
        
        
        sense_seqence = []
        max_sense_len = 0

        for word, fragment in zip(sentence, fragments):
            fragment, syms = mask.mask_fragment(fragment)
            if encoding == 'ret-int':
                fragment, integration_label = address.abstract(fragment)
            else:
                integration_label = {}

            words.insert(word)
            clauses.insert(tuple(fragment))
            integration_labels.insert(dictlist_to_tuple(integration_label))

            word_seq = ["-BOS-"]

            for ch in word:
                chars.insert(ch)
                word_seq.append(ch)
            word_seq.append("-EOS-")
            char_sent.append(word_seq)

            sense_seq = []

            if "work" in syms:
                for ch in syms["work"]:
                    chars.insert(ch)
                    sense_seq.append(ch)
                if "\"v.00\"" in syms:
                    pos_num = syms["\"v.00\""]
                elif "\"a.00\"" in syms:
                    pos_num = syms["\"a.00\""]
                elif "\"r.00\"" in syms:
                    pos_num = syms["\"r.00\""]
                elif "\"n.00\"" in syms:
                    pos_num = syms["\"n.00\""]
                match = SENSE_STRING_PATTERN.search(pos_num)
                chars.insert("["+match.group('pos')+"]")
                sense_seq.append("["+match.group('pos')+"]")
                chars.insert("["+match.group('number')+"]")
                sense_seq.append("["+match.group('number')+"]")
                content_frg_idx.add(clauses.token_to_ix[tuple(fragment)])
                sense_seq.append("-EOS-")
            
            elif "\"tom\"" in syms:
                for ch in syms["\"tom\""]:
                    if ch !="\"":
                        chars.insert(ch)
                        sense_seq.append(ch)
                sense_seq.append("-EOS-")
                content_frg_idx.add(clauses.token_to_ix[tuple(fragment)])
                prpname_frg_idx.add(clauses.token_to_ix[tuple(fragment)])

            else:
                sense_seq.append("[PAD]")

                   
            sent.append(word)
            target.append((tuple(fragment), dictlist_to_tuple(integration_label)))
            
            sense_seqence.append(sense_seq)

            
            max_sense_len = max(len(sense_seq), max_sense_len)


        # sent.append("-EOS-")
        # target.append(("-EOS-", "-EOS-"))

        sents.append(sent)
        char_sents.append(char_sent)
        targets.append(target)
        target_senses.append(sense_seqence)
        max_sense_lens.append(max_sense_len)

    print(f"max sequence length: {max_seq_len}", file=sys.stderr)


    return words, chars, clauses, integration_labels, content_frg_idx, prpname_frg_idx, sents, char_sents, targets, target_senses, max_sense_lens


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
    encode()
    # words, chars, fragments, integration_labels, content_frg_idx, prpname_frg_idx, sents, char_sents, targets, \
    #      target_senses, max_sense_lens =encode2()

    # for seq in target_senses:
    #     print(seq)
    # for sen in char_sents:
    #     print(sen)