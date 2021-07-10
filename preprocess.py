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

def encode2(encoding='ret-int', primary_file = 'Data\\toy\\train.txt', optional_file = None, optional_file2 = None, language = "en"):
    words = dictionary()
    senses = dictionary()
    clauses = dictionary()
    integration_labels = dictionary()

    content_frg_idx = set([])

    sents = []
    targets = []
    sents2 = []
    targets2 = []
    max_seq_len = 0

    files = [primary_file]
    if optional_file != None:
        files.append(optional_file)
    if optional_file2 != None:
        files.append(optional_file2)
    for file_idx, file in enumerate(files):
        data_file = open(file, encoding = 'utf-8')
        for i, (sentence, fragments, unaligned) in enumerate(
                clf.read(data_file), start=1):
            if len(sentence)<=38 and "政務顧問" not in sentence:
                max_seq_len = max(max_seq_len, len(sentence))
                #alignment.align(unaligned, fragments, i)
                syms = tuple(symbols.guess_symbol(w, language) for w in sentence)
                fragments = constants.add_constant_clauses(syms, fragments)
                fragments = constants.replace_constants(fragments)
                fragments = tuple(drs.sorted(f) for f in fragments)
                fragments = address.debruijnify(fragments)

                #sent = ["-BOS-"]
                #target = [("-BOS-","-BOS-", "-BOS-")]
                sent = []
                target = []

                for word, fragment in zip(sentence, fragments):
                    fragment, syms = mask.mask_fragment(fragment)
                    if encoding == 'ret-int':
                        fragment, integration_label = address.abstract(fragment)
                    else:
                        integration_label = {}


                    words.insert(word)
                    senses.insert(dictlist_to_tuple(syms))
                    clauses.insert(tuple(fragment))
                    integration_labels.insert(dictlist_to_tuple(integration_label))

                    if "work" in syms or "\"tom\"" in syms:
                        content_frg_idx.add(clauses.token_to_ix[tuple(fragment)])
                    
                    
                    sent.append(word)
                    target.append((dictlist_to_tuple(syms), tuple(fragment), dictlist_to_tuple(integration_label)))
     

                #sent.append("-EOS-")
                #target.append(("-EOS-","-EOS-", "-EOS-"))
                #if len(sent) <=38:
                if file_idx == 0:
                    sents.append(sent)
                    targets.append(target)
                elif file_idx > 0:
                    sents2.append(sent)
                    targets2.append(target)

    print(f"max sequence length: {max_seq_len}", file=sys.stderr)
    if optional_file ==None:
        return words, senses, clauses, integration_labels, sents, targets, content_frg_idx, None, None
    else:
        return words, senses, clauses, integration_labels, sents, targets, content_frg_idx, sents2, targets2


def encode(encoding='ret-int', data_file = open('Data\\mergedata\\gold\\gold.clf', encoding = 'utf-8')):
    retrieval_labels = set()
    integration_labels = []
    max_seq_len = 0
    null_label = 0
    all_label = 0
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
                if integration_label == {"b": [], "e": [], "n": [], "p": [], "s": [], "t": [], "x": []}:
                    null_label+=1
                all_label+=1
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
    print("labels:" + str(null_label/all_label))



if __name__ == '__main__':
    encode()
    # a, b, c, d,e,fo=encode2()
    # for s,t in zip(e,fo):
    #     sent, target_s, target_f, traget_i = list(map(lambda x: tokens_to_ixs(x[0], x[1]),[(
    #         a.token_to_ix, s), (
    #             b.token_to_ix, [w[0] for w in t]), (
    #                 c.token_to_ix, [w[1] for w in t]), (
    #                     d.token_to_ix, [w[2] for w in t])]))

    #     print(len(s),traget_i)