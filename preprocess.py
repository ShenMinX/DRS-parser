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

from error_eval import ana_metrics2

class dictionary():

    def __init__(self, token_to_ix, ix_to_token):
        self.insert_mark()
        self.token_to_ix = token_to_ix
        self.ix_to_token = ix_to_token
        
    
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

VARAIBLE_PATTERN = re.compile(r'(?P<type>[benpstx])(?P<index>\d+)$')

def rename_var(fragments):
    #type_refs_map = collections.defaultdict(list)
    syms_list = []
    type_refs_map = {'b': [], 'e': [], 'n': [], 'p': [], 's': [], 't': [], 'x': []}
    new_fragments = []
    for f in fragments:
        new_fragment = []
        for c in f:
            new_clause = []
            for s in c:
                match = VARAIBLE_PATTERN.match(s)
                new_s = s
                if match:
                    type_ = match.group('type')
                    index_ = match.group('index')
                    if index_ not in type_refs_map[type_]:
                        type_refs_map[type_].append(index_)
                    new_s = type_+str(type_refs_map[type_].index(index_)+1)
                new_clause.append(new_s)
            new_fragment.append(tuple(new_clause))
        new_fragment, syms = mask.mask_fragment(tuple(new_fragment))
        new_fragments.append(new_fragment)
        syms_list.append(syms)
    return tuple(new_fragments), syms_list

def mask_norename(fragments):
    new_fragments = []
    syms_list = []
    for f in fragments:
        new_f, syms = mask.mask_fragment(f)
        new_fragments.append(new_f)
        syms_list.append(syms)
    return tuple(new_fragments), syms_list

def encode2(encoding='ret-int', primary_file = 'Data\\toy\\train.txt', optional_file = None, optional_file2 = None, language = "en"):
    words = dictionary({},{})
    senses = dictionary({},{})
    clauses = dictionary({},{})
    integration_labels = dictionary({},{})

    sents = []
    targets = []
    sents2 = []
    targets2 = []
    max_seq_len = 0

    orgn_sents = []

    unks = {}

    if language in ["en","de","nl","it"]:     
        print("lang: "+ language)  
        unk_file = open('Data\\'+language+'\\all_unk.txt', encoding = 'utf-8')
        for entry in unk_file:
            entry_list = entry.rstrip("\n").split("\t")
            if len(entry_list)==3 and entry_list[2] !='':
                unks[entry_list[1]]=entry_list[2]
        unk_file.close()

    files = [primary_file]
    if optional_file != None:
        files.append(optional_file)
    if optional_file2 != None:
        files.append(optional_file2)
    for file_idx, file in enumerate(files):
        data_file = open(file, encoding = 'utf-8')
        for i, (sentence, fragments, unaligned) in enumerate(
                clf.read(data_file), start=1):
            if len(sentence)<=38:
                max_seq_len = max(max_seq_len, len(sentence))
                #alignment.align(unaligned, fragments, i)
                syms = tuple(symbols.guess_symbol(w, language) for w in sentence)
                fragments = constants.add_constant_clauses(syms, fragments)
                fragments = constants.replace_constants(fragments)
                #fragments, syms_list = rename_var(fragments)
                #fragments, syms_list = mask_norename(fragments)
                fragments = tuple(drs.sorted(f) for f in fragments)
                fragments = address.debruijnify(fragments)

                sent = []
                target = []

                #for word, fragment, syms in zip(sentence, fragments, syms_list):
                for word, fragment in zip(sentence, fragments):
                    fragment, syms = mask.mask_fragment(fragment)
                    if encoding == 'ret-int':
                        fragment, integration_label = address.abstract(fragment)
                    else:
                        integration_label = {}

                    for c in word:
                        if c in unks:
                            word = word.replace(c, unks[c])

                    
                    senses.insert(dictlist_to_tuple(syms))
                    clauses.insert(tuple(fragment))
                    integration_labels.insert(dictlist_to_tuple(integration_label))


                    if word != "":
                        words.insert(word)
                        sent.append(word)
                    target.append((dictlist_to_tuple(syms), tuple(fragment), dictlist_to_tuple(integration_label)))
     

                #if len(sent) <=38:
                if file_idx == 0:
                    sents.append(sent)
                    targets.append(target)
                    orgn_sents.append(sentence)
                elif file_idx > 0:
                    sents2.append(sent)
                    targets2.append(target)


    print(f"max sequence length: {max_seq_len}", file=sys.stderr)
    print(f"# senses: {len(senses.token_to_ix)}")
    print(f"# fragments: {len(clauses.token_to_ix)}")
    print(f"# integration_labels: {len(integration_labels.token_to_ix)}")
    if optional_file ==None:
        return words, senses, clauses, integration_labels, sents, targets, orgn_sents, None, None
    else:
        return words, senses, clauses, integration_labels, sents, targets, orgn_sents, sents2, targets2


def encode(encoding='ret-int', data_file = open('Data\\en\\gold\\dev.txt', encoding = 'utf-8')):
    retrieval_labels = set()
    integration_labels = []
    max_seq_len = 0
    null_label = 0
    all_label = 0
    sen_prpty_file = open( 'Data\\en\\gold\\sen_prpty_dev.txt', 'w', encoding="utf-8")
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

        # print('\t'.join(('-BOS-',) * column_count))
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
    #         print(*fields, sep='\t')
    #     print('\t'.join(('-EOS-',) * column_count))
    #     print()
    # print(f'# of retrieval labels: {len(retrieval_labels)}', file=sys.stderr)

        print(fragments)
        sen_prpty_file.write(ana_metrics2(fragments, i))

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