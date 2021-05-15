#!/usr/bin/env python3


import address
import constants
import clf
import drs
import fix
import guess
import json
import mask
import quantities
import srl
import sys
import times
import util

import preprocess


def parse_label(string):
    if string in ('-BOS-', '-EOS-', '[MASK_LABEL]'): # wonky label
        return ()
    return json.loads(string)


def tuple_to_dictlist(tup):
    if len(tup)!=0:
        return dict((x, list(y)) for x, y in tup)
    else:
        return {}


def tuple_to_list(tup):
    return [list(x) for x in tup]


def symbolize(fragment, word, lemma=None):
    fragment = guess.guess_name(fragment, word)
    fragment = times.guess_times(fragment)
    fragment = quantities.guess_quantities(fragment)
    if lemma:
        fragment = guess.guess_concept_from_lemma(fragment, lemma)
    else:
        fragment = guess.guess_concept_from_word(fragment, word)
    return fragment


def read_lemmas(blocks):
    block = next(blocks)
    assert block[-1] == '\n'
    return tuple(l.rstrip() for l in blocks[:-1])


def decode(sentence, symbols, fragments, integration_actions, i, encoding='ret-int', gold_symbols=True, roles=None, lemmas=None, mode=2):
    checker = drs.Checker(mode)
    if roles:
        roler = srl.Roler((json.loads(l) for l in roles), checker)
    if lemmas:
        lemmas = util.blocks(lemmas)
    # for i, block in enumerate(util.blocks(sys.stdin), start=1):
    #     assert block[0].startswith('-BOS-\t-BOS-\t')
    #     assert block[-2].startswith('-EOS-\t-EOS-\t')
    #     assert block[-1] == '\n'
    #     block = block[1:-2]
    #     block = tuple(l.rstrip().split('\t') for l in block)
    #     sentence = tuple(l[0] for l in block)
    #     symbols = tuple(json.loads(l[1]) for l in block)
    #     fragments = tuple(parse_label(l[2]) for l in block)
    #     integration_actions = tuple(parse_label(l[3]) for l in block)
    if encoding == 'ret-int':
        fragments = tuple(
            address.unabstract(f, i)
            for f, i
            in zip(fragments, integration_actions)
        )
    fragments = address.undebruijnify(fragments)
    if lemmas:
        sentence_lemmas = []
        while len(sentence_lemmas) < len(sentence):
            sentence_lemmas.extend(read_lemmas(lemmas))
        if len(sentence_lemmas) > len(sentence):
            raise RuntimeError('Length mismatch: more lemmas than tokens')
        sentence_lemmas = tuple(sentence_lemmas)
    else:
        sentence_lemmas = (None,) * len(sentence)
    if gold_symbols:
        fragments = tuple(
            mask.unmask_fragment(f, s)
            for f, s
            in zip(fragments, symbols)
        )
    else:
        fragments = tuple(
            symbolize(f, w, l) # TODO use lemmas
            for f, w, l
            in zip(fragments, sentence, sentence_lemmas)
        )
    if roles:
        fragments = roler.overwrite_roles(fragments, sentence)
    fragments = constants.replace_constants_rev(fragments)
    fragments = constants.remove_constant_clauses(fragments)
    clauses = [c for f in fragments for c in f]
    # TODO make fixes unnecessary for reconstruction by aligning unaligned clauses
    if mode == 2:
        fix.add_missing_box_refs(clauses, checker)
    fix.add_missing_concept_refs(clauses)
    fix.add_missing_arg0_refs(clauses)
    fix.add_missing_arg1_refs(clauses)
    if mode == 2:
        import fix2
        fix2.ensure_nonempty(clauses)
        fix2.ensure_main_box(clauses)
    elif mode == 3:
        import fix3
        clauses = fix3.ensure_no_loops(clauses)
        clauses = fix3.ensure_connected(clauses)
        fix3.ensure_nonempty(clauses)
    fix.dedup(clauses)
    if mode == 2:
        fix2.check(clauses, i)
    elif mode == 3:
        fix3.check(clauses, i)
    print(f"%%% {' '.join(sentence)}")
    clf.write((clauses,), sys.stdout)


if __name__ == '__main__':

    words, senses, clauses, integration_labels, sents, targets = preprocess.encode2()
    for i, (sen, tar) in enumerate(zip(sents, targets)):
        decode(sen[1: -1], [tuple_to_dictlist(t[0]) for t in tar[1:-1]], [tuple_to_list(t[1]) for t in tar[1:-1]], [tuple_to_dictlist(t[2]) for t in tar[1:-1]], i+1)