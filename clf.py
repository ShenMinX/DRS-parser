"""Functions for handling clauses.

As in the PMB CLF format.
"""


import collections
import re
import util


from typing import List, NewType, Sequence, TextIO, Tuple


Clause = NewType('Clause', Tuple)


SEP_PATTERN = re.compile(r' *% ?')
TOK_PATTERN = re.compile(r'([^ ]+ \[(?P<fr>\d+)\.\.\.(?P<to>\d+)\])')


def token_sortkey(token):
    match = TOK_PATTERN.match(token)
    return int(match.group('fr'))


def read(flo: TextIO): #-> Sequence[Tuple[str, Tuple[Sequence[Clause]]]]:
    for block in util.blocks(flo):
        if block[-1].rstrip() == '':
            block = block[:-1]
        while block[0].startswith('%%% '):
            sentence = tuple(t for t in block[0].rstrip()[4:].split(' ') if t != 'ø')
            block = block[1:]
        token_fragment_map = collections.defaultdict(list)
        unaligned_clauses = []
        for line in block:
            clause, tokens = SEP_PATTERN.split(line, 1)
            clause = tuple(clause.split(' '))
            if clause == ('',):
                clause = ()
            else:
                assert 3 <= len(clause) <= 4
            tokens = TOK_PATTERN.findall(tokens)
            for token in tokens:
                token_fragment_map[token[0]].append(clause)
                if not clause:
                    token_fragment_map[token[0]].pop(-1)
            if not tokens:
                unaligned_clauses.append(clause)
        fragments = [
            token_fragment_map[k]
            for k
            in sorted(token_fragment_map, key=token_sortkey)
        ]
        yield (sentence, fragments, unaligned_clauses)


def write(drs, flo):
    for token in drs:
        for clause in token:
            print(' '.join(clause), file=flo)
    print(file=flo)
