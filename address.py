"""Functions related to referent addresses."""


import drs
import collections
import re
import util


RELATIVE_REF_PATTERN = re.compile(r'(?P<type>[benpstx])(?P<index>0|-\d+)')


def debruijnify(fragments):
    type_refs_map = collections.defaultdict(list)
    def dbrnf(arg):
        if not drs.is_ref(arg):
            return arg
        type_ = arg[0]
        refs = type_refs_map[type_]
        index = util.nrfind(refs, arg)
        if index != 0:
            refs.pop(index)
        refs.append(arg)
        return f"{type_}{index}"
    return tuple(tuple(tuple(dbrnf(a) for a in c) for c in f) for f in fragments)


def undebruijnify(drs):
    result = []
    type_refs_map = collections.defaultdict(list)
    def ddbrnf(arg):
        match = RELATIVE_REF_PATTERN.match(arg)
        if not match:
            return arg
        type_ = match.group('type')
        index = int(match.group('index'))
        refs = type_refs_map[type_]
        if index == 0 or len(refs) < abs(index):
            if refs:
                nmax = max(int(r[1:]) for r in refs)
            else:
                nmax = 0
            ref = f'{type_}{nmax + 1}'
        else:
            ref = refs.pop(index)
        refs.append(ref)
        return ref
    return tuple(tuple(tuple(ddbrnf(a) for a in c) for c in f) for f in drs)


def abstract(fragment):
    integration_label = {'b': [], 'e': [], 'n': [], 'p': [], 's': [], 't': [], 'x': []}
    local_counts = {type_: 0 for type_ in 'benpstx' }
    def bstrct(arg):
        # case 1: not a ref
        match = RELATIVE_REF_PATTERN.match(arg)
        if not match:
            return arg
        # case 2: a new ref
        type_ = match.group('type')
        index = int(match.group('index'))
        if index == 0:
            local_counts[type_] += 1
            return arg
        # case 2: a locally resolved ref
        if abs(index) <= local_counts[type_]:
            return arg
        # case 3: a nonlocally resolved ref
        distance = index + local_counts[type_]
        integration_label[type_].append(distance)
        index = -1 - local_counts[type_]
        ref = f'{type_}{index}'
        local_counts[type_] += 1
        return ref
    abstract_fragment = tuple(tuple(bstrct(a) for a in c) for c in fragment)
    for l in integration_label.values():
        while l and l[-1] == -1:
            l.pop()
    return abstract_fragment, integration_label


def unabstract(fragment, integration_label):
    local_counts = {type_: 0 for type_ in 'benpstx' }
    def nbstrct(arg):
        # case 1: not a ref
        match = RELATIVE_REF_PATTERN.match(arg)
        if not match:
            return arg
        # case 2: a new ref
        type_ = match.group('type')
        index = int(match.group('index'))
        if index == 0:
            local_counts[type_] += 1
            return arg
        # case 3: a locally resolved ref
        if abs(index) <= local_counts[type_]:
            return arg
        # case 4: a nonlocally resolved ref
        try:
            distance = integration_label[type_].pop(0)
        except IndexError:
            distance = -1
        index = distance - local_counts[type_]
        ref = f'{type_}{index}'
        local_counts[type_] += 1
        return ref
    return tuple(tuple(nbstrct(a) for a in c) for c in fragment)