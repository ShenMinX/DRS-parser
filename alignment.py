import sys


def align(unaligned, fragments, i):
    while unaligned:
        success = False
        for i, clause in enumerate(unaligned):
            j = find_fragment(clause, fragments)
            if j is not None:
                success = True
                fragments[j].append(unaligned.pop(i))
                break
        if not success:
            print(f'WARNING: failed to align clauses in DRS {i}: {unaligned}',
                    file=sys.stderr)
            break


def find_fragment(unaligned_clause, fragments):
    for arg in unaligned_clause[2:]:
        for j, fragment in enumerate(fragments):
            for aligned_clause in fragment:
                if arg in aligned_clause[2:]:
                    return j
        for j, fragment in enumerate(fragments):
            for aligned_clause in fragment:
                if arg == aligned_clause[0]:
                    return j
