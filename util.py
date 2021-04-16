def blocks(stream):
    """Splits a text stream by empty lines.

    Reads a file-like object and returns its contents chopped into a sequence
    of blocks terminated by empty lines.
    """
    block = []
    for line in stream:
        block.append(line)
        (chomped,) = line.splitlines()
        if chomped == '':
            yield block
            block = []
    if block: # in case the last block is not terminated
        yield block


def nrfind(seq, element):
    """Finds the position of element in seq from the end.

    Returns the greatest negative index i such that seq[i] == element, or 0
    if element does not occur in seq.
    """
    for i, e in enumerate_reversed(seq):
        if e == element:
            return i
    return 0


def enumerate_reversed(seq):
    """Enumerates a sequence from end to beginning.

    Returns a sequence of tuples (i, e) where seq[i] == e, starting with
    i == -1 and ending with i == -len(seq).
    """
    return (
        (-i, e)
        for i, e
        in enumerate(reversed(seq), start=1)
    )
