# FIXME support languages other than English


import drs
import re
import sys
from googletrans import Translator # dependency: pip install googletrans==4.0.0-rc1


from word2number import w2n


def guess_quantities(fragment, lang = 'en'):
    result = []
    for clause in fragment:
        if (clause[1] in ('Quantity', 'EQU')
            and not drs.is_constant(clause[3])
            and not drs.is_ref(clause[3])):
            string = quote(guess_quantity(unquote(clause[3]), lang))
            result.append((clause[0], clause[1], clause[2], string))
        else:
            result.append(clause)
    return tuple(result)


DECIMAL_NUMBER_PATTERN = re.compile(r'\d[\d,]*(\.\d+)?$')
ZERO_DECIMALS_PATTERN = re.compile(r'\.0+$')


def guess_quantity(origin_string, lang = 'en'):
    if lang != 'en':
        translator = Translator()
        string_list = translator.translate(origin_string.replace("~", " "), dest="en").text.split()
        string = "~".join(string_list)
    else:
        string = origin_string
    if string.endswith('-') and len(string) > 1:
        return guess_quantity(string[:-1])
    if string.startswith('a~') and len(string) > 2:
        return guess_quantity(string[2:])
    if string.endswith('~and~a~half') and len(string) > 11:
        return str(int(guess_quantity(string[:-11])) + 0.5)
    split = string.split('~')
    if len(split) == 2 and split[1] in ('hundred', 'thousand', 'million',
                                        'billion'):
        try:
            n1 = tok2num(split[0])
            n2 = tok2num(split[1])
            return str(int(n1 * n2))
        except ValueError:
            pass
    try:
        return str(w2n.word_to_num(string.replace('~', ' ')))
    except ValueError:
        pass
    match = DECIMAL_NUMBER_PATTERN.match(string)
    if match:
        return ZERO_DECIMALS_PATTERN.sub('', string.replace(',', ''))
    if string in ('loads', 'lot', 'lots', 'many', 'much', 'numerous', 'plenty',
                  'several'):
        return '+'
    if string in ('bit', 'few', 'little'):
        return '-'
    if string == 'dozen':
        return '12'
    if string in ('half-dozen', 'half~dozen'):
        return '6'
    if string == 'twice':
        return '2'
    if string == 'half':
        return '0.5'
    if string in ('a', 'an'):
        return '1'
    if string in ('how', 'where', 'which', 'what'):
        return '?'
    if string == 'per':
        return '1' # whatever
    return origin_string

def tok2num(tok):
    """Converts a token without whitespace to a number.
    """
    try:
        return w2n.word_to_num(tok)
    except ValueError:
        pass
    match = DECIMAL_NUMBER_PATTERN.match(tok)
    if match:
        return float(tok.replace(',', ''))
    raise ValueError()


def unquote(string):
    assert string.startswith('"')
    assert string.endswith('"')
    return string[1:-1]


def quote(string):
    return '"' + string + '"'