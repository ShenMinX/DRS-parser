import nltk
import operator
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
LEMMATIZER = WordNetLemmatizer()


def guess_name(fragment, word):
    string = '"{}"'.format(word.lower().replace('"', "'"))
    return replace('"tom"', string, fragment)


def construct_ws(word, pos, frq_senses):
    lemma = LEMMATIZER.lemmatize(word, pos='n').lower()
    key = word+pos
    if key in frq_senses:
        ss = max(frq_senses[key].items(), key=operator.itemgetter(1))[0]
        concept = ss[:-5]
        pos_number = ss[-4:]
        return concept, '"'+pos_number+'"'
    else:
        return lemma, '"'+pos+'.01'+'"'

def guess_concept_from_word(fragment, word, frq_senses):
    result = []
    for clause in fragment:
        if clause[2] == '"n.00"':
            #lemma = LEMMATIZER.lemmatize(word, pos='n').lower()
            concept, pos_sum = construct_ws(word, 'n', frq_senses)
            result.append((clause[0], concept, pos_sum, clause[3]))
        elif clause[2] == '"v.00"':
            #lemma = LEMMATIZER.lemmatize(word, pos='v').lower()
            concept, pos_sum = construct_ws(word, 'v', frq_senses)
            result.append((clause[0], concept, pos_sum, clause[3]))
        elif clause[2] == '"a.00"':
            #lemma = LEMMATIZER.lemmatize(word, pos='a').lower()
            concept, pos_sum = construct_ws(word, 'a', frq_senses)
            result.append((clause[0], concept, pos_sum, clause[3]))
        elif clause[2] == '"r.00"':
            #lemma = LEMMATIZER.lemmatize(word, pos='r').lower()
            concept, pos_sum = construct_ws(word, 'r', frq_senses)
            result.append((clause[0], concept, pos_sum, clause[3]))
        else:
            result.append(clause)
    return tuple(result)


def guess_concept_from_lemma(fragment, lemma):
    result = []
    for clause in fragment:
        if clause[2] == '"n.00"':
            result.append((clause[0], lemma, '"n.01"', clause[3]))
        elif clause[2] == '"v.00"':
            result.append((clause[0], lemma, '"v.01"', clause[3]))
        elif clause[2] == '"a.00"':
            result.append((clause[0], lemma, '"a.01"', clause[3]))
        elif clause[2] == '"r.00"':
            result.append((clause[0], lemma, '"r.01"', clause[3]))
        else:
            result.append(clause)
    return tuple(result)


def replace(old, new, obj):
    if old == obj:
        return new
    if isinstance(obj, tuple):
        return tuple(replace(old, new, e) for e in obj)
    return obj

