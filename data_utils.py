import json
import string
import re

import collections
import nltk
import sys


def camel_cut(name):
    ans = []
    start, end = 0, len(name)
    for i in range(1, len(name)-1):
        if name[i].isupper() and name[i+1].islower() and name[i-1] != ' ':
            end = i
            ans.append(name[start: end].lower())
            start, end = i, len(name)
    ans.append(name[start: end].lower())
    return ans


def tokenize(identifier):
    filename = 'data/' + identifier + '_desc.json'
    with open(filename, 'r') as f:
        lines = f.readlines()
    nl_names = []
    nl_lengths = []
    name_lengths = []

    for line in lines:
        line = json.loads(line)
        nl = line['nl']
        name = line['name']
        name_tokens = camel_cut(name)
        name_lengths.append(len(name_tokens))
        words = []
        nl_tokens = []
        sents = nltk.sent_tokenize(nl)
        for sent in sents:
            words.extend(nltk.word_tokenize(sent))

        pun = '[' + string.punctuation + ' ' + string.digits + ']'
        lemmatizer = nltk.WordNetLemmatizer()
        for word in words:
            if re.match(pun, word) is not None:
                continue
            word = lemmatizer.lemmatize(word)
            if word.find('.') != -1:
                ws = word.split('.')
                ws = [w.lower() for w in ws]
                nl_tokens.extend(ws)
                continue
            if word.find('_') != -1:
                ws = word.split('_')
                ws = [w.lower() for w in ws]
                nl_tokens.extend(ws)
                continue
            nl_tokens.append(word.lower())

        nl_name = {'name': name_tokens, 'nl': nl_tokens}
        nl_lengths.append(len(nl_tokens))
        nl_names.append(nl_name)
    nl_counters = dict(collections.Counter(nl_lengths))
    name_counters = dict(collections.Counter(name_lengths))
    print("nl lengths counter:")
    print(nl_counters)
    print("name lengths counter:")
    print(name_counters)
    f = open('data/desc.txt', 'a')
    f.write('nl lengths counter of ' + identifier + ':' + str(nl_counters) + '\n')
    f.write('name lengths counter of ' + identifier + ':' + str(name_counters) + '\n')
    f.close()

    f = open('data/' + identifier + '_desc_tokens.json', 'w')
    for nl_name in nl_names:
        f.write(json.dumps(nl_name) + '\n')

    return nl_names


def gen_vocab(identifier):
    filename = 'data/' + identifier + '_desc_tokens.json'
    with open(filename, 'r') as f:
        lines = f.readlines()
    name_vocab = []
    nl_vocab = []

    for line in lines:
        line = json.loads(line)
        nl = line['nl']
        name = line['name']
        name_vocab.extend(name)
        nl_vocab.extend(nl)

    name_count = dict(collections.Counter(name_vocab))
    nl_count = dict(collections.Counter(nl_vocab))
    name_del = []
    nl_del = []
    for name in name_count.keys():
        if name_count[name] < 5:
            name_del.append(name)
    for nl in nl_count.keys():
        if nl_count[nl] < 5:
            nl_del.append(nl)
    name_vocab = list(set(name_vocab))
    nl_vocab = list(set(nl_vocab))

    for nl in nl_del:
        nl_vocab.remove(nl)
    for name in name_del:
        name_del.remove(name)

    f = open('data/' + identifier + '_name_vocab.json', 'w+')
    f.write(json.dumps(name_vocab))
    f.close()

    f = open('data/' + identifier + '_nl_vocab.json', 'w+')
    f.write(json.dumps(nl_vocab))
    f.close()

    f = open('data/desc.txt', 'a')
    f.write('The vocabulary length of ' + identifier + ' nl:' + str(len(nl_vocab)) + '\n')
    f.write('The vocabulary length of ' + identifier + ' name:' + str(len(name_vocab)) + '\n')
    f.close()
    print("The vocabulary length of nl:", len(nl_vocab))
    print("The vocabulary length of name:", len(name_vocab))

    return nl_vocab, name_vocab


if __name__ == '__main__':
    identifier = sys.argv[1]
    print("***********Tokenize***********")
    tokenize(identifier)
    print("***********Building vocabulary***********")
    gen_vocab(identifier)
    print('Done')



