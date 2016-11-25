import os
import json
import math
import pickle
import sqlite3
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle

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


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#


def with_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, p)

NAME_DICTIONARY_PATH = 'data/method_name_vocab.json'
NL_DICTIONARY_PATH = 'data/method_nl_vocab.json'
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'

buckets = [
    (15, 2),
    (20, 3),
    (25, 4),
    (30, 15)
]

def time(s):
    ret = ''
    if s >= 60 * 60:
        h = math.floor(s / (60 * 60))
        ret += '{}h'.format(h)
        s -= h * 60 * 60
    if s >= 60:
        m = math.floor(s / 60)
        ret += '{}m'.format(m)
        s -= m * 60
    if s >= 1:
        s = math.floor(s)
        ret += '{}s'.format(s)
    return ret

def load_dictionary():
    with open(with_path(NL_DICTIONARY_PATH), 'r') as fp:
        ask_dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
        ask_index_word = OrderedDict()
        ask_word_index = OrderedDict()
        for index, word in enumerate(ask_dictionary):
            ask_index_word[index] = word
            ask_word_index[word] = index
        ask_dim = len(ask_dictionary)
    with open(with_path(NAME_DICTIONARY_PATH), 'r') as fp:
        answer_dictionary = [EOS, UNK, PAD, GO] + json.load(fp)
        answer_index_word = OrderedDict()
        answer_word_index = OrderedDict()
        for index, word in enumerate(answer_dictionary):
            answer_index_word[index] = word
            answer_word_index[word] = index
        answer_dim = len(answer_dictionary)
    return ask_dim, ask_dictionary, ask_index_word, ask_word_index, \
           answer_dim, answer_dictionary, answer_index_word, answer_word_index

def save_model(sess, name='model.ckpt'):
    if not os.path.exists('model'):
        os.makedirs('model')
    saver = tf.train.Saver()
    saver.save(sess, with_path('model/' + name))

def load_model(sess, name='model.ckpt'):
    saver = tf.train.Saver()
    saver.restore(sess, with_path('model/' + name))

ask_dim, ask_dictionary, ask_index_word, ask_word_index, \
answer_dim, answer_dictionary, answer_index_word, answer_word_index = load_dictionary()

print('ask_dim: ', ask_dim)
print('answer_dim: ', answer_dim)

ASK_EOS_ID = ask_word_index[EOS]
ASK_UNK_ID = ask_word_index[UNK]
ASK_PAD_ID = ask_word_index[PAD]
ASK_GO_ID = ask_word_index[GO]

ANSWER_EOS_ID = answer_word_index[EOS]
ANSWER_UNK_ID = answer_word_index[UNK]
ANSWER_PAD_ID = answer_word_index[PAD]
ANSWER_GO_ID = answer_word_index[GO]

class BucketData(object):

    def __init__(self, buckets_dir, encoder_size, decoder_size):
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.name = 'bucket_%d_%d.db' % (encoder_size, decoder_size)
        self.path = os.path.join(buckets_dir, self.name)
        self.conn = sqlite3.connect(self.path)
        self.cur = self.conn.cursor()
        sql = '''SELECT MAX(ROWID) FROM conversation;'''
        self.size = self.cur.execute(sql).fetchall()[0][0]

    def random(self):
        while True:
            # 选择一个[1, MAX(ROWID)]中的整数，读取这一行
            rowid = np.random.randint(1, self.size + 1)
            sql = '''
            SELECT ask, answer FROM conversation
            WHERE ROWID = {};
            '''.format(rowid)
            ret = self.cur.execute(sql).fetchall()
            if len(ret) == 1:
                ask, answer = ret[0]
                if ask is not None and answer is not None:
                    return ask, answer

def read_bucket_dbs(buckets_dir):
    ret = []
    for encoder_size, decoder_size in buckets:
        bucket_data = BucketData(buckets_dir, encoder_size, decoder_size)
        ret.append(bucket_data)
    return ret

def sentence_indice_ask(sentence):
    words = sentence.split(' ')
    ret = []
    for  word in words:
        if word in ask_word_index:
            ret.append(ask_word_index[word])
        else:
            ret.append(ask_word_index[UNK])
    return ret

def sentence_indice_answer(sentence):
    words = sentence.split(' ')
    ret = []
    for  word in words:
        if word in answer_word_index:
            ret.append(answer_word_index[word])
        else:
            ret.append(answer_word_index[UNK])
    return ret

def indice_sentence_ask(indice):
    words = indice.split(' ')
    ret = []
    for index in words:
        word = ask_index_word[index]
        if word == EOS:
            break
        if word != UNK and word != GO and word != PAD:
            ret.append(word)
    return ''.join(ret)

def indice_sentence_answer(indice):
    words = indice.split(' ')
    ret = []
    for index in words:
        word = answer_index_word[index]
        if word == EOS:
            break
        if word != UNK and word != GO and word != PAD:
            ret.append(word)
    return ''.join(ret)

def vector_sentence_ask(vector):
    return indice_sentence_ask(vector.argmax(axis=1))

def vector_sentence_answer(vector):
    return indice_sentence_answer(vector.argmax(axis=1))

def generate_bucket_dbs(
        input_dir,
        output_dir,
        buckets,
        tolerate_unk=2
    ):
    count = 0
    pool = {}
    def _get_conn(key):
        if key not in pool:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            name = 'bucket_%d_%d.db' % key
            path = os.path.join(output_dir, name)
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS conversation (ask text, answer text);""")
            conn.commit()
            pool[key] = (conn, cur)
        return pool[key]
    all_inserted = {}
    for encoder_size, decoder_size in buckets:
        key = (encoder_size, decoder_size)
        all_inserted[key] = 0
    # 从input_dir列出数据库列表
    db_paths = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in (x for x in sorted(filenames) if x.endswith('.db')):
            db_path = os.path.join(dirpath, filename)
            db_paths.append(db_path)
    # 对数据库列表中的数据库挨个提取
    for db_path in db_paths:
        print('读取数据库: {}'.format(db_path))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        def is_valid_ask(s):
            words = s.split(' ')
            unk = 0
            for w in words:
                if w not in ask_word_index:
                    unk += 1
            if unk > tolerate_unk:
                return False
            return True
        def is_valid_answer(s):
            words = s.split(' ')
            unk = 0
            for w in words:
                if w not in answer_word_index:
                    unk += 1
            if unk > tolerate_unk:
                return False
            return True
        def sen_length(sen):
            return len(sen.split(' '))
        # 读取最大的rowid，如果rowid是连续的，结果就是里面的数据条数
        # 比SELECT COUNT(1)要快
        total = c.execute('''SELECT MAX(ROWID) FROM conversation;''').fetchall()[0][0]
        ret = c.execute('''SELECT ask, answer FROM conversation;''')
        wait_insert = []
        def _insert(wait_insert):
            if len(wait_insert) > 0:
                for encoder_size, decoder_size, ask, answer in wait_insert:
                    key = (encoder_size, decoder_size)
                    conn, cur = _get_conn(key)
                    cur.execute("""
                    INSERT INTO conversation (ask, answer) VALUES ('{}', '{}');
                    """.format(ask.replace("'", "''"), answer.replace("'", "''")))
                    all_inserted[key] += 1
                for conn, _ in pool.values():
                    conn.commit()
                wait_insert = []
            return wait_insert
        for ask, answer in tqdm(ret, total=total):
            if is_valid_ask(ask) and is_valid_answer(answer):
                for i in range(len(buckets)):
                    encoder_size, decoder_size = buckets[i]
                    #print(sen_length(ask), '/' , encoder_size, " ", sen_length(answer), '/', decoder_size)
                    if sen_length(ask) <= encoder_size and sen_length(answer) < decoder_size:
                        #print('insert')
                        count = count + 1
                        wait_insert.append((encoder_size, decoder_size, ask, answer))
                        if len(wait_insert) > 1000000:
                            wait_insert = _insert(wait_insert)
                        break
    wait_insert = _insert(wait_insert)
    print(count)
    return all_inserted

if __name__ == '__main__':
    print('generate bucket dbs')
    all_inserted = generate_bucket_dbs('./db', './bucket_dbs', buckets, 2)
    for key, inserted_count in all_inserted.items():
        print(key)
        print(inserted_count)
    print('done')



