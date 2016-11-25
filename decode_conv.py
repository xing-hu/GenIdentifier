#!/usr/bin/env python3

import os
import re
import sys
import sqlite3
import json
import random
from collections import Counter

from tqdm import tqdm


def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def valid(a, max_len=0):
    if len(a) > 0 and not contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:
            return True
    return False

def insert(a, b, cur):
    cur.execute("""
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_test(a, b, cur):
    cur.execute("""
    INSERT INTO conversation_test (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_if(question, answer, cur, input_len=500, output_len=500, des = 'train'):
    if valid(question, input_len) and valid(answer, output_len):
        if des == 'train':
            insert(question, answer, cur)
        else:
            insert_test(question, answer, cur)
        return 1
    return 0

def main(file_path):
    f = open(file_path)
    lines = f.read().split('\n')

    print('一共读取 %d 行数据' % len(lines))

    db = 'db/conversation.db'
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation
        (ask text, answer text);
        """)
    conn.commit()

    cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_test
            (ask text, answer text);
            """)
    conn.commit()

    words = Counter()
    a = ''
    b = ''
    inserted = 0

    test_pairs = []

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        try:
            pair = json.loads(line)
        except json.decoder.JSONDecodeError:
            continue
        y = random.random()
        if y >= 0.8:
            test_pairs.append(pair)
            continue
        nls = pair['nl']
        names = pair['name']
        if len(nls) < 2 or len(names) == 0:
            continue
        nl = nls[0]
        for n in nls[1:]:
            nl = nl + " " + n
        name = names[0]
        for n in names[1:]:
            name = name + " " + n
        words.update(Counter(name))
        words.update(Counter(nl))
        ask = nl
        answer = name
        inserted += insert_if(ask, answer, cur)
        # 批量提交
        if inserted != 0 and inserted % 50000 == 0:
            conn.commit()

    for index, pair in tqdm(enumerate(test_pairs), total=len(test_pairs)):
        nls = pair['nl']
        names = pair['name']
        if len(nls) < 2 or len(names) == 0:
            continue
        nl = nls[0]
        for n in nls[1:]:
            nl = nl + " " + n
        name = names[0]
        for n in names[1:]:
            name = name + " " + n
        words.update(Counter(name))
        words.update(Counter(nl))
        ask = nl
        answer = name
        inserted += insert_if(ask, answer, cur, 500, 500, 'test')
        # 批量提交
        if inserted != 0 and inserted % 50000 == 0:
            conn.commit()
    conn.commit()

if __name__ == '__main__':
    file_path = 'data/method_desc_tokens.json'
    if not os.path.exists(file_path):
        print('文件 {} 不存在'.format(file_path))
    else:
        main(file_path)
