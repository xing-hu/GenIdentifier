import json
import string
import re

import nltk
import collections

f = open('data/method_desc.json', 'r')
lines = f.readlines()
nl_methods = []
for line in lines:
    line = json.loads(line)
    nl = line['nl']
    name = line['method_name']
    nl_method = {'name': name, 'nl': nl}
    nl_methods.append(nl_method)

f.close()
f = open('data/method_desc.json', 'w')
for nl_method in nl_methods:
    f.write(json.dumps(nl_method) + '\n')
f.close()
print(len(nl_methods))



