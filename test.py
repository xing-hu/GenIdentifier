import json

f = open('data/para_desc.json', 'r')
lines = f.readlines()
nl_names = []
nl_seen = set()
name_seen = set()
for line in lines:
    line = json.loads(line)
    t_dict = {'nl': line['nl'], 'name': line['name']}
    if t_dict['nl'] not in nl_seen and t_dict['name'] not in name_seen:
        nl_seen.add(t_dict['nl'])
        name_seen.add(t_dict['name'])
        nl_names.append(t_dict)
print "origin: ", len(lines)
print "target: ", len(nl_names)
f.close()

f = open('data/para_desc1.json', 'w')
for nl_name in nl_names:
    f.write(json.dumps(nl_name) + '\n')
f.close()