import os
import json

DATA_PATH = '../data-0614/data-0614/test_seen/'
CHITCHAT_PATH = './electra_nlg_output.json'

g = json.load(open(CHITCHAT_PATH))

o = {}
fns = os.listdir(DATA_PATH)
for fn in sorted(fns):
	ds = json.load(open(DATA_PATH + fn))
	for d in ds:
		o[d['dialogue_id']] = {}
		for t in d['turns']:
			o[d['dialogue_id']][t['turn_id']] = {'speaker': t['speaker'], 'utterance': t['utterance']}

def show(name):
    for k in sorted(o[name].keys()):
        sk = str(k)
        if sk not in g[name].keys():
            print(f"{o[name][k]['speaker']}: {o[name][k]['utterance']}")
        else:
            if len(g[name][sk]['start']) > 0:
                print(f"{o[name][k]['speaker']} (chit-chat): {g[name][sk]['start']}")
            if len(g[name][sk]['mod']) > 0:
                print(f"{o[name][k]['speaker']} (modified): {g[name][sk]['mod']}")
            else:
                print(f"{o[name][k]['speaker']}: {o[name][k]['utterance']}")
            if len(g[name][sk]['end']) > 0:
                print(f"{o[name][k]['speaker']} (chit-chat): {g[name][sk]['end']}")

print('choice:', ', '.join(o.keys()))
name = input("which: ")
show(name)
