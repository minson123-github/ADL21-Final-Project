import os
import json

with open('test.inference.gpt2_10epoch_1e-3_fp16.json') as f:
	g = json.load(f)

fns = os.listdir('data/data/test/')
fns.sort()
k = 0
for fn in fns:
	with open(f'data/data/test/{fn}', encoding='utf-8') as f:
		c = json.load(f)
    for u in c:
		for a in range(1, len(u['turns']), 2):
			g[k] = g[k].split("<|response|>")[0] + "<|response|>" + u['turns'][a]['utterance']
			k += 1

with open('test.inference.gpt2_10epoch_1e-3_fp16.json', 'w') as f:
	json.dump(g, f)
