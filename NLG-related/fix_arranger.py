import os
import json


for s in ['test', 'dev']:
	g = []
	fns = os.listdir(f'data/data/{s}/')
	fns.sort()
	k = 0
	for fn in fns:
		with open(f'data/data/{s}/{fn}', encoding='utf-8') as f:
			c = json.load(f)
		for u in c:
			for a in range(1, len(u['turns']), 2):
				g.append("<|response|>" + u['turns'][a]['utterance'])
				k += 1

	with open(s+'.inference.gpt2_10epoch_1e-3_fp16.json', 'w') as f:
		json.dump(g, f)
