import json
import os
from tqdm.auto import tqdm

PATH = './data/data/'
for folder in ['train', "dev", "test"]:
	path = PATH + folder + '/'
	fns = os.listdir(path)
	for fn in tqdm(fns, desc=folder):
		with open(path + fn, "r", encoding='utf-8') as f:
			data = json.load(f)
		for i in range(len(data)):
			for j in range(len(data[i]["turns"])):
				if "frames" not in data[i]["turns"][j]:
					data[i]["turns"][j]["frames"] = []
				if "beginning" not in data[i]["turns"][j]:
					data[i]["turns"][j]["beginning"] = []
				if "end" not in data[i]["turns"][j]:
					data[i]["turns"][j]["end"] = []
		with open(path + fn, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=1)
