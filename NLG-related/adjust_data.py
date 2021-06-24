import json
import os

PATH = './data/data/train/'

fns = os.listdir(PATH)
for fn in fns:
	with open(PATH + fn, "r", encoding='utf-8') as f:
		data = json.load(f)
	for i in range(len(data)):
		for j in range(len(data[i]["turns"])):
			if "frames" not in data[i]["turns"][j]:
				data[i]["turns"][j]["frames"] = []
			if "beginning" not in data[i]["turns"][j]:
				data[i]["turns"][j]["beginning"] = []
			if "end" not in data[i]["turns"][j]:
				data[i]["turns"][j]["end"] = []
	with open(PATH + fn, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=1)
