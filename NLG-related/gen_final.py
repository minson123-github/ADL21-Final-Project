import os
import json

with open("./acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt", "r") as f:
    model_outputs = f.read().strip().split("\n")
    for i in range(len(model_outputs)):
        model_outputs[i] = model_outputs[i].split()
        for j in range(len(model_outputs[i])):
            model_outputs[i][j] = float(model_outputs[i][j])
        assert(len(model_outputs[i]) == 3)
    print(len(model_outputs))


with open("./lm.output.test.cc.txt", "r") as f:
    data = f.read()
data = data.split("[TransformerGenerator]:")[1:]
for i in range(len(data)):
    data[i] = data[i].split("\n")[0].strip()
data_cc = data

model_outputs = model_outputs[-len(data_cc):]

fns = os.listdir('data/data/test/')
fns.sort()

res = {}

k = 0

for fn in fns:
    with open(f'data/data/test/{fn}') as f:
        c = json.load(f)
    for d in c:
        res[d['dialogue_id']] = {}
        for i in range(1, len(d['turns']), 2):
            assert(d['turns'][i]['speaker'] == "SYSTEM")
            res[d['dialogue_id']][d['turns'][i]['turn_id']] = {"start": "", "end": "", "mod": ""}
            assert(len(model_outputs[k]) == 3)
            o = 0
            for j in range(1, 3):
                if model_outputs[i][j] > model_outputs[i][o]:
                    o = j
            if o == 1:
                res[d['dialogue_id']][d['turns'][i]['turn_id']]['start'] = data_cc[k]
            elif o == 2:
                res[d['dialogue_id']][d['turns'][i]['turn_id']]['end'] = data_cc[k]
            k += 1
with open("nlg_output.json", "w") as f:
    json.dump(res, f)
