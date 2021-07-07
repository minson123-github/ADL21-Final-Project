import os
import json

res = os.listdir('./results/')
res.sort()

model_outputs = []
for t in res:
    with open(f"./results/{t}/is_test_true_eval_logits.txt", "r") as f:
        model_outputs_ = f.read().strip().split("\n")
        for i in range(len(model_outputs_)):
            model_outputs_[i] = model_outputs_[i].split()
            for j in range(len(model_outputs_[i])):
                model_outputs_[i][j] = float(model_outputs_[i][j])
            assert(len(model_outputs_[i]) == 3)
        model_outputs.append(model_outputs_)

data_cc = []
for t in res:
    with open(f"./results/{t}/lm.output.test.cc.txt", "r") as f:
        data = f.read()
    data = data.split("[TransformerGenerator]:")[1:]
    for i in range(len(data)):
        data[i] = data[i].split("\n")[0].strip()
    data_cc.append(data)

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
            t, o = 0, 0
            for s in range(len(model_outputs)):
                for j in range(1, 3):
                    if model_outputs[s][i][j] > model_outputs[s][i][o]:
                        o = j
            if o == 1:
                res[d['dialogue_id']][d['turns'][i]['turn_id']]['start'] = data_cc[s][k]
            elif o == 2:
                res[d['dialogue_id']][d['turns'][i]['turn_id']]['end'] = data_cc[s][k]
            k += 1
with open("nlg_output.json", "w") as f:
    json.dump(res, f)
