import os
import json
import random
from tqdm.auto import tqdm

PATH = 'data-0625/'


schema = {}
schema_slots = {}
raw_schema = json.load(open(PATH + 'schema.json'))
for service in raw_schema:
    schema[service['service_name']] = service['description']
    schema_slots[service['service_name']] = []
    for slot in service['slots']:
        schema[f"{service['service_name']}-{slot['name']}"] = slot['description']
        schema_slots[service['service_name']].append(slot['name'])

for mode in ["train", "dev"]:
    processed = []
    dials = map(lambda dial: f'{PATH}{mode}/{dial}', sorted(os.listdir(PATH + mode)))
    for dial in tqdm(list(dials), mode):
        data = json.load(open(dial, "r"))
        for conv in data:
            glob_context = '<|context|>'
            for turn in conv['turns']:
                if turn['speaker'] == 'SYSTEM':
                    glob_context += ' <|system|> '
                elif turn['speaker'] == 'USER':
                    glob_context += ' <|user|> '
                else:
                    raise Exception
                glob_context += turn['utterance']
                for frame in turn['frames']:
                    context = glob_context + ' <|endofcontext|>'
                    context += ' <|servicedetail|> '
                    context += schema[frame['service']]
                    context += ' <|endofservicedetail|>'
                    if 'state' not in frame:
                        continue
                    for slot in schema_slots[frame['service']]:
                        cur_context = context + ' <|slotdetail|> '
                        cur_context += schema[f"{frame['service']}-{slot}"]
                        cur_context += ' <|endofslotdetail|>'
                        cur_context += ' <|belief|> '
                        cur_context += f"{frame['service']}-{slot}"
                        cur_context += ' <|endofbelief|>'
                        cur_context += ' <|beliefval|> '
                        if slot in frame['state']['slot_values'].keys():
                            cur_context += frame['state']['slot_values'][slot][0]
                        else:
                            cur_context += '<|emptyslot|>'
                        cur_context += ' <|endofbeliefval|>'
                        processed.append(cur_context)
    random.shuffle(processed)
    open(f'{mode}.txt', "w").write("\n".join(processed))
