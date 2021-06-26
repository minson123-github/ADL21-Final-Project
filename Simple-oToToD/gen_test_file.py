import os
import json
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

for mode in ['test_seen', 'test_unseen']:
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
            glob_context += ' <|endofcontext|>'
            for service in conv['services']:
                context = glob_context
                context += ' <|servicedetail|> '
                context += schema[service]
                context += ' <|endofservicedetail|>'
                for slot in schema_slots[service]:
                    cur_context = context + ' <|slotdetail|> '
                    cur_context += schema[f"{service}-{slot}"]
                    cur_context += ' <|endofslotdetail|>'
                    cur_context += ' <|belief|> '
                    cur_context += f"{service}-{slot}"
                    cur_context += ' <|endofbelief|>'
                    processed.append(cur_context)

    open(f'{mode}.txt', "w").write("\n".join(processed))
