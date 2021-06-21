import os
import sys
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
	"-schema",
	"--schema_file",
	type=str,
	help="The file with schema info."
)
parser.add_argument(
	"-train", 
	"--train_data_dir", 
	type=str, 
	help="The directory with training data."
)
parser.add_argument(
	"-eval", 
	"--eval_data_dir", 
	type=str, 
	help="The directory with evaluate data."
)
parser.add_argument(
	"-output", 
	"--output_dir", 
	type=str, 
	help="The directory to put multiwoz coversion result."
)
args = parser.parse_args()

schema_path = args.schema_file
train_data_dir = args.train_data_dir
eval_data_dir = args.eval_data_dir
target_dir = args.output_dir

with open(schema_path, 'r') as fp:
	schema = json.load(fp)

train_files = os.listdir(train_data_dir)
eval_files = os.listdir(eval_data_dir)

# domain extract
domains = set()
for service_info in schema:
	domains.add(service_info['service_name'])

print(f'domain numbers: {len(domains)}')
print(f'domains:\n{[domain for domain in domains]}')

# slot extract
slots = dict()
for service_info in schema:
	domain = service_info['service_name']
	for slot in service_info['slots']:
		slot_name = slot['name']
		if slot_name.split('-')[0] != domain:
			slot_name = domain + '-' + slot_name
		try:
			slot_values = slot['possible_values']
			if slot_name not in slots:
				slots.update({slot_name: slot_values})
			else:
				slots[slot_name] += slot_values
		except KeyError:
			pass

with open('ontology_2_0.json', 'r') as fp:
	slots_2_0 = json.load(fp)
	for slot_name, slot_values in slots_2_0.items():
		slot = slot_name.replace(' ', '')
		if slot not in slots:
			slots.update({slot: slot_values})
		for value in slot_values:
			if value not in slots[slot]:
				slots[slot].append(value)

print(f'slot numbers: {len(slots)}')
print(f'slots:\n{slots.keys()}')

# extract dialogue acts
dialogue_acts = dict()
val_dialogue_id = []

for file_name in train_files:
	file_path = os.path.join(train_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	for dialogue in dialogues:
		dialogue_id = dialogue['dialogue_id']
		acts = dict()
		for i, turn in enumerate(dialogue['turns']):
			act = dict()
			for frame in turn['frames']:
				if 'actions' not in frame:
					continue
				for action in frame['actions']:
					slot = 'none'
					slot_value = 'none'
					if 'slot' in action:
						slot = action['slot']
					if 'values' in action and len(action['values']):
						slot_value = action['values'][0]
					elif 'value' in action:
						slot_value = action['value']
					if action['act'] not in act:
						act.update({action['act']: [[slot, slot_value]]})
					else:
						act[action['act']].append([slot, slot_value])
			if len(act) == 0:
				acts.update({str(i + 1): 'No Annotation'})
			else:
				acts.update({str(i + 1): act})
		dialogue_acts.update({dialogue_id: acts})

# extract dialogue acts and valListFile
for file_name in eval_files:
	file_path = os.path.join(eval_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	for dialogue in dialogues:
		dialogue_id = dialogue['dialogue_id']
		val_dialogue_id.append(dialogue_id)
		acts = dict()
		for i, turn in enumerate(dialogue['turns']):
			act = dict()
			for frame in turn['frames']:
				if 'actions' not in frame:
					continue
				for action in frame['actions']:
					slot = 'none'
					slot_value = 'none'
					if 'slot' in action:
						slot = action['slot']
					if 'values' in action and len(action['values']):
						slot_value = action['values'][0]
					elif 'value' in action:
						slot_value = action['value']
					if action['act'] not in act:
						act.update({action['act']: [[slot, slot_value]]})
					else:
						act[action['act']].append([slot, slot_value])
			if len(act) == 0:
				acts.update({str(i + 1): 'No Annotation'})
			else:
				acts.update({str(i + 1): act})
		dialogue_acts.update({dialogue_id: acts})
'''
data_fp = os.path.join(target_dir, 'data.json')
act_fp = os.path.join(target_dir, 'dialogue_acts.json')
ontology_fp = os.path.join(target_dir, 'ontology.json')
'''
# create support informations
with open(os.path.join(target_dir, 'dialogue_acts.json'), 'w') as fp:
	json.dump(dialogue_acts, fp, indent=4)

with open(os.path.join(target_dir, 'ontology.json'), 'w') as fp:
	json.dump(slots, fp, indent=4)

with open(os.path.join(target_dir, 'valListFile.json'), 'w') as fp:
	for dialogue_id in val_dialogue_id:
		fp.write(dialogue_id + '\n')

# extract data.json
total = 0
for file_name in train_files:
	file_path = os.path.join(train_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	total += len(dialogues)
	'''
	for dialogue in dialogues:
		if 'SNG0350' in dialogue['dialogue_id']:
			print(json.dumps(dialogue, indent=4))
	'''

for file_name in eval_files:
	file_path = os.path.join(eval_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	total += len(dialogues)
data = {}
print('start to extract data.\n', flush=True)
bar = tqdm(total = total)
for file_name in train_files:
	file_path = os.path.join(train_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	for dialogue in dialogues:
		dialogue_id = dialogue['dialogue_id']
		if len(dialogue['services']) == 0:
			bar.update(1)
			continue
		goal_dict = {domain: {} for domain in dialogue['services']}
		goal_dict.update({'message': {}})
		goal_dict.update({'topic': {}})
		log_list = []
		metadata = {}
		for turn in dialogue['turns']:
			turn_dict = {'text': turn['utterance']}
			if turn['speaker'] == 'USER':
				metadata = {domain: {'book': {'booked': []}, 'semi': {}} for domain in domains}
			for frame in turn['frames']:
				domain = frame['service']
				for slot in frame['slots']:
					slot_name = slot['slot']
					if 'copy_from' in slot:
						slot_name = slot['copy_from']
					slot_value = ''
					if 'value' in slot:
						if isinstance(slot['value'], list):
							slot_value = slot['value'][0]
						else:
							slot_value = slot['value']
					else:
						slot_value = turn['utterance'][slot['start']: slot['exclusive_end']]
					is_book_info = False
					try:
						if slot_name.index(domain) == 0:
							mid_index = slot_name.index('-')
							if mid_index + 5 <= len(slot_name) and slot_name.index('book') == mid_index + 1:
								is_book_info = True
					except ValueError:
						pass
					if is_book_info:
						mid_index = slot_name.index('-')
						book_info = slot_name[mid_index + 5:]
						metadata[domain]['book'][book_info] = slot_value
					else:
						if domain + '-' in slot_name and slot_name.index(domain + '-') == 0:
							mid_index = slot_name.index('-')
							metadata[domain]['semi'][slot_name[mid_index + 1:]] = slot_value
						else:
							metadata[domain]['semi'][slot_name] = slot_value
			
			for frame in turn['frames']:
				domain = frame['service']
				if 'state' not in frame:
					continue
				for slot in frame['state']['slot_values'].items():
					slot_name = slot[0]
					slot_value = slot[1][0]
					is_book_info = False
					try:
						if slot_name.index(domain) == 0:
							mid_index = slot_name.index('-')
							if mid_index + 5 <= len(slot_name) and slot_name.index('book') == mid_index + 1:
								is_book_info = True
					except ValueError:
						pass
					if is_book_info:
						mid_index = slot_name.index('-')
						book_info = slot_name[mid_index + 5:]
						metadata[domain]['book'][book_info] = slot_value
					else:
						if domain + '-' in slot_name and slot_name.index(domain + '-') == 0:
							mid_index = slot_name.index('-')
							metadata[domain]['semi'][slot_name[mid_index + 1:]] = slot_value
						else:
							metadata[domain]['semi'][slot_name] = slot_value
					

			if turn['speaker'] == 'USER':
				turn_dict['metadata'] = {}
			else:
				turn_dict['metadata'] = metadata
			log_list.append(turn_dict)
		data[dialogue_id] = {'goal': goal_dict, 'log': log_list}
		bar.update(1)

for file_name in eval_files:
	file_path = os.path.join(eval_data_dir, file_name)
	with open(file_path, 'r') as fp:
		dialogues = json.load(fp)
	for dialogue in dialogues:
		dialogue_id = dialogue['dialogue_id']
		if len(dialogue['services']) == 0:
			bar.update(1)
			continue
		goal_dict = {domain: {} for domain in dialogue['services']}
		goal_dict.update({'message': {}})
		goal_dict.update({'topic': {}})
		log_list = []
		metadata = {}
		for turn in dialogue['turns']:
			turn_dict = {'text': turn['utterance']}
			if turn['speaker'] == 'USER':
				metadata = {domain: {'book': {'booked': []}, 'semi': {}} for domain in domains}
			for frame in turn['frames']:
				domain = frame['service']
				for slot in frame['slots']:
					slot_name = slot['slot']
					if 'copy_from' in slot:
						slot_name = slot['copy_from']
					slot_value = ''
					if 'value' in slot:
						if isinstance(slot['value'], list):
							slot_value = slot['value'][0]
						else:
							slot_value = slot['value']
					else:
						slot_value = turn['utterance'][slot['start']: slot['exclusive_end']]
					is_book_info = False
					try:
						if slot_name.index(domain) == 0:
							mid_index = slot_name.index('-')
							if mid_index + 5 <= len(slot_name) and slot_name.index('book') == mid_index + 1:
								is_book_info = True
					except ValueError:
						pass
					if is_book_info:
						mid_index = slot_name.index('-')
						book_info = slot_name[mid_index + 5:]
						metadata[domain]['book'][book_info] = slot_value
					else:
						if domain + '-' in slot_name and slot_name.index(domain + '-') == 0:
							mid_index = slot_name.index('-')
							metadata[domain]['semi'][slot_name[mid_index + 1:]] = slot_value
						else:
							metadata[domain]['semi'][slot_name] = slot_value
			
			for frame in turn['frames']:
				domain = frame['service']
				if 'state' not in frame:
					continue
				for slot in frame['state']['slot_values'].items():
					slot_name = slot[0]
					slot_value = slot[1][0]
					is_book_info = False
					try:
						if slot_name.index(domain) == 0:
							mid_index = slot_name.index('-')
							if mid_index + 5 <= len(slot_name) and slot_name.index('book') == mid_index + 1:
								is_book_info = True
					except ValueError:
						pass
					if is_book_info:
						mid_index = slot_name.index('-')
						book_info = slot_name[mid_index + 5:]
						metadata[domain]['book'][book_info] = slot_value
					else:
						if domain + '-' in slot_name and slot_name.index(domain + '-') == 0:
							mid_index = slot_name.index('-')
							metadata[domain]['semi'][slot_name[mid_index + 1:]] = slot_value
						else:
							metadata[domain]['semi'][slot_name] = slot_value
					

			if turn['speaker'] == 'USER':
				turn_dict['metadata'] = {}
			else:
				turn_dict['metadata'] = metadata
			log_list.append(turn_dict)
		data[dialogue_id] = {'goal': goal_dict, 'log': log_list}
		bar.update(1)

with open(os.path.join(target_dir, 'data.json'), 'w') as fp:
	json.dump(data, fp, indent=4)
