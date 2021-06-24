import os
import json
from tqdm import tqdm
from datasets import Dataset

# get domain slot pairs and description from schema file
def get_domain_slot(schema_path):
	with open(schema_path, 'r') as fp:
		schema = json.load(fp)
	domain_slot = {}
	for info in schema:
		domain = info['service_name']
		slots = []
		for slot in info['slots']:
			slot_name = slot['name']
			description = slot['description']
			slots.append((slot_name, description))
		domain_slot[domain] = {'description': info['description'], 'slots': slots}
	return domain_slot

def read_data(data_dir, schema_info, is_test):
	data_list = []
	files = os.listdir(data_dir)
	for file_name in files:
		file_path = os.path.join(data_dir, file_name)
		with open(file_path, 'r') as fp:
			all_info = json.load(fp)
		for info in all_info:
			if is_test: # file to predict final slot and value
				data = {'turns': []}
				data['dialogue_id'] = info['dialogue_id']
				data['domains'] = info['services']
				turn_info = dict()
				for turn in info['turns']:
					if turn['speaker'] == 'USER':
						turn_info = {'usr': turn['utterance']}
					else:
						turn_info['sys'] = turn['utterance']
					data['turns'].append(turn_info)
				data_list.append(data)
			else:
				data = {'turns': []}
				data['dialogue_id'] = info['dialogue_id']
				data['domains'] = info['services']
				domain_slot = set()
				turn_info = dict()
				domain_slot_pairs = {}
				for turn in info['turns']:
					if turn['speaker'] == 'USER':
						turn_info = {'usr': turn['utterance'], 'slot_values': {}}
						for frame in turn['frames']:
							domain = frame['service']
							for k, v in frame['state']['slot_values'].items():
								slot_name = k
								slot_value = v[0] if isinstance(v, list) else v
								if slot_name[:len(domain) + 1] == domain + '-':
									slot_name = slot_name[len(domain) + 1:]
								domain_slot.add((domain, slot_name))
								turn_info['slot_values'][(domain, slot_name)] = slot_value
					else:
						turn_info['sys'] = turn['utterance']
						for dialogue_domain in info['services']:
							for slot, _ in schema_info[dialogue_domain]['slots']:
								if (dialogue_domain, slot) not in turn_info['slot_values']:
									turn_info['slot_values'][(dialogue_domain, slot)] = 'none'
						data['turns'].append(turn_info)
				domain_slot = [(domain, slot) for domain, slot in domain_slot]
				data['domain_slot'] = domain_slot
				data_list.append(data)

	return data_list

def get_dict_list(args, raw_data, tokenizer, schema_info):
	results = {'input_sentences': [], 'output_sentences': []}
	slots_descr = {}
	for domain in schema_info.keys():
		for slot in schema_info[domain]['slots']:
			slots_descr[(domain, slot[0])] = slot[1]
	total = 0
	for data in raw_data:
		for turn in data['turns']:
			for (domain, slot_name), slot_value in turn['slot_values'].items():
				total += 1

	bar = tqdm(total=total)
	
	for data in raw_data:
		dialogue_history = ''
		for turn in data['turns']:
			dialogue_history += ('System: ' + turn['sys'] + 'User: ' + turn['usr'])
			for (domain, slot_name), slot_value in turn['slot_values'].items():
				domain_descr = schema_info[domain]['description']
				slot_descr = slots_descr.get((domain, slot_name), 'none')
				if args['use_descr']:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_descr} of the {domain_descr}'
				else:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_name} of the {domain}'
				output_sentence = slot_value + f" {tokenizer.eos_token}"
				results['input_sentences'].append(input_sentence)
				results['output_sentences'].append(output_sentence)
				bar.update(1)
	
	return results
			
def get_train_dataset(args, tokenizer):
	domain_slots = get_domain_slot(args['schema_dir'])
	train_data = read_data(args['train_dir'], domain_slots, False)
	eval_data = read_data(args['eval_dir'], domain_slots, False)
	train_dict = get_dict_list(args, train_data, tokenizer, domain_slots)
	eval_dict = get_dict_list(args, eval_data, tokenizer, domain_slots)
	'''
	for k in train_dict:
		train_dict[k] = train_dict[k][:100000]
	for k in eval_dict:
		eval_dict[k] = eval_dict[k][:100000]
	'''
	train_dataset = Dataset.from_dict(train_dict)
	eval_dataset = Dataset.from_dict(eval_dict)
	return train_dataset, eval_dataset
