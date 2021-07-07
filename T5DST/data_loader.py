import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from functools import partial
import pandas as pd
import pickle

class DSTDataset(Dataset):
	def __init__(self, data, args):
		self.input_tokens = data['input_tokens']
		if 'output_tokens' in data:
			self.ID = None
			self.output_tokens = data['output_tokens']
			self.domain_slot = None
		else:
			self.output_tokens = None
			self.ID = data['ID']
			self.domain_slot = data['domain-slot']
		self.args = args
	
	def __getitem__(self, index):
		if self.output_tokens != None:
			return self.input_tokens[index], self.output_tokens[index]
		return self.input_tokens[index], self.ID[index], self.domain_slot[index]

	def __len__(self):
		return len(self.input_tokens)

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
				domain_slot_pairs = set()
				for domain in info['services']:
					for slot, _ in schema_info[domain]['slots']:
						domain_slot_pairs.add((domain, slot))
				data['domain_slot'] = [(domain, slot) for domain, slot in domain_slot_pairs]
				data_list.append(data)
			else:
				data = {'turns': []}
				data['dialogue_id'] = info['dialogue_id']
				data['domains'] = info['services']
				domain_slot = set()
				turn_info = dict()
				for turn in info['turns']:
					if turn['speaker'] == 'USER':
						turn_info = {'usr': turn['utterance'], 'slot_values': {}}
						for frame in turn['frames']:
							domain = frame['service']
							for k, v in frame['state']['slot_values'].items():
								slot_name = k
								slot_value = v[0] if isinstance(v, list) else v
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

def get_dict_list(args, raw_data, tokenizer, schema_info, slot_types):
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
				slot_type = slot_types[(domain, slot_name)]
				if args['use_descr'] and args['slot_type']:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_type} of {slot_descr} of the {domain_descr}'
				elif args['use_descr']:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_descr} of the {domain_descr}'
				elif args['slot_type']:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_type} of {slot_name} of the {domain}'
				else:
					input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_name} of the {domain}'
				output_sentence = slot_value + f" {tokenizer.eos_token}"
				results['input_sentences'].append(input_sentence)
				results['output_sentences'].append(output_sentence)
				if slot_value != 'none':
					for t in range(1, args['notnone_multiple']):
						results['input_sentences'].append(input_sentence)
						results['output_sentences'].append(output_sentence)
				bar.update(1)
	bar.close()
	tokenize_results = {'input_tokens': [], 'output_tokens': []}
	print('Start to tokenize...\n', flush=True)
	total=len(results['input_sentences'])
	bar=tqdm(total=total)
	for input_sentence, output_sentence in zip(results['input_sentences'], results['output_sentences']):
		input_token = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=False, verbose=False, return_attention_mask=False)
		output_token = tokenizer(output_sentence, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
		tokenize_results['input_tokens'].append(input_token['input_ids'])
		tokenize_results['output_tokens'].append(output_token['input_ids'])
		bar.update(1)
	
	return tokenize_results

def get_test_list(args, raw_data, tokenizer, schema_info, slot_types):
	results = {'ID': [], 'input_sentences': [], 'domain-slot': []}
	slots_descr = {}
	for domain in schema_info.keys():
		for slot in schema_info[domain]['slots']:
			slots_descr[(domain, slot[0])] = slot[1]
	
	for data in raw_data:
		dialogue_history = ''
		for turn in data['turns']:
			dialogue_history += ('System: ' + turn['sys'] + 'User: ' + turn['usr'])
		for domain, slot in data['domain_slot']:
			domain_descr = schema_info[domain]['description']
			slot_descr = slots_descr.get((domain, slot), 'none')
			slot_type = slot_types[(domain, slot)]
			if args['use_descr'] and args['slot_type']:
				input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_type} of {slot_descr} of the {domain_descr}'
			elif args['use_descr']:
				input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_descr} of the {domain_descr}'
			elif args['slot_type']:
				input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot_type} of {slot} of the {domain}'
			else:
				input_sentence = dialogue_history + f'{tokenizer.sep_token} {slot} of the {domain}'
			results['domain-slot'].append(f'{domain}-{slot}')
			results['input_sentences'].append(input_sentence)
			results['ID'].append(data['dialogue_id'])
	tokenize_results = {'input_tokens': [], 'domain-slot': [], 'ID': []}
	print('Start to tokenize...\n', flush=True)
	total=len(results['input_sentences'])
	bar=tqdm(total=total)
	for info in zip(results['input_sentences'], results['ID'], results['domain-slot']):
		input_sentence = info[0]
		input_token = tokenizer(input_sentence, return_tensors="pt", add_special_tokens=False, verbose=False, return_attention_mask=False)
		tokenize_results['input_tokens'].append(input_token['input_ids'])
		tokenize_results['ID'].append(info[1])
		tokenize_results['domain-slot'].append(info[2])
		bar.update(1)
	
	return tokenize_results

def collate_fn(data, tokenizer):
	batch = {}
	batch_inputs = {'input_ids': [s.squeeze() for s, _ in data]}
	batch_outputs = {'input_ids': [s.squeeze() for _, s in data]}
	input_tokens = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt", verbose=False, return_attention_mask=True)
	output_tokens = tokenizer.pad(batch_outputs, padding=True, return_tensors="pt", return_attention_mask=False)
	output_tokens['input_ids'].masked_fill_(output_tokens['input_ids']==tokenizer.pad_token_id, -100)
	batch['encoder_input'] = input_tokens['input_ids'].squeeze()
	batch['attention_mask'] = input_tokens['attention_mask'].squeeze()
	batch['decoder_output'] = output_tokens['input_ids'].squeeze()
	return batch

def get_slot_type():
	df = pd.read_csv('slot_type.csv')
	slot_types = {}
	for i in range(len(df)):
		domain = df.at[i, 'domains']
		slot = df.at[i, 'slots']
		slot_type = df.at[i, 'slots_type']
		slot_types[(domain, slot)] = slot_type
	return slot_types

def test_collate_fn(data, tokenizer):
	batch = {}
	batch_inputs = {'input_ids': [s.squeeze() for s, _, __ in data]}
	batch_ids = [s for _, s, __ in data]
	batch_domain_slot = [s for _, __, s in data]
	input_tokens = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt", verbose=False, return_attention_mask=True)
	batch['encoder_input'] = input_tokens['input_ids'].squeeze()
	batch['attention_mask'] = input_tokens['attention_mask'].squeeze()
	batch['ID'] = batch_ids
	batch['domain-slot'] = batch_domain_slot
	return batch

def get_train_dataloader(args, tokenizer):
	domain_slots = get_domain_slot(args['schema_dir'])
	slot_types = get_slot_type()
	train_data = read_data(args['train_dir'], domain_slots, False)
	eval_data = read_data(args['eval_dir'], domain_slots, False)
	if os.path.exists('train_tokenize.pickle'):
		with open('train_tokenize.pickle', 'rb') as fp:
			train_dict = pickle.load(fp)
	else:
		train_dict = get_dict_list(args, train_data, tokenizer, domain_slots, slot_types)
		with open('train_tokenize.pickle', 'wb') as fp:
			pickle.dump(train_dict, fp)

	if os.path.exists('eval_tokenize.pickle'):
		with open('eval_tokenize.pickle', 'rb') as fp:
			eval_dict = pickle.load(fp)
	else:
		eval_dict = get_dict_list(args, eval_data, tokenizer, domain_slots, slot_types)
		with open('eval_tokenize.pickle', 'wb') as fp:
			pickle.dump(eval_dict, fp)

	train_dataset = DSTDataset(train_dict, args)
	eval_dataset = DSTDataset(eval_dict, args)
	train_dataloader = DataLoader(train_dataset, args['train_batch_size'], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=4)
	eval_dataloader = DataLoader(eval_dataset, args['eval_batch_size'], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=4)
	return train_dataloader, eval_dataloader

def get_test_dataloader(args, tokenizer):
	domain_slots = get_domain_slot(args['schema_dir'])
	slot_types = get_slot_type()
	test_data = read_data(args['test_dir'], domain_slots, True)
	test_list = get_test_list(args, test_data, tokenizer, domain_slots, slot_types)
	test_dataset = DSTDataset(test_list, args)
	test_dataloader = DataLoader(test_dataset, args['test_batch_size'], shuffle=False, collate_fn=partial(test_collate_fn, tokenizer=tokenizer), num_workers=4)
	return test_dataloader
