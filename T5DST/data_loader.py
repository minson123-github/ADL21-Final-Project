import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from functools import partial

class DSTDataset(Dataset):
	def __init__(self, data, args):
		self.input_sentences = data['input_sentences']
		if 'output_sentences' in data:
			self.output_sentences = data['output_sentences']
		else:
			self.output_sentences = None
		self.args = args
	
	def __getitem__(self, index):
		if self.output_sentences != None:
			return self.input_sentences[index], self.output_sentences[index]
		return self.input_sentences[index]

	def __len__(self):
		return len(self.input_sentences)

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

def collate_fn(data, tokenizer):
	batch = {'encoder_input': [], 'attention_mask': [], 'decoder_output': []}
	for inputs, outputs in data:
		input_tokens = tokenizer(inputs, padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
		output_tokens = tokenizer(outputs, padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
		output_tokens['input_ids'].masked_fill_(output_tokens['input_ids']==tokenizer.pad_token_id, -100)
		batch['encoder_input'].append(input_tokens['input_ids'].squeeze())
		batch['attention_mask'].append(input_tokens['attention_mask'].squeeze())
		batch['decoder_output'].append(output_tokens['input_ids'].squeeze())
	return batch
			
def get_train_dataloader(args, tokenizer):
	domain_slots = get_domain_slot(args['schema_dir'])
	train_data = read_data(args['train_dir'], domain_slots, False)
	eval_data = read_data(args['eval_dir'], domain_slots, False)
	train_dict = get_dict_list(args, train_data, tokenizer, domain_slots)
	eval_dict = get_dict_list(args, eval_data, tokenizer, domain_slots)
	print(train_dict['input_sentences'][300])
	print(train_dict['output_sentences'][300])
	train_dataset = DSTDataset(train_dict, args)
	eval_dataset = DSTDataset(eval_dict, args)
	train_dataloader = DataLoader(train_dataset, args['train_batch_size'], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=4)
	eval_dataloader = DataLoader(eval_dataset, args['eval_batch_size'], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=4)
	return train_dataloader, eval_dataloader
