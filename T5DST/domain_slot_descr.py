import sys
import json
import pandas as pd

content = {'domains': [], 'domains_descr': [], 'slots': [], 'slots_descr': []}

with open(sys.argv[1], 'r') as fp:
	schema_info = json.load(fp)
	for info in schema_info:
		domain = info['service_name']
		domain_descr = info['description']
		for slot_info in info['slots']:
			slot = slot_info['name']
			slot_descr = slot_info['description']
			content['domains'].append(domain)
			content['domains_descr'].append(domain_descr)
			content['slots'].append(slot)
			content['slots_descr'].append(slot_descr)

df = pd.DataFrame(content)
df.to_csv('domain-slot_description.csv')
