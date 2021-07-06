from config import get_args
from data_loader import *
from transformers import T5TokenizerFast, T5ForConditionalGeneration, AdamW
from pytorch_lightning import Trainer, seed_everything
from model import DST_Seq2Seq
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import os

class CheckpointEveryNSteps(pl.Callback):
	def __init__(self, save_freq, save_dir):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.save_cnt = 0
	
	def on_batch_end(self, trainer: pl.Trainer, _):
		global_step = trainer.global_step
		if global_step % self.save_freq == 0:
			self.save_cnt += 1
			filename = f'model-checkpoint_{self.save_cnt}.ckpt'
			ckpt_path = os.path.join(self.save_dir, filename)
			trainer.save_checkpoint(ckpt_path)

def test_process(args):
	seed_everything(args['seed'])
	save_path = os.path.join(args['saving_dir'], 'model')
	if not os.path.exists(save_path):
		print('Model not found...')
		exit(0)
	if args['load_checkpoint'] == None:
		model = T5ForConditionalGeneration.from_pretrained(save_path)
		model.to('cuda')
		tokenizer = T5TokenizerFast.from_pretrained(save_path)
	else:
		print('Load from checkpoint...', flush=True)
		t5_model = T5ForConditionalGeneration.from_pretrained(args['pretrained'])
		t5_tokenizer = T5TokenizerFast.from_pretrained(args['pretrained'], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
		t5_model.resize_token_embeddings(new_num_tokens=len(t5_tokenizer))
		checkpoint_path = os.path.join(args['saving_dir'], 'checkpoints', 'model-checkpoint_{}.ckpt'.format(args['load_checkpoint']))
		checkpoint = DST_Seq2Seq.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args, model=t5_model, tokenizer=t5_tokenizer)
		model = checkpoint.model
		model.to('cuda')
		tokenizer = checkpoint.tokenizer
	test_dataloader = get_test_dataloader(args, tokenizer)
	pred = {}
	model.eval()
	with torch.no_grad():
		for batch in tqdm(test_dataloader):
			dst_outputs = model.generate(
							input_ids=batch['encoder_input'].cuda(), 
							attention_mask=batch['attention_mask'].cuda(), 
							eos_token_id=tokenizer.eos_token_id, 
							max_length=200, 
							num_beams=args['n_beams'], 
							length_penalty=args['length_penalty']
						)
			dst_outputs = dst_outputs.cpu()
			batch_values = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
			for _id, domain_slot, slot_value in zip(batch['ID'], batch['domain-slot'], batch_values):
				if _id not in pred:
					pred[_id] = {}
				pred[_id][domain_slot] = slot_value
		with open(args['predict_file'], 'w') as fp:
			json.dump(pred, fp, indent=4)

def train_process(args):
	seed_everything(args['seed'])
	t5_model = T5ForConditionalGeneration.from_pretrained(args['pretrained'])
	tokenizer = T5TokenizerFast.from_pretrained(args['pretrained'], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
	t5_model.resize_token_embeddings(new_num_tokens=len(tokenizer))

	model = DST_Seq2Seq(args, tokenizer, t5_model)
	domain_slots = get_domain_slot(args['schema_dir'])
	train_dataloader, eval_dataloader = get_train_dataloader(args, tokenizer)
	save_path = os.path.join(args["saving_dir"], 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	
	callbacks = []
	if args['saving_steps'] != None:
		callbacks = [CheckpointEveryNSteps(args['saving_steps'], os.path.join(args['saving_dir'], 'checkpoints'))]
		if not os.path.exists(os.path.join(args['saving_dir'], 'checkpoints')):
			os.makedirs(os.path.join(args['saving_dir'], 'checkpoints'))

	trainer = Trainer(
				default_root_dir=save_path, 
				accumulate_grad_batches=args["gradient_accumulation_steps"], 
				max_epochs=args['n_epochs'], 
				gpus=args["n_gpus"], 
				deterministic=True, 
				num_nodes=1, 
				accelerator="dp", 
				callbacks = callbacks
			)
	print('start to training...', flush=True)
	trainer.fit(model, train_dataloader, eval_dataloader)
	model.model.save_pretrained(save_path)
	model.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
	args = get_args()
	if args['mode'] == 'train':
		train_process(args)
	elif args['mode'] == 'test':
		test_process(args)
