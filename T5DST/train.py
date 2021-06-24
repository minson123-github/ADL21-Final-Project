from config import get_args
from data_loader import *
import datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

class DST_Seq2Seq(pl.LightningModule):
	def __init__(self, args, tokenizer, model):
		super().__init__()
		self.tokenizer = tokenizer
		self.model = model
		self.lr = args["lr"]

	def training_step(self, batch, batch_idx):
		self.model.train()
		outputs = self.model(
					input_ids=batch['encoder_input'], 
					attention_mask=batch['attention_mask'], 
					labels=batch['decoder_output']
				)
		return {'loss': outputs.loss, 'log': {'train_loss': outputs.loss}}

	def validation_step(self, batch, batch_idx):
		self.model.eval()
		outputs = self.model(
					input_ids=batch['encoder_input'], 
					attention_mask=batch['attention_mask'], 
					labels=batch['decoder_output']
				)
		return {'eval_loss': outputs.loss, 'log': {'eval_loss': outputs.loss}}
	
	def validation_epoch_end(self, outputs):
		eval_loss_mean = sum([output['eval_loss'] for output in outputs]) / len(outputs)
		results = {'progress_bar': {'eval_loss': eval_loss_mean.item()}, 'log': {'eval_loss': eval_loss_mean.item()}, 'eval_loss': eval_loss_mean.item()}
		return results

	def configure_optimizers(self):
		return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

def train_process(args):
	seed_everything(args['seed'])
	t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
	tokenizer = T5Tokenizer.from_pretrained('t5-small', bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
	t5_model.resize_token_embeddings(new_num_tokens=len(tokenizer))

	model = DST_Seq2Seq(args, tokenizer, t5_model)
	domain_slots = get_domain_slot(args['schema_dir'])
	train_dataloader, eval_dataloader = get_train_dataloader(args, tokenizer)
	save_path = os.path.join(args["saving_dir"], 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	trainer = Trainer(
				default_root_dir=save_path, 
				accumulate_grad_batches=args["gradient_accumulation_steps"], 
				max_epochs=args['n_epochs'], 
				gpus=args["n_gpus"], 
				deterministic=True, 
				num_nodes=1, 
				accelerator="ddp"
			)
	print('start to training...', flush=True)
	trainer.fit(model, train_dataloader, eval_dataloader)
	model.model.save_pretrained(save_path)
	model.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
	args = get_args()
	train_process(args)
