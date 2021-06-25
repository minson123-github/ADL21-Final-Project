import pytorch_lightning as pl
from transformers import AdamW

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
