from config import get_args
from data_loader import *
import datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration

args = get_args()
domain_slots = get_domain_slot(args['schema_dir'])

tokenizer = T5Tokenizer.from_pretrained('t5-small', bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
train_dataset, eval_dataset = get_train_dataset(args, tokenizer)
