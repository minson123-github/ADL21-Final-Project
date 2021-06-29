#!/usr/bin/env python3

"""
Args
"""
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='gpt2')
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(['--do_train',
                                                                           '--do_eval',
                                                                           '--fp16',
                                                                           '--gradient_accumulation_steps', '18',
                                                                           '--learning_rate', '1e-3',
                                                                           '--num_train_epochs', '10',
                                                                           '--output_dir', './model_10empty',
                                                                           '--per_device_train_batch_size', '2',
                                                                           '--train_file', './train.txt',
                                                                           '--validation_file', './dev.txt'])


"""
Seed
"""
from transformers import set_seed

set_seed(training_args.seed)


"""
Logging
"""
import logging
import sys
import transformers

logger = logging.getLogger('simple-oToToD')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info(f"Training/evaluation parameters {training_args}")


"""
Checkpoint
"""
import os
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )


"""
Dataset
"""
from datasets import load_dataset

data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
extension = (
    data_args.train_file.split(".")[-1]
    if data_args.train_file is not None
    else data_args.validation_file.split(".")[-1]
)
if extension == "txt":
    extension = "text"
datasets = load_dataset(extension, data_files=data_files)


"""
Initializee Models
"""
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                    revision=model_args.model_revision)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                          revision=model_args.model_revision,
                                          use_fast=model_args.use_fast_tokenizer)
tokenizer.add_special_tokens({'additional_special_tokens':
                              ['<|belief|>',
                               '<|endofbelief|>',
                               '<|context|>',
                               '<|endofcontext|>',
                               '<|user|>',
                               '<|system|>',
                               '<|servicedetail|>',
                               '<|endofservicedetail|>',
                               '<|slotdetail|>',
                               '<|endofslotdetail|>',
                               '<|beliefval|>',
                               '<|endofbeliefval|>',
                               '<|emptyslot|>']})
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                             config=config,
                                             revision=model_args.model_revision)
model.resize_token_embeddings(len(tokenizer))


"""
Preprocess Dataset
"""
import os

NUM_PROC = os.cpu_count()

if training_args.do_train:
    column_names = datasets["train"].column_names
else:
    column_names = datasets["validation"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]
tokenized_datasets = datasets.map(lambda examples: tokenizer(examples[text_column_name]),
                                  batched=True,
                                  num_proc=NUM_PROC,
                                  remove_columns=column_names,
                                  desc="Running tokenizer on dataset")
block_size = min(1024, tokenizer.model_max_length)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts,
                                     batched=True,
                                     num_proc=NUM_PROC,
                                     desc=f"Grouping texts in chunks of {block_size}")



"""
Training
"""
from transformers import Trainer, default_data_collator

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=lm_datasets["train"] if training_args.do_train else None,
                  eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
                  tokenizer=tokenizer,
                  data_collator=default_data_collator)
train_result = trainer.train(resume_from_checkpoint=None if last_checkpoint is None else last_checkpoint)
trainer.save_model()


"""
Train Result
"""
metrics = train_result.metrics

metrics["train_samples"] = len(lm_datasets["train"])

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


"""
Evaluate
"""
logger.info("*** Evaluate ***")

metrics = trainer.evaluate()

metrics["eval_samples"] = len(lm_dataset["validation"])
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
