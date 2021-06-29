#!/usr/bin/env python3

"""
Args
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    type=str,
                    default=None,
                    required=True,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--input_file",
                    type=str,
                    required=True,
                    help="Path to input file")
parser.add_argument("--output_file",
                    type=str,
                    required=True,
                    help="Path to output file")
parser.add_argument("--length",
                    type=int,
                    default=20)
parser.add_argument("--beams",
                    type=int,
                    default=1,
                    help="The number of beams to be used when finding best output")
parser.add_argument("--repetition_penalty",
                    type=float,
                    default=1.0,
                    help="primarily useful for CTRL model; in that case, use 1.2")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument("--no_cuda",
                    action="store_true",
                    help="Avoid using CUDA when available")
parser.add_argument("--fp16",
                    action="store_true",
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

"""
Seed
"""
from transformers import set_seed

set_seed(args.seed)


"""
Logging
"""
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Result Generator")
logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")



"""
Initialize Mdoel
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
model.config.pad_token_id = model.config.eos_token_id
model = model.to(args.device)
if args.fp16:
    model.half()
model.eval()
args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
logger.info(args)


"""
Prediction
"""
from tqdm.auto import trange

input_texts = open(args.input_file).read().split("\n")
input_texts_id = []
results = {}
for i in range(len(input_texts)):
    ctx_pos = input_texts[i].find("<|context|>")
    idval = input_texts[i][:ctx_pos]
    input_texts[i] = input_texts[i][ctx_pos:]
    input_texts_id.append(idval)
    results[idval] = {}


for idx in trange(len(input_texts)):
    encoded_input = tokenizer.encode(input_texts[idx], add_special_tokens=False, return_tensors="pt")
    encoded_input = encoded_input.to(args.device)

    if encoded_input.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_input

    generated_sequence = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_input[0]),
        um_beams=args.beams,
        early_stopping=True,
        repetition_penalty=args.repetition_penalty,
    )
    generated_sequence = generated_sequence.flatten()

    generated_sequence = generated_sequence.tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    
    slot_st_pos = text.find("<|beliefval|>")
    slot_ed_pos = text.find("<|endofbeliefval|>")
    if -1 in [slot_st_pos, slot_ed_pos]:
        continue
    slot = text[slot_st_pos + len("<|beliefval|>"):slot_ed_pos]
    slot = slot.strip()
    if slot == "<|emptyslot|>":
        continue

    slot_name_st_pos = input_texts[idx].find("<|belief|>")
    slot_name_ed_pos = input_texts[idx].find("<|endofbelief|>")
    slot_name = input_texts[idx][slot_name_st_pos + len("<|belief|>"):slot_name_ed_pos]
    slot_name = slot_name.strip()
    
    results[input_texts_id[idx]][slot_name] = slot

with open(args.output_file, "w") as fp:
    fp.write('id,state\n')
    for dialogue_id, states in sorted(results.items(), key=lambda x: x[0]):
        if len(states) == 0:  # no state ?
            str_state = 'None'
        else:
            str_state = ''
            for slot, value in sorted(states.items(), key=lambda x: x[0]):
                # NOTE: slot = "{}-{}".format(service_name, slot_name)
                str_state += "{}={}|".format(
                        slot.lower(), value.replace(',', '_').lower())
            str_state = str_state[:-1]
        fp.write('{},{}\n'.format(dialogue_id, str_state))
