NLG
===

# Data

Download the dataset and extract it into `./data/`. This folder should includes `./data/data/train`, `./data/data/dev`, `./data/data/test`.  
The alternative way is to `mkdir ./data && ln -s ../../data-0625 data/data && ln -s ./test_seen data/data/test`.

# Training

```bash
python3 adjust_data.py
python3 generate_delex.py
python3 gen_parlai_data.py
parlai train_model -t fromfile:parlaiformat --fromfile_datapath ./parlai --fromfile-datatype-extension true  -m transformer/generator --init-model zoo:tutorial_transformer_generator/model --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --skip-generation True --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True --model-file ./train_90M
mkdir results
mkdir results/greedy
parlai interactive -mf ./train_90M < lm.input.dev.cc.txt > lm.output.dev.cc.txt
parlai interactive -mf ./train_90M < lm.input.test.cc.txt > lm.output.test.cc.txt
cp lm.output.test.cc.txt results/greedy
python3 gen_arranger_input.py
python3 fix_arranger.py
python3 run_multiple_choice.py --model_type roberta --task_name acc --model_name_or_path roberta-base --do_train --do_eval --do_test --do_lower_case --data_dir . --learning_rate 2e-5 --num_train_epochs 3 --max_seq_length 512 --output_dir acc_arranger_roberta_base_3epoch --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=1 --gradient_accumulation_steps 24 --overwrite_output --save_steps 10000
cp acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt results/greedy
mkdir results/beam16
parlai interactive -mf ./train_90M --beam-size 16 < lm.input.dev.cc.txt > lm.output.dev.cc.txt
parlai interactive -mf ./train_90M --beam-size 16 < lm.input.test.cc.txt > lm.output.test.cc.txt
cp lm.output.test.cc.txt results/beam16
python3 gen_arranger_input.py
python3 fix_arranger.py
python3 run_multiple_choice.py --model_type roberta --task_name acc --model_name_or_path roberta-base --do_eval --do_test --do_lower_case --data_dir acc_arranger_roberta_base_3epoch --learning_rate 2e-5 --num_train_epochs 3 --max_seq_length 512 --output_dir acc_arranger_roberta_base_3epoch --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=1 --gradient_accumulation_steps 24 --overwrite_output --save_steps 10000
cp acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt results/greedy
python3 gen_final.py
```

# Reproduce

```bash
bash download.sh
python3 adjust_data.py
python3 generate_delex.py
python3 gen_parlai_data.py
mkdir results
mkdir results/greedy
parlai interactive -mf ./train_90M < lm.input.dev.cc.txt > lm.output.dev.cc.txt
parlai interactive -mf ./train_90M < lm.input.test.cc.txt > lm.output.test.cc.txt
cp lm.output.test.cc.txt results/greedy
python3 gen_arranger_input.py
python3 fix_arranger.py
python3 run_multiple_choice.py --model_type roberta --task_name acc --model_name_or_path roberta-base --do_eval --do_test --do_lower_case --data_dir acc_arranger_roberta_base_3epoch --learning_rate 2e-5 --num_train_epochs 3 --max_seq_length 512 --output_dir acc_arranger_roberta_base_3epoch --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=1 --gradient_accumulation_steps 24 --overwrite_output --save_steps 10000
cp acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt results/greedy
mkdir results/beam16
parlai interactive -mf ./train_90M --beam-size 16 < lm.input.dev.cc.txt > lm.output.dev.cc.txt
parlai interactive -mf ./train_90M --beam-size 16 < lm.input.test.cc.txt > lm.output.test.cc.txt
cp lm.output.test.cc.txt results/beam16
python3 gen_arranger_input.py
python3 fix_arranger.py
python3 run_multiple_choice.py --model_type roberta --task_name acc --model_name_or_path roberta-base --do_eval --do_test --do_lower_case --data_dir acc_arranger_roberta_base_3epoch --learning_rate 2e-5 --num_train_epochs 3 --max_seq_length 512 --output_dir acc_arranger_roberta_base_3epoch --per_gpu_eval_batch_size=16 --per_gpu_train_batch_size=1 --gradient_accumulation_steps 24 --overwrite_output --save_steps 10000
cp acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt results/greedy
python3 gen_final.py
```
