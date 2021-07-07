# T5DST

This part we use T5 to implement DST task. The implement method reference from:
**Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue StateTracking** [[PDF]](https://www.aclweb.org/anthology/2021.naacl-main.448.pdf).

## Experiments
Before you start use our model, you may need to run `pip3 install transformers` to install newest transformers package.
Here are some examples for reproduce and predict, if you want to know how these args works, you can read `config.py` or just run `python3 run.py --help` to get args description.
In here, we have two models. They are in `model1`, `model2` directory. `model1` has higher kaggle seen domain public score and `model2` has higher kaggle unseen domain public score. Their difference is number of training epochs, we take `model2` as an example for predict.
**Model Reproduce**
```console
python3 run.py \
	--mode="train" \
	--schema_dir="../data-0625/schema.json" \
	--train_dir="../data-0625/train" \
	--eval_dir="../data-0625/dev" \
	--saving_dir="ckpt" \
	--gradient_accumulation_steps=1 \
	--train_batch_size=8 \
	--eval_batch_size=8 \
	--n_epochs=2 \
	--pretrained="t5-base" \
	--slot_type=1 \
	--use_descr=1 \
	--n_gpus=1 \
```
**Seen Domain Predict**
In predict phase, `run.py` would use the model in `saving_dir` for predict.
```console
python3 run.py \
	--mode="test" \
	--schema_dir="../data-0625/schema.json" \
	--train_dir="../data-0625/train" \
	--eval_dir="../data-0625/dev" \
	--test_dir="../data-0625/test_seen" \
	--saving_dir="model2" \
	--test_batch_size=8 \
	--predict_file="pred.json" \
	--n_beams=7 \
	--slot_type=1 \
	--use_descr=1 \
	--n_gpus=1 \
```
**Unseen Domain Predict**
```console
python3 run.py \
	--mode="test" \
	--schema_dir="../data-0625/schema.json" \
	--train_dir="../data-0625/train" \
	--eval_dir="../data-0625/dev" \
	--test_dir="../data-0625/test_unseen" \
	--saving_dir="model2" \
	--test_batch_size=8 \
	--predict_file="pred.json" \
	--n_beams=10 \
	--slot_type=1 \
	--use_descr=1 \
	--n_gpus=1 \
```
