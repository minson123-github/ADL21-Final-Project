import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--schema_dir', type=str, help="directory path of schema.json.")
	parser.add_argument('--train_dir', type=str, help='directory path of training data.')
	parser.add_argument('--eval_dir', type=str, help='directory path of evaluate data.')
	parser.add_argument('--test_dir', type=str, help='directory path of testing data.')
	parser.add_argument('--train_batch_size', type=int, help='Batch size for training.')
	parser.add_argument('--eval_batch_size', type=int, help='Batch size for evaluate.')
	parser.add_argument('--test_batch_size', type=int, help='Batch size for predict.')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
	parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs.')
	parser.add_argument('--seed', type=int, default=24, help='random seed')
	parser.add_argument('--gradient_accumulation_steps', type=int, help='Accumulate gradients on several steps')
	parser.add_argument('--saving_dir', type=str, default="ckpt", help="path for saving model.")
	parser.add_argument('--use_descr', type=bool, default=False, help='Whether to use domain and slot description.')
	parser.add_argument('--n_gpus', type=int, default=1, help='number of gpu.')
	parser.add_argument('--predict_file', type=str, help='model predict file results.')
	parser.add_argument('--mode', type=str, help='mode: train/evaluate/test')
	parser.add_argument('--n_beams', type=int, default=5, help='beam size in beam search.')

	args = parser.parse_args()
	return vars(args)
