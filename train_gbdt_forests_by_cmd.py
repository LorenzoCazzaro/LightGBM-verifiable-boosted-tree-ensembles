import subprocess
import argparse

_DATASET_FOLDER = "./datasets/{}/dataset/"
_TRAINING_SET_NAME = "training_set_normalized.csv"

parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depths', type=str)
parser.add_argument('random_states', type=str)
parser.add_argument('ks', type=str)
parser.add_argument('--mins_data_in_leaf', type=str, default="20")
args = parser.parse_args()

dataset_list = args.datasets.split("-")
n_tree_list = args.n_trees.split("-")
depth_list = args.depths.split("-")
random_state_list = args.random_states.split("-")
k_list = args.ks.split(" ")
min_data_in_leaf_list = args.mins_data_in_leaf.split("-")

for dataset in dataset_list:
	print("training {}".format(dataset))
	for n_tree in n_tree_list:
		for depth in depth_list:
			for random_state in random_state_list:
				for k in k_list:
					for min_data_in_leaf in min_data_in_leaf_list:
						print("training lightgbm_{}_{}_{}_{}_{}.txt".format(n_tree, depth, random_state, k, min_data_in_leaf))
						p = subprocess.run(["python3", "train_lse_gbdt_forest_by_cmd.py", dataset, n_tree, depth, random_state, k, "--min_data_in_leaf", min_data_in_leaf], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
