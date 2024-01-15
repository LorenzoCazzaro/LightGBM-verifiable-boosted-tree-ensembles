import subprocess
import argparse

_DATASET_FOLDER = "./datasets/{}/models/"

parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str)
parser.add_argument('n_trees', type=str)
parser.add_argument('depths', type=str)
parser.add_argument('random_states', type=str)
parser.add_argument('ks_model', type=str)
parser.add_argument('ks_verif', type=str)
parser.add_argument('--mins_data_in_leaf', type=str, default="20")
args = parser.parse_args()

dataset_list = args.datasets.split("-")
n_tree_list = args.n_trees.split("-")
depth_list = args.depths.split("-")
random_state_list = args.random_states.split("-")
k_model_list = args.ks_model.split(" ")
k_verif_list = args.ks_verif.split("-")
min_data_in_leaf_list = args.mins_data_in_leaf.split("-")

for dataset in dataset_list:
	print("converting {}".format(dataset))
	for k_verif in k_verif_list:
		for n_tree in n_tree_list:
			for depth in depth_list:
				for random_state in random_state_list:
					for k_model in k_model_list:
						for min_data_in_leaf in min_data_in_leaf_list:
							print("verifying lightgbm_{}_{}_{}_{}_{}.txt for k={}".format(n_tree, depth, random_state, k_model, min_data_in_leaf, k_verif))
							p = subprocess.run(["python3", "verify_forest_with_silva_by_cmd.py", dataset, n_tree, depth, random_state, k_model, k_verif, "--min_data_in_leaf", min_data_in_leaf], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
							#print(p.stdout)
							#print(p.stderr)