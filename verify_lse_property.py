import json
import argparse

def verify_lsc_two_trees(i, j, thresholds_x_tree, k):
    thresholds_tree_1: json = thresholds_x_tree[i]
    thresholds_tree_2: json = thresholds_x_tree[j]
    for feature in list(thresholds_tree_1.keys()):
        if feature in thresholds_tree_2.keys():
            thresholds_x_feature_tree_1 = thresholds_tree_1[feature]
            thresholds_x_feature_tree_2 = thresholds_tree_2[feature]
            for threshold_1 in thresholds_x_feature_tree_1:
                for threshold_2 in thresholds_x_feature_tree_2:
                    if abs(threshold_1 - threshold_2) <= 2*k:
                        print("Thresholds {} of feature {} of tree {} and {} of feature {} of tree {} violate the lsc".format(threshold_1, feature, i, threshold_2, feature, j))
                        return False
    return True


def check_lse_condition(lightgbm_model_json, k):
    thresholds_x_tree = {}

    tree_info = lightgbm_model_json["tree_info"]
    tree_id = 0
    for tree in tree_info:
        thresholds_x_tree[tree_id] = {}
        stack = [tree["tree_structure"]]
        while len(stack) > 0:
            node = stack.pop()
            if "split_index" in node.keys():
                feature = int(node["split_feature"])
                if feature not in thresholds_x_tree[tree_id].keys():
                    thresholds_x_tree[tree_id][feature] = [float(node["threshold"])]
                else:
                    thresholds_x_tree[tree_id][feature].append(float(node["threshold"]))
                if "right_child" in node.keys():
                    stack.append(node["right_child"])
                if "left_child" in node.keys():
                    stack.append(node["left_child"])
        tree_id += 1
    
    #print(thresholds_x_tree)

    tree_ids = list(thresholds_x_tree.keys())
    for i in range(0, len(tree_ids)):
        for j in range(i+1, len(tree_ids)):
            if not verify_lsc_two_trees(i, j, thresholds_x_tree, k):
                return False
    return True

parser = argparse.ArgumentParser()
parser.add_argument('model_json_dump', type=str)
parser.add_argument('k', type=float)
args = parser.parse_args()

f = open(args.model_json_dump, "r")
model_json = json.load(f)
print("LSC holds." if check_lse_condition(model_json, args.k) else "LSC does not hold.")

