def from_lightgbm_json_to_silva(lightgbm_model_json):
    to_silva_str = ""
    num_trees = len(lightgbm_model_json["tree_info"])
    n_features = len(lightgbm_model_json["feature_names"])
    n_classes = lightgbm_model_json["num_class"]+1
    to_silva_str += "classifier-forest {}\n".format(num_trees)

    tree_info = lightgbm_model_json["tree_info"]
    for tree in tree_info:
        to_silva_str += "classifier-decision-tree {} {}\n".format(n_features, n_classes)
        to_silva_str += "0 1\n" 
        stack = [tree["tree_structure"]]
        while len(stack) > 0:
            node = stack.pop()
            if "split_index" in node.keys():
                to_silva_str += "SPLIT {} {}\n".format(node["split_feature"], node["threshold"])
                if "right_child" in node.keys():
                    stack.append(node["right_child"])
                if "left_child" in node.keys():
                    stack.append(node["left_child"])
            else:
                leaf_value = float(node["leaf_value"])
                to_silva_str += "LEAF_LOGARITHMIC {} {}\n".format(-leaf_value, leaf_value)
    return to_silva_str