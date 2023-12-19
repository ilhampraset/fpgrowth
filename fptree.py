import itertools
import json


class Node:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment(self, count):
        self.count += count


def build_fp_tree(dataset, min_support):
    header_table = {}
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + dataset[transaction]

    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    frequent_items = set(header_table.keys())

    if len(frequent_items) == 0:
        return None, None

    for item in header_table:
        header_table[item] = [header_table[item], None]

    fp_tree = Node("Null", 1, None)

    for transaction, count in dataset.items():
        transaction = [item for item in transaction if item in frequent_items]
        transaction.sort(key=lambda x: header_table[x][0], reverse=True)
        current_node = fp_tree

        for item in transaction:
            current_node = insert_node(item, current_node, header_table, count)

    return fp_tree, header_table


def insert_node(item, node, header_table, count):
    if item in node.children:
        node.children[item].increment(count)
    else:
        new_node = Node(item, count, node)
        node.children[item] = new_node
        update_header_table(item, new_node, header_table)

    return node.children[item]


def update_header_table(item, target_node, header_table):
    if header_table[item][1] is None:
        header_table[item][1] = target_node
    else:
        current_node = header_table[item][1]
        while current_node.node_link is not None:
            current_node = current_node.node_link
        current_node.node_link = target_node


def print_fp_tree(node, level=0):
    if node is not None:
        print("  " * level + f"{node.name} ({node.count})")
        for child in node.children.values():
            print_fp_tree(child, level + 1)


def fp_tree_to_json(node):
    if node is not None:
        node_json = {
            "name": node.name,
            "count": node.count,
            "children": [fp_tree_to_json(child) for child in node.children.values()]
        }
        return node_json
    else:
        return None


def extract_frequent_patterns(header_table, prefix=[]):
    frequent_patterns = {}

    for item, node in header_table.items():
        current_pattern = prefix + [item]
        frequent_patterns[frozenset(current_pattern)] = node.count

        # Build conditional database
        conditional_database = {}
        while node.node_link is not None:
            node = node.node_link
            prefix_path = []
            while node.parent is not None:
                prefix_path.append(node.name)
                node = node.parent
            if prefix_path:
                conditional_database[frozenset(prefix_path)] = node.count

        conditional_fp_tree, conditional_header_table = build_fp_tree(conditional_database, 1)

        if conditional_fp_tree is not None:
            conditional_patterns = extract_frequent_patterns(conditional_header_table, current_pattern)
            frequent_patterns.update(conditional_patterns)

    return frequent_patterns


def generate_association_rules(frequent_patterns):
    association_rules = []
    for itemset, support in frequent_patterns.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for subset in itertools.combinations(itemset, i):
                    antecedent = set(subset)
                    consequent = itemset - antecedent
                    confidence = support / frequent_patterns[antecedent]
                    rule = (antecedent, consequent, confidence)
                    association_rules.append(rule)
    return association_rules


def conditional_fp_tree_to_json(header_table):
    conditional_fp_tree_json = {}
    for item, node in header_table.items():
        conditional_fp_tree_json[item] = fp_tree_to_json(node.node_link)
    return conditional_fp_tree_json


