import pickle
import cPickle


def print_dict_ordered_by_value(dict):
    #if values are tuples, sort by the first element of the tuple
    if isinstance(dict.values()[0], tuple):
        sorted_items = sorted(dict.items(), key=lambda x: x[1][0], reverse=True)
    else:
        sorted_items = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_items:
        print str(k) + "   :   " + str(v)


def print_dict_for_table(dict):
    if isinstance(dict.values()[0], tuple):
        sorted_items = sorted(dict.items(), key=lambda x: x[1][0], reverse=True)
    else:
        sorted_items = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_items:
        if isinstance(v, float):
            print str(k) + "   &   " + '%.4f' % v + "\\"
        elif isinstance(v, tuple):
            print str(k) + "   &   " + str(v[0]) + "   &   " + '%.4f' % v[1] + "\\"
        else:
            print str(k) + "   &   " + str(v) + "\\"
    if isinstance(dict.values()[0], int):
        print(sum(dict.values()))


def recursive_flatten_list(l):
    if not any([isinstance(x, list) for x in l]):
        return l
    if not any([not any([isinstance(x, list) for x in sublist]) for sublist in l]):
        return flatten_list(l)
    else:
        recursive_flatten_list([recursive_flatten_list(sublist) for sublist in l])


def flatten_list(l):
    if len(l) == 0:
        return l
    if not (isinstance(l[0], list) or isinstance(l[0], set)):
        return l
    else:
        return [item for sublist in l for item in sublist]


def extract_nodes(structure):
    nodes = set([])
    if not is_node(structure):
        for edge in structure:
            nodes.add(edge[0])
            nodes.add(edge[2])
    else:
        nodes.add(structure[0])
    return nodes


def path_contains_node(path, node):
    for edge in path:
        if node in edge:
            return True
    return False


def contains_element(group, x):
    if not isinstance(x[0], tuple):
        return x in group
    else:
        for y in group:
            if equal_paths(y, x):
                return True
        return False


def equal_paths(p1, p2):
    return set(p1) == set(p2)


def equivalent_element(group, x):
    if not isinstance(x[0], tuple):
        if x in group:
            return x
        else:
            return None
    else:
        for y in group:
            if equal_paths(y, x):
                return y
        return None


def is_path(edges):
    node_dict = {}
    for parent, relation, child in edges:
        if parent in node_dict:
            node_dict[parent] += 1
        else:
            node_dict[parent] = 1
        if child in node_dict:
            node_dict[child] += 1
        else:
            node_dict[child] = 1
        if node_dict[parent] > 2 or node_dict[child] > 2:
            return False
    return True


def insert_to_dict(dict, key, value):
    if value not in dict[key]:
        dict[key].add(value)


def is_node(structure):
    return not isinstance(structure[0], tuple)


def is_edge_label(x):
    return str(x).startswith(':')


def clean_edge_label(x):
    return x.lstrip(':')


def expand_node_label(node_id, node_dict):
    if isinstance(node_id, tuple):
        return str(node_id[0]) + '/' + node_dict[node_id[0]]
    else:
        return str(node_id) + '/' + node_dict[node_id]


def contract_node_label(node_label):
    id = node_label.split('/')[0]
    return int(id) if id.isdigit() else id


def get_children(node, graph):
    return [child for parent, child in graph.keys() if parent == node and graph[(parent, child)]]


def get_parents(node, graph):
    return [parent for parent, child in graph.keys() if child == node and graph[(parent, child)]]


def get_top_nodes(structure, graph):
    nodes = extract_nodes(structure)
    parents = [get_parents(n, graph) for n in nodes]
    has_parents = [any([x in nodes for x in p]) for p in parents]
    return [n for n, flag in zip(nodes, has_parents) if not flag]


def reverse_edges_in_path(path):
    return [reverse_edge_label(x) if is_edge_label(x) else x for x in path]


def reverse_edge_label(label):
    if label[-3:] == '-of':
        return label[:-3]
    else:
        return label + '-of'


def make_structure_readable(structure, node_dict):
    if is_node(structure):
        return [expand_node_label(structure, node_dict)]
    else:
        return [[expand_node_label(n, node_dict) if not is_edge_label(n) else n for n in edge] for edge in structure]


def show_as_string(structure, node_dict):
    expanded = make_structure_readable(structure, node_dict)
    if is_node(structure):
        return expanded[0]
    else:
        return '; '.join([' '.join(edge) for edge in expanded])


def show_as_edges(structure, node_dict):
    expanded = make_structure_readable(structure, node_dict)
    if not isinstance(expanded[0], list):
        return expanded
    else:
        return [' '.join(edge) for edge in expanded]


def levenshtein_distance(word1, word2):
    """
    Calculates the edit distance between word1 (longer) and word2 (shorter)
    :param word1
    :param word2
    :return edit distance
    """
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)
    if len(word2) == 0:
        return len(word1)

    previous_row = range(len(word2) + 1)
    for i, c1 in enumerate(word1):
        current_row = [i + 1]
        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def pickle(sent_dict, filename):
    f = open(filename, 'wb')
    cPickle.dump(sent_dict, f)
    f.close()


def unpickle(filename):
    f = open(filename, 'rb')
    sent_dict = cPickle.load(f)
    f.close()
    return sent_dict


# def pickle_parsed_data(sent_dict, filename):
#     f = open(filename, 'w')
#     cPickle.dump(sent_dict, f)
#     f.close()
#
#
# def unpickle_parsed_data(filename):
#     f = open(filename, 'r')
#     sent_dict = cPickle.load(f)
#     f.close()
#     return sent_dict


def sentence_len_freq(amr_file):
    up_to_10 = 0
    up_to_20 = 0
    up_to_30 = 0
    up_to_40 = 0
    up_to_50 = 0
    over_50 = 0
    for line in open(amr_file, "r"):
        split_line = line.split()
        if split_line:
            if split_line[1] == "::snt":
                length = len(split_line[2:])
                if length <= 10:
                    up_to_10 += 1
                    print ' '.join(split_line[2:])
                elif length <= 20:
                    up_to_20 += 1
                    print ' '.join(split_line[2:])
                elif length <= 30:
                    up_to_30 += 1
                elif length <= 40:
                    up_to_40 += 1
                elif length <= 50:
                    up_to_50 += 1
                else:
                    over_50 += 1
    print ("up_to_10: " + str(up_to_10) + "\nprint up_to_20: " + str(up_to_20) + "\nprint up_to_30: " + str(up_to_30) +
           "\nprint up_to_40: " + str(up_to_40) + "\nprint up_to_50: " + str(up_to_50) + "\nover_50: " + str(over_50))


def write_out_parses(sent_dict, filename):
    out_file = open(filename, 'w')
    for s_id in sent_dict:
        sentence_data = sent_dict[s_id]
        dependencies = sentence_data['dep_graph']
        lemmas = sentence_data['lemmas']
        words = sentence_data['words']
        id_pairs = sorted(dependencies.keys(), key=lambda x: x[1])
        for parent, child in id_pairs:
            out_file.write(str(child) + '\t' + words[child] + '\t' + lemmas[child] +
                           '\t_\t_\t_\t' + str(parent) + '\t' + dependencies[(parent, child)] + '\t_\t_\n')
        out_file.write('\n')
    out_file.close()


def filter_corpus(whole="../data/corpus/split/training/amr-release-1.0-training-all_prefilter.txt",
                  selected="../data/random_amrs.txt"):
    already_used = []
    for line in open(selected, "r"):
        split_line = line.split()
        if split_line:
            if split_line[1] == "::id":
                already_used.append(split_line[2])
    ignore = False
    f = open("../data/corpus/split/training/amr-release-1.0-training-all.txt", "w")
    filtered = []
    for line in open(whole, "r"):
        split_line = line.split()
        if split_line:
            if split_line[1] == "::id":
                if split_line[2] in already_used:
                    ignore = True
                    filtered.append(split_line[2])
                else:
                    ignore = False
        if not ignore:
            if not line.startswith("# AMR release"):
                f.write(line)
    f.close()
    print len(filtered)
