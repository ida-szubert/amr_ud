#../parser_evaluation/
#                     aligned_amrs.txt.parsed
#                     smatch_scores.txt
#                     smatch_verbose.txt
import ast
import re
from alignment_analysis import read_in_hand_alignments
from utils import contains_element, show_as_string, is_node, print_dict_ordered_by_value, show_as_edges
from parse import draw_graphs, read
from collections import defaultdict
from alignment import normalize_edges
from parse import compare_corenlp_to_fixed_parses, count_corenlp_mistakes
from copy import deepcopy


def parse_verbose_smatch(smatch_file):
    all_data = {}
    current_id = 0
    analysis = {}
    current_amr = 0
    for line in open(smatch_file, "r"):
        if line.startswith("AMR pair") or line.startswith("F-score"):
            if analysis:
                all_data[current_id] = analysis
            current_id += 1
            analysis = {1:[], 2:[], 'amr1mapping':{}, 'mapping':{}, 'missed':[]}
            current_amr = 0
        elif line.startswith("AMR 1 (one-line)"):
            amr1 = line[18:]
            nodes = [n.lstrip('(') for n in re.findall("\([a-z][0-9]*", amr1)]
            renamed_nodes = ["a{}".format(str(i)) for i in range(len(nodes))]
            analysis['amr1mapping'] = {r:n for (r, n) in zip(renamed_nodes, nodes)}
        elif line.startswith("Instance triples of AMR 1"):
            current_amr = 1
        elif line.startswith("Instance triples of AMR 2"):
            current_amr = 2
        elif line.startswith("["):
            analysis[current_amr].extend(ast.literal_eval(line))
        elif line.startswith("Best node mapping alignment"):
            mappings = [re.findall("[a-z][0-9]+", m) for m in line.split()]
            mappings = [x for x in mappings if len(x)>1]
            analysis['mapping'] = {a: b for [a, b] in mappings}
    return all_data


def extract_missed_triples(triple_dict, unlabelled=False, mislabeled=False):
    for data_dict in triple_dict.values():
        mapping = data_dict['mapping']
        amr1 = data_dict[1]
        amr2 = data_dict[2]
        for (rel, parent, child) in amr1:
            # both parent and child are nodes, normal relation
            if re.match("^[a-z][0-9]+$", child):
                if parent in mapping and child in mapping:
                    if unlabelled:
                        # there is no edge between parent and child in amr2
                        if not any([p==mapping[parent] and c==mapping[child] for (r, p, c) in amr2]):
                            data_dict['missed'].append((rel, parent, child))
                    elif mislabeled:
                        # there is an edge between parent and child  but it's not labeled rel
                        if any([p==mapping[parent] and c==mapping[child] for (r, p, c) in amr2]) and \
                                        (rel, mapping[parent], mapping[child]) not in amr2:
                            data_dict['missed'].append((rel, parent, child))
                        # amr2_edges = [(r, p, c) for (r, p, c) in amr2 if p==mapping[parent] and c==mapping[child]]
                        # if not any([r == rel for (r, p, c) in amr2_edges]):

                    else:
                        # there is no edge labeled rel between parent and child in amr2
                        # (no edge at all or a mislabeled one)
                        if (rel, mapping[parent], mapping[child]) not in amr2:
                            data_dict['missed'].append((rel, parent, child))
                else:
                    # either parent or child node is missing from amr2
                    pass
                    # if not mislabeled:
                    #     data_dict['missed'].append((rel, parent, child))
            # only parent is a node; instance triple or non-node child such as 'interrogative' or '-'
            else:
                # parent node is missing from amr2
                if parent not in mapping and rel != "instance":
                    pass
                    # if not mislabeled:
                    #     data_dict['missed'].append((rel, parent, child))
                elif rel != "instance":
                    if unlabelled:
                        if not any([p==mapping[parent] and c==child for (r, p, c) in amr2]):
                            data_dict['missed'].append((rel, parent, child))
                    elif mislabeled:
                        if any([p==mapping[parent] and c==child for (r, p, c) in amr2]) and \
                                        (rel, mapping[parent], child) not in amr2:
                            data_dict['missed'].append((rel, parent, child))
                    else:
                        if (rel, mapping[parent], child) not in amr2:
                            data_dict['missed'].append((rel, parent, child))
    return triple_dict


def find_alignments_for_misses(alignments, misses):
    parser_eval = {}
    for id, amr_data in misses.items():
        node_mapping = amr_data['amr1mapping']
        missed = amr_data['missed']
        aligned_nodes = alignments[id]['alignments']['nodes']
        aligned_paths = alignments[id]['alignments']['paths']
        parser_eval[id] = {}
        parser_eval[id]['amr_graph'] = alignments[id]['amr_graph']
        parser_eval[id]['dep_graph'] = alignments[id]['dep_graph']
        parser_eval[id]['amr_concepts'] = alignments[id]['amr_concepts']
        parser_eval[id]['words'] = alignments[id]['words']
        parser_eval[id]['sentence'] = alignments[id]['sentence']
        parser_eval[id]['missed'] = {}
        missed_aligned = parser_eval[id]['missed']
        for (rel, parent, child) in missed:
            if rel == "TOP":
                continue
            parent_norm = node_mapping[parent]
            child_norm = node_mapping[child] if child in node_mapping else child.rstrip('_')
            rel_norm = ":"+rel
            # case 1: missing nodes
            if rel == 'instance':
                if (parent_norm,) in aligned_nodes:
                    missed_aligned[(rel, parent, child)] = [(parent_norm,), aligned_nodes[(parent_norm,)]]
                else:
                    pass #for now; not sure what to do about unaligned amr nodes that the parser didn't recover
            # case 2: missing edge
            else:
                # normalize child to 'noneX' id if needed
                if not re.match("^[a-z][0-9]*$", child_norm):
                    child_norm = [k for k,v in alignments[id]['amr_concepts'].items() if v == child_norm][0]
                triple_norm = (parent_norm, rel_norm, child_norm)
                if (triple_norm,) in aligned_paths:
                    missed_aligned[(rel, parent, child)] = [(triple_norm,), aligned_paths[(triple_norm,)]]
                else:
                    # find shortest aligned path that contains the missing edge
                    paths = [p for p in aligned_paths if contains_element(p, triple_norm)]
                    if paths:
                        paths.sort(key=len)
                        missed_aligned[(rel, parent, child)] = [paths[0], aligned_paths[paths[0]], triple_norm]
    return parser_eval


def print_out(parser_eval, out_file):
    with open(out_file, "w") as f:
        for id, all_data in parser_eval.items():
            f.write(str(id) + ": " + all_data['sentence']+"\n")
            for amr, alignments in all_data['missed'].items():
                amr_norm = alignments[0]
                dep_set = alignments[1]
                f.write(str(amr)+"\n")
                for dep in dep_set:
                    f.write("\t" + show_as_string(amr_norm, all_data['amr_concepts']) +
                            "    #    " + show_as_string(dep, all_data['words'])+"\n")
            f.write("\n\n")


def prepare_subgraphs_for_drawing(parser_eval):
    chosen_subgraphs = {}
    for id, all_data in parser_eval.items():
        amr_nodes = []
        amr_edges = []
        dep_nodes = []
        dep_edges = []
        things_of_interest = all_data['missed'].values()
        amr_side = [x[0] for x in things_of_interest]
        dep_side = [x[1] for x in things_of_interest]
        word_dict = all_data['words']
        dep_node_ids = {l: str(l) + '/ ' + word_dict[l] for l in word_dict.keys()}
        concept_dict = all_data['amr_concepts']
        amr_node_ids = {l:str(l) + '/ ' + concept_dict[l] for l in concept_dict.keys()}
        for amr_thing in amr_side:
            if is_node(amr_thing):
                amr_nodes.append(amr_node_ids[amr_thing[0]])
            else:
                for (parent, rel, child) in amr_thing:
                    amr_edges.append((amr_node_ids[parent], amr_node_ids[child]))
        for dep_set in dep_side:
            for dep_thing in dep_set:
                if is_node(dep_thing):
                    dep_nodes.append(dep_node_ids[dep_thing[0]])
                else:
                    for (parent, rel, child) in dep_thing:
                        dep_edges.append((dep_node_ids[parent], dep_node_ids[child]))
        chosen_subgraphs[id] = [amr_nodes, amr_edges, dep_nodes, dep_edges]
    return chosen_subgraphs


def composite_freq_of_misses(parser_eval, normalize_dep=True):
    relation_freq = defaultdict(int)
    for id, all_data in parser_eval.items():
        missed = all_data['missed'].values()
        for m in [x for x in missed if len(x) >1]:
            for dep in m[1]:
                if not is_node(dep):
                    dep = normalize_edges(dep, all_data['dep_graph'])
                    # key = [normalize_dep_label(e[1]) for e in dep] if normalize_dep else [e[1] for e in dep]
                    key = list(set([normalize_dep_label(e[1]) for e in dep] if normalize_dep else [e[1] for e in dep]))
                    key.sort()
                    relation_freq[tuple(key)] += 1
    return relation_freq


def freq_of_misses(parser_eval, normalize_dep=True):
    relation_freq = defaultdict(int)
    for id, all_data in parser_eval.items():
        missed = all_data['missed'].values()
        problematic_edges = set([])
        # pick only those missed AMR subgraphs that are aligned to some UD subgraph
        for m in [x for x in missed if len(x) >1]:
            # look at the set of aligned UD subgraphs
            for dep in m[1]:
                if not is_node(dep):
                    dep = normalize_edges(dep, all_data['dep_graph'])
                    problematic_edges = problematic_edges.union(list(dep))
        for edge in problematic_edges:
            if normalize_dep:
                relation_freq[normalize_dep_label(edge[1])] += 1
            else:
                relation_freq[edge[1]] += 1
    return relation_freq


def rel_freq_of_misses(parser_eval, normalize_dep=True, composite=False, alignments=None):
    if composite:
        all_freq = freq_of_aligned_dep_subgraphs(alignments, normalize_dep)
        missed_freq = composite_freq_of_misses(parser_eval)
    else:
        all_freq = freq_of_deps(parser_eval, normalize_dep)
        missed_freq = freq_of_misses(parser_eval, normalize_dep=normalize_dep)
    return {dep: (n, round(float(n)/all_freq[dep], 3)) for (dep, n) in missed_freq.items()}


def freq_of_aligned_dep_subgraphs(alignments, normalize_dep=True):
    subgraph_freq = defaultdict(int)
    for s in manual:
        aligned = manual[s]['alignments']['nodes'].values()
        aligned.extend(manual[s]['alignments']['paths'].values())
        for a_set in aligned:
            for a in a_set:
                if not is_node(a):
                    a = normalize_edges(a, manual[s]['dep_graph'])
                    # key = [normalize_dep_label(e[1]) for e in a] if normalize_dep else [e[1] for e in a]
                    key = list(set([normalize_dep_label(e[1]) for e in a] if normalize_dep else [e[1] for e in a]))
                    key.sort()
                    subgraph_freq[tuple(key)] += 1
    return subgraph_freq


def freq_of_deps(parser_eval, normalize_dep=True):
    all_relation_freq = defaultdict(int)
    for id, all_data in parser_eval.items():
        dep_graph = all_data['dep_graph']
        for dep in dep_graph.values():
            if normalize_dep:
                all_relation_freq[normalize_dep_label(dep)] += 1
            else:
                all_relation_freq[dep] += 1
    return all_relation_freq


def normalize_dep_label(dep, total=False):
    if not total and dep in [':compound-prt', ':acl-relcl', ':nmod-tmod']:
        return dep
    else:
        return dep.split('-')[0]


def read_in_corenlp_parses(data_file):
    corenlp_parses = {}
    index = 1
    sentence_next = False
    s = ""
    parse = {}
    for line in open(data_file, "r"):
        if line.startswith("Sentence"):
            sentence_next = True
            if s:
                corenlp_parses[index] = {"sentence": s, "dep_graph": parse}
                index += 1
                parse = {}
        elif not(line.startswith("[") or line.startswith("(") or line.startswith(" ") or line=="\n"):
            if sentence_next:
                s = line.strip()
                sentence_next = False
            else:
                dep, nodes = line.strip().split("(",1)
                parent, child = [int(n.rstrip(")").rstrip("\'").split("-")[-1]) for n in nodes.split(", ")]
                parse[(parent, child)] = ":ROOT" if dep == "root" else ":" + dep
    return corenlp_parses


def filter_out_parser_mistakes(eval_alignments, corenlp_mistakes):
    filtered_eval_alignments = deepcopy(eval_alignments)
    for s_id, s_dict in eval_alignments.items():
        to_ignore = corenlp_mistakes[s_id]['added_v']
        to_ignore.extend(corenlp_mistakes[s_id]['label_change_v'])
        # to_exclude = []
        for edge, info in s_dict['missed'].items():
            dep_set = info[1]
            filtered_dep_set = filtered_eval_alignments[s_id]['missed'][edge][1]
            # new_dep_set = deepcopy(dep_set)
            for dep in dep_set:
                if any([e in to_ignore for e in dep]):
                    try:
                        filtered_dep_set.remove(dep)
                    except KeyError:
                        pass
                    # new_dep_set.remove(dep)
            if not filtered_dep_set:
                del filtered_eval_alignments[s_id]['missed'][edge]
    return filtered_eval_alignments


def show_examples(parser_eval, target_dep, normalize_dep=True, target_amr=None):
    examples = find_examples(parser_eval, target_dep, normalize_dep, target_amr)
    print("\nExamples of AMR parsing failing on {}".format(target_dep))
    for s_id, s_dict in examples.items():
        print("\n" + str(s_id) + ": " + s_dict['sentence'])
        for edge, info, dep in s_dict['examples']:
            dep = tuple([(p, normalize_dep_label(l, total=True), c) for (p, l, c) in dep])
            dep_verbose = show_as_edges(dep, s_dict['words'])
            amr_verbose = show_as_edges(info[0], s_dict['amr_concepts'])
            if len(info) > 2:
                amr_verbose = ["** "+amr_verbose[i]+" **" if info[2]==info[0][i] else amr_verbose[i] for i in range(len(amr_verbose))]
            print("\t" + "AMR: {}".format(', '.join(amr_verbose)) + "\t\t\tUD: {}".format(', '.join(dep_verbose)))


def find_examples(parser_eval, target_dep, normalize_dep=True, target_amr=None):
    target = ast.literal_eval(target_dep)
    examples = {}
    for s_id, s_dict in parser_eval.items():
        s_id_examples = []
        for edge, info in s_dict['missed'].items():
            for dep in info[1]:
                if not is_node(dep):
                    # key = [normalize_dep_label(e[1]) for e in dep] if normalize_dep else [e[1] for e in dep]
                    key = list(set([normalize_dep_label(e[1]) for e in dep] if normalize_dep else [e[1] for e in dep]))
                    key.sort()
                    if tuple(key) == target:
                        if target_amr:
                            amr_rel = info[2][1] if len(info) > 2 else info[0][0][1]
                            if target_amr == amr_rel:
                                s_id_examples.append((edge, info, dep))
                        else:
                            s_id_examples.append((edge, info, dep))
        if s_id_examples:
            examples[s_id] = {'sentence': parser_eval[s_id]['sentence'],
                              'words': parser_eval[s_id]['words'],
                              'amr_concepts': parser_eval[s_id]['amr_concepts'],
                              'examples': s_id_examples}
    return examples


def stats_for_target(parser_eval, target_dep, normalize_dep=True, target_amr=None):
    examples = find_examples(parser_eval, target_dep, normalize_dep, target_amr)
    freq = defaultdict(int)
    for s_dict in examples.values():
        for edge, info, dep in s_dict['examples']:
            amr_rel = info[2][1] if len(info) > 2 else info[0][0][1]
            freq[amr_rel] += 1
    return freq

amr_data = parse_verbose_smatch("../parser_evaluation/smatch_verbose1.txt")
extracted_misses = extract_missed_triples(amr_data, unlabelled=True, mislabeled=False)
manual = read_in_hand_alignments('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p')
eval_alignments = find_alignments_for_misses(manual, extracted_misses)
# #print_out(eval_alignments, "../parser_evaluation/missed_subgraphs.txt")
# chosen_subgraphs = prepare_subgraphs_for_drawing(eval_alignments)
# draw_graphs(eval_alignments, '../parser_evaluation/graphs', special_subgraphs=chosen_subgraphs)
# auto_amrs = read("../parser_evaluation/aligned_amrs.txt.parsed1")
# draw_graphs(auto_amrs, '../parser_evaluation/graphs', want_dep=False)
#
# # print("Dependency edges present in subgraphs corresponding to missed AMR subgraphs")
# # print("\n\tProportion of all dependencies with that label:\n")
# relative_miss_freq = rel_freq_of_misses(eval_alignments)
# print_dict_ordered_by_value(relative_miss_freq)

# print("Dependency edges corresponding to missed AMR edges\n\tcases of single UD edge responsible for the mistake\n")
# miss_freq_single_edge = rel_freq_of_misses(eval_alignments, composite=True, alignments=manual)
# print_dict_ordered_by_value(miss_freq_single_edge)


marcos_parses = read_in_corenlp_parses("../parser_evaluation/aligned_amrs.txt.out")
corenlp_mistakes = count_corenlp_mistakes('../data/aligned_amrs.txt', '../data/aligned_parses.txt',
                                          automatic_parse_dict=marcos_parses, manual_parse_dict=manual, verbose=True,
                                          norm_dep=True)


filtered_eval_alignments = filter_out_parser_mistakes(eval_alignments, corenlp_mistakes)
print("Analysis limited to UD subgraphs present in CoreNLP parses\n\n")
print("Dependency subgraphs corresponding to missed AMR subgraphs\n")
# miss_freq_subgraph = rel_freq_of_misses(filtered_eval_alignments, composite=True, alignments=manual)
miss_freq_subgraph = rel_freq_of_misses(eval_alignments, composite=True, alignments=manual)
print_dict_ordered_by_value(miss_freq_subgraph)


target = "(':nsubj',)"
target_amr = ":ARG0"
# show_examples(filtered_eval_alignments, target)
# show_examples(eval_alignments, target)
# show_examples(filtered_eval_alignments, target, target_amr=target_amr)
show_examples(eval_alignments, target, target_amr=target_amr)
# corresponding_amr_freq = stats_for_target(filtered_eval_alignments, target)
corresponding_amr_freq = stats_for_target(eval_alignments, target)
print_dict_ordered_by_value(corresponding_amr_freq)

print('booooooo')


