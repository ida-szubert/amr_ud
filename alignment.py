import copy
import collections
from num2words import num2words
import morfessor
from utils import *


def read_neg_polarity_items(file_name):
    neg_dict = {}
    n = open(file_name, 'r')
    for line in n:
        if line.startswith('"'):
            negative_word = line.split()[0].rstrip('"').lstrip('"')
            positive_word = line.split()[1].rstrip('"').lstrip('"')
            neg_dict[negative_word] = positive_word
    return neg_dict


def get_node_alignment(sentences, neg_dict):
    return get_easy_alignment_points(sentences, neg_dict)


def get_full_alignment(sentences, neg_dict):
    return get_inferred_alignment_points(get_easy_alignment_points(sentences, neg_dict))


def get_full_alignment_oracle(manual_alignments):
    for s_id in manual_alignments:
        s = manual_alignments[s_id]
        if s_id == 2:
            pass
        add_entity_type_alignments(s['alignments'], s['amr_concepts'], s['amr_graph'])
        get_edge_alignments(s['alignments'], s['amr_graph'], s['dep_graph'])
    return get_inferred_alignment_points(manual_alignments)

#######################
# NODE ALIGNMENT
#######################


def get_easy_alignment_points(sentences, neg_dict):
    """
    Finds node alignments based on edit distance between AMR concepts and lemmas,
    and simple edge alignments (whenever two directly connected nodes in AMR align to two directly connected nodes in
    DG, align the edges as well)
    :param sentences: a dictionary whose keys are numeric sentence ids and values are tuples of the form
    (sentence, amr_graph, amr_concepts, dependency_graph, lemmas)
    :return: input dictionary augmented with alignments; alignments are stored in a dictionary, with entries for
    node, edge, and inferred alignments
    """
    io = morfessor.MorfessorIO()
    morphology_model = io.read_binary_model_file('segmentation_model.bin')
    for s_id in sentences.keys():
        alignments = {'nodes': collections.defaultdict(set),
                      'paths': collections.defaultdict(set)}
        s = sentences[s_id]
        ###HERE CHANGE NODE ALIGNMENT MODES###
        # version 2 gets better precision, but worse recall
        # much worse results on path finding
        # get_node_alignments_v2(alignments, s['amr_concepts'], s['lemmas'], s['amr_graph'], s['dep_graph'], neg_dict, morphology_model)
        get_node_alignments(alignments, s['amr_concepts'], s['lemmas'], s['amr_graph'], s['dep_graph'], neg_dict, morphology_model)
        get_edge_alignments(alignments, s['amr_graph'], s['dep_graph'])
        sentences[s_id]['alignments'] = alignments
    return sentences


def get_node_alignments(alignments, amr_concepts, lemmas, amr_relations, dep_relations, neg_dict, morph_model):
    for c_id in amr_concepts:
        for l_id in lemmas.keys():
            should_align = align(c_id, l_id, neg_dict, amr_concepts, lemmas, amr_relations, dep_relations, morph_model)
            if should_align:
                alignments['nodes'][(c_id,)].add((l_id,))
    # light_verbs = ['take', 'have', 'make', 'did', 'get', 'give', 'carry']
    # for l_id in lemmas:
    #     if lemmas[l_id] in light_verbs and l_id not in flatten_list([dep_list for (amr, dep_list) in alignments['nodes'].items()]):
    #         extended_alignments(l_id, dep_relations, alignments['nodes'])
    add_particle_node_alignments(alignments, amr_concepts, lemmas, amr_relations, dep_relations)
    add_obvious_edge_alignments(alignments, dep_relations)
    add_entity_type_alignments(alignments, amr_concepts, amr_relations)


def align(node1_id, node2_id, neg_dict, node_dict1, node_dict2, graph1, graph2, morph_model):
    node1 = node_dict1[node1_id]
    node2 = node_dict2[node2_id]
    # node2_edges = [e_lab for (n1, n2), e_lab in graph2.items() if n2 == node2_id]
    if similar_strings(node1, node2):
        return True
    # node1 is positive counterpart of node2
    elif aligned_through_neg_dict(node1, node2, morph_model, neg_dict):
        return True
    # node1 is '-' and node2 is a negation-expressing word
    elif aligned_negation_nodes(node1, node2):
        return True
    # node1 has complex label and node2 corresponds to part of that label
    elif complex_node_label(node1):
        if any([(likely_same_root(x, node2, morph_model) or similar_strings(x, node2)) for x in node1.split('-')]):
            return True
    elif complex_node_label(node2):
        if any([(likely_same_root(node1, x, morph_model) or similar_strings(node1, x)) for x in node2.split('-')]):
            return True
    # node1 is '-' and node2 is aligned to a parent of node1 through neg_dict
    elif node1 == '-':
        node1_parents = [node_dict1[parent] for parent, child in graph1 if child == node1_id]
        if any([aligned_through_neg_dict(parent, node2, morph_model, neg_dict) for parent in node1_parents]):
            return True
    # it's likely, based on morphological segmentation, that node1 and node2 align
    elif likely_same_root(node1, node2, morph_model):
        return True


def add_particle_node_alignments(alignments, amr_concepts, lemmas, amr_relations, dep_relations):
    node_alignments = alignments['nodes']
    for c_id, l_ids in node_alignments.items():
        current_alignments = copy.deepcopy(l_ids)
        for (l_id,) in current_alignments:
            particle_children = [n2 for (n1, n2), e_lab in dep_relations.items() if n1 == l_id and e_lab == ":compound-prt"]
            for child in particle_children:
                l_ids.add((child,))


def add_entity_type_alignments(alignments, amr_concepts, amr_relations):
    n_alignments = alignments['nodes']
    for (n, ), deps in n_alignments.items():
        name_parents = [np for np in get_parents(n, amr_relations) if amr_concepts[np] == 'name']
        if name_parents:
            name_node = name_parents[0]
            if len(get_children(name_node, amr_relations)) == 1:
                entity_parents = [ep for ep in get_parents(name_node, amr_relations) if amr_concepts[ep] in ontology]
                for e in entity_parents:
                    path = tuple([(name_node, ':op1', n), (e, ':name', name_node)])
                    alignments['paths'][path] = deps


def add_obvious_edge_alignments(alignments, dep_relations):
    """
    When an AMR node align to two connected DG nodes, add the edge between them to the alignments of
    the AMR node.
    Cases: phrasal verbs, multi-word expressions
    :param alignments:
    :param dep_relations:
    :return:
    """
    align_copy = copy.deepcopy(alignments['nodes'])
    for c_id, l_list in align_copy.items():
        if len(l_list) > 1:
            for l1 in l_list:
                for l2 in l_list:
                    if l1 != l2:
                        if (l1[0], l2[0]) in dep_relations:
                            alignments['nodes'][c_id].add(((l1[0], dep_relations[(l1[0], l2[0])], l2[0]),))
                        if (l2[0], l1[0]) in dep_relations:
                            alignments['nodes'][c_id].add(((l2[0], dep_relations[(l2[0], l1[0])], l1[0]),))


def complex_node_label(node_label):
    return node_label != '-' and len(node_label.split('-')) > 1


def extended_alignments(dep_id, dep_graph, node_alignments):
    dep_id_children = [child for (parent, child) in dep_graph if parent == dep_id]
    for child in dep_id_children:
        for amr in node_alignments:
            if (child, ) in node_alignments[amr]:
                edge_label = dep_graph[dep_id, child]
                if 'obj' in edge_label:
                    node_alignments[amr].add(((dep_id, dep_graph[dep_id, child], child),))


def aligned_negation_nodes(node1, node2):
    negation_words = ['no', 'none', 'not', 'never']
    return node2 in negation_words and node1 == '-'


def aligned_through_neg_dict(node1, node2, morph_model, neg_dict):
    if node2 in neg_dict.keys():
        if node1 == neg_dict[node2] or likely_same_root(node1, neg_dict[node2], morph_model):
            return True


def similar_strings(node1, node2):
    distance = levenshtein_distance(node1.lower(), node2.lower())
    longer = node1 if len(node1) > len(node2) else node2
    if len(longer) == 0:
        pass
    if (float(distance) / len(longer)) < 0.15:
        return True
    elif node1.isdigit():
        expanded_node1 = expand_number(float(node1))
        distance = levenshtein_distance(expanded_node1, node2.lower())
        longer = expanded_node1 if len(expanded_node1) > len(node2) else node2
        if (float(distance) / len(longer)) < 0.15:
            return True
    else:
        return False


def expand_number(number):
    return num2words(number)


def likely_same_root(node1, node2, morph_model):
    morphemes1 = morph_model.viterbi_segment(node1.lower())[0]
    morphemes2 = morph_model.viterbi_segment(node2.lower())[0]
    if not (morphemes1 and morphemes2):
        return False
    if len(morphemes1) == 1 and len(morphemes2) < 3:
        if morphemes1[0] in morphemes2:
            return True
    if len(morphemes2) == 1 and len(morphemes1) < 3:
        if morphemes2[0] in morphemes1:
            return True
    if len(morphemes1) == 2 and len(morphemes2) == 2:
        if morphemes1[0] in morphemes2 or morphemes2[0] in morphemes1:
            return True
    else:
        return False


############################
# NODE ALIGNMENT: 2 STEP
############################

def get_node_alignments_v2(alignments, amr_concepts, lemmas, amr_relations, dep_relations, neg_dict, morph_model):
    amr_possible, dep_possible = \
        get_node_alignments_step1(alignments, amr_concepts, lemmas, amr_relations, dep_relations, neg_dict, morph_model)
    if amr_possible or dep_possible:
        pass
    get_node_alignments_step2(alignments, amr_concepts, lemmas, amr_relations, dep_relations, amr_possible, dep_possible)
    add_obvious_edge_alignments(alignments, dep_relations)


def get_node_alignments_step2(alignments, amr_concepts, lemmas, amr_relations, dep_relations, amr_possible, dep_possible):
    amr2dep = alignments["nodes"]
    dep2amr = alignments["reversed"]["nodes"]
    reintroduce_node_alignments(amr2dep, dep2amr, amr_concepts, lemmas, amr_relations, dep_relations, amr_possible)
    reintroduce_node_alignments(dep2amr, amr2dep, lemmas, amr_concepts, dep_relations, amr_relations, dep_possible)


def reintroduce_node_alignments(s2t, t2s, source_lexicon, target_lexicon, source_graph, target_graph, possibilities):
    change_made = False
    for s_node in possibilities:
        candidate_t = copy.deepcopy(possibilities[s_node])
        # find closest aligned AMR nodes
        # s_children = [n2 for (n1, n2) in source_graph if n1 == s_node[0] and len(s2t[(n2,)])>0]
        # try:
        s_children = get_closest_aligned(s_node, source_graph, s2t, "children", [])
        # except RuntimeError:
        #     s_children = get_closest_aligned(s_node, source_graph, s2t, "children")
        # s_parents = [n1 for (n1, n2) in source_graph if n2 == s_node[0] and len(s2t[(n1,)])>0]
        s_parents = get_closest_aligned(s_node, source_graph, s2t, "parents", [])
        children_alignments = flatten_list([[y[0] for y in s2t[(x,)]] for x in s_children])
        parent_alignments = flatten_list([[y[0] for y in s2t[(x,)]] for x in s_parents])
        for t_node in candidate_t:
            # t_children = [n2 for (n1, n2) in target_graph if n1 == t_node[0] and len(t2s[(n2,)])>0]
            t_children = get_closest_aligned(t_node, target_graph, t2s, "children", [])
            # t_parents = [n1 for (n1, n2) in target_graph if n2 == t_node[0] and len(t2s[(n1,)])>0]
            t_parents = get_closest_aligned(t_node, target_graph, t2s, "parents", [])
            children_match = any([x in children_alignments for x in t_children])
            parents_match = any([x in parent_alignments for x in t_parents])
            child_parent_match = any([x in parent_alignments for x in t_children])
            parent_child_match = any([x in children_alignments for x in t_parents])
            if children_match or parents_match or child_parent_match or parent_child_match:
                s2t[s_node].add(t_node)
                t2s[t_node].add(s_node)
                possibilities[s_node].remove(t_node)
                change_made = True
    if change_made:
        reintroduce_node_alignments(s2t, t2s, source_lexicon, target_lexicon, source_graph, target_graph, possibilities)


def get_closest_aligned(node, graph, alignments, direction, checked):
    # print(direction+"\t"+str(node[0]))
    if direction == "parents":
        neighbours = [n1 for (n1, n2) in graph if n2 == node[0] and n1 not in checked]
    elif direction == "children":
        neighbours = [n2 for (n1, n2) in graph if n1 == node[0] and n2 not in checked]
    checked.extend(neighbours)
    aligned_neighbours = []
    for n in neighbours:
        if len(alignments[(n,)])>0:
            aligned_neighbours.append(n)
        else:
            aligned_neighbours.extend(get_closest_aligned((n,), graph, alignments, direction, checked))
    return aligned_neighbours


def get_node_alignments_step1(alignments, amr_concepts, lemmas, amr_relations, dep_relations, neg_dict, morph_model):
    for c_id in amr_concepts:
        for l_id in lemmas.keys():
            should_align = align(c_id, l_id, neg_dict, amr_concepts, lemmas, amr_relations, dep_relations, morph_model)
            if should_align:
                alignments['nodes'][(c_id,)].add((l_id,))

    add_reversed_node_alignment_dict(alignments)
    amr_possible, dep_possible = filter_multiple_node_alignments(alignments, amr_relations, dep_relations)
    add_particle_node_alignments(alignments, amr_concepts, lemmas, amr_relations, dep_relations)
    light_verbs = ['take', 'have', 'make', 'did', 'get', 'give', 'carry']
    for l_id in lemmas:
        if lemmas[l_id] in light_verbs and l_id not in flatten_list([dep_list for (amr, dep_list) in alignments['nodes'].items()]):
            extended_alignments(l_id, dep_relations, alignments['nodes'])
    return amr_possible, dep_possible


def add_reversed_node_alignment_dict(alignments):
    reversed_a = {'nodes': collections.defaultdict(set),
                  'paths': collections.defaultdict(set)}
    for c_id, l_ids in alignments["nodes"].items():
        for l_id in l_ids:
            reversed_a["nodes"][l_id].add(c_id)
    alignments["reversed"] = reversed_a


def filter_multiple_node_alignments(alignments, amr_relations, dep_relations):
    amr_possible = filter_multiple(alignments["nodes"], dep_relations)
    dep_possible = filter_multiple(alignments["reversed"]["nodes"], amr_relations)
    reconcile_filtered_alignments(alignments, alignments["reversed"])
    return amr_possible, dep_possible


def filter_multiple(alignments, target_graph):
    possible_alignments = {}
    for node in alignments:
        if len(alignments[node]) > 1:
            possible_alignments[node] = alignments[node]
            alignments[node] = set([])
            for (t,) in possible_alignments[node]:
                t_neighbours = get_parents(t, target_graph)
                t_neighbours.extend(get_children(t, target_graph))
                if any([t2 in t_neighbours for (t2,) in possible_alignments[node]]):
                    alignments[node].add((t,))
            possible_alignments[node] = possible_alignments[node].difference(alignments[node])
    return possible_alignments


def reconcile_filtered_alignments(alignments, reversed_alignments):
    amr2dep = alignments["nodes"]
    dep2amr = reversed_alignments["nodes"]
    for amr_node in amr2dep:
        deps = copy.deepcopy(amr2dep[amr_node])
        for dep in deps:
            if amr_node not in dep2amr[dep]:
                amr2dep[amr_node].remove(dep)
    for dep_node in dep2amr:
        amrs = copy.deepcopy(dep2amr[dep_node])
        for amr in amrs:
            if dep_node not in amr2dep[amr]:
                dep2amr[dep_node].remove(amr)


#######################
# EDGE ALIGNMENT
#######################


def get_edge_alignments(alignments, amr_edge_dict, dep_edge_dict):
    """
    Given two graphs and an alignment between their nodes, extracts edges which are also aligned.
    :param alignments: dictionary storing alignments for a given AMR-DG pair
    :param amr_edge_dict: dictionary of AMR edges
    :param dep_edge_dict: dictionary of DG edges
    :return: updates the alignments, no return value.
    """
    node_pairs = create_node_pairs(alignments, limited=False)
    for amr1, dep1, amr2, dep2 in node_pairs:
        amr_1_2_edge = ((amr1, amr_edge_dict[(amr1, amr2)], amr2),) if (amr1, amr2) in amr_edge_dict else ''
        amr_2_1_edge = ((amr2, amr_edge_dict[(amr2, amr1)], amr1),) if (amr2, amr1) in amr_edge_dict else ''
        if isinstance(dep1, int) and isinstance(dep2, int):
            dep_1_2_edge = ((dep1, dep_edge_dict[(dep1, dep2)], dep2),) if (dep1, dep2) in dep_edge_dict else ''
            dep_2_1_edge = ((dep2, dep_edge_dict[(dep2, dep1)], dep1),) if (dep2, dep1) in dep_edge_dict else ''
        elif isinstance(dep1, int):
            dep_1_2_edge_part = dep_edge_dict[(dep1, dep2[0])] if (dep1, dep2[0]) in dep_edge_dict else ''
            dep_1_2_edge = ((dep1, dep_1_2_edge_part, dep2[0]), dep2) if dep_1_2_edge_part else ''
            dep_2_1_edge_part = dep_edge_dict[(dep2[2], dep1)] if (dep2[2], dep1) in dep_edge_dict else ''
            dep_2_1_edge = (dep2, (dep2[2], dep_2_1_edge_part, dep1)) if dep_2_1_edge_part else ''
        elif isinstance(dep2, int):
            dep_1_2_edge_part = dep_edge_dict[(dep1[2], dep2)] if (dep1[2], dep2) in dep_edge_dict else ''
            dep_1_2_edge = (dep1, (dep1[2], dep_1_2_edge_part, dep2)) if dep_1_2_edge_part else ''
            dep_2_1_edge_part = dep_edge_dict[(dep2, dep1[0])] if (dep2, dep1[0]) in dep_edge_dict else ''
            dep_2_1_edge = ((dep2, dep_2_1_edge_part, dep1[0]), dep1) if dep_2_1_edge_part else ''
        else:
            dep_1_2_edge_part = dep_edge_dict[(dep1[2], dep2[0])] if (dep1[2], dep2[0]) in dep_edge_dict else ''
            dep_1_2_edge = (dep1, (dep1[2], dep_1_2_edge_part, dep2[0]), dep2) if dep_1_2_edge_part else ''
            dep_2_1_edge_part = dep_edge_dict[(dep2[2], dep1[0])] if (dep2[2], dep1[0]) in dep_edge_dict else ''
            dep_2_1_edge = (dep2, (dep2[2], dep_2_1_edge_part, dep1[0]), dep1) if dep_2_1_edge_part else ''
        if amr_1_2_edge and dep_1_2_edge:
            new_dep_1_2_edge = tuple(check_case(dep_1_2_edge, dep_edge_dict))
            insert_to_dict(alignments['paths'], amr_1_2_edge, new_dep_1_2_edge)
        if amr_2_1_edge and dep_2_1_edge:
            new_dep_2_1_edge = tuple(check_case(dep_2_1_edge, dep_edge_dict))
            insert_to_dict(alignments['paths'], amr_2_1_edge, new_dep_2_1_edge)
        if amr_1_2_edge and dep_2_1_edge:
            new_dep_2_1_edge = tuple(check_case(dep_2_1_edge, dep_edge_dict))
            insert_to_dict(alignments['paths'], amr_1_2_edge, new_dep_2_1_edge)
        if amr_2_1_edge and dep_1_2_edge:
            new_dep_1_2_edge = tuple(check_case(dep_1_2_edge, dep_edge_dict))
            insert_to_dict(alignments['paths'], amr_2_1_edge, new_dep_1_2_edge)


def create_node_pairs(alignments, limited=True):
    node_pairs = []
    aligned_nodes = [n for n in alignments['nodes'].keys() if alignments['nodes'][n]]
    for n in aligned_nodes:
        for m in aligned_nodes:
            for n_i in alignments['nodes'][n]:
                for m_j in alignments['nodes'][m]:
                    if (n == m and n_i != m_j) or n != m:
                        if limited:
                            if is_node(n_i) and is_node(m_j):
                                node_pairs.append((n[0], n_i[0], m[0], m_j[0]))
                        else:
                            edges_aligned_to_n = [d for d in alignments['nodes'][n] if not is_node(d)]
                            edges_aligned_to_m = [d for d in alignments['nodes'][m] if not is_node(d)]
                            if (edges_aligned_to_n and n_i in edges_aligned_to_n) or not edges_aligned_to_n:
                                if (edges_aligned_to_m and m_j in edges_aligned_to_m) or not edges_aligned_to_m:
                                    node_pairs.append((n[0], n_i[0], m[0], m_j[0]))
    return node_pairs

#######################
# PATH ALIGNMENT
#######################


def get_inferred_alignment_points(easy_alignments):
    """
    Finds alignments between paths of different lengths in the two graphs.
    Given two pairs of aligned nodes such that in one graph there is a direct link between the nodes,
    but in the other graph the path is longer, finds all such longer paths and alignes them to the direct path
    from the first graph
    :param easy_alignments: dictionary returned by get_easy_alignment_points
    :return: updates the input dictionary, no return value
    """
    for s_id in easy_alignments.keys():
        s = easy_alignments[s_id]
        alignments = s['alignments']
        # look at DG graph node pairs and find AMR counterparts
        inferred_alignment_points_helper(alignments, s['dep_graph'], s['amr_graph'], s['lemmas'], s['amr_concepts'], 'DG')
        # look at AMR graph node pairs and fine DG counterparts
        inferred_alignment_points_helper(alignments, s['amr_graph'], s['dep_graph'], s['amr_concepts'], s['lemmas'], 'AMR')
        enforce_conjunction_constraint(alignments, s['amr_graph'], s['dep_graph'], s['amr_concepts'])
    overlaps = overlapping_structures(easy_alignments, verbose=False)
    cleaned_alignments = remove_overlaps(easy_alignments, overlaps)
    return cleaned_alignments
    # return easy_alignments


def inferred_alignment_points_helper(alignments, s_graph, t_graph, s_lexicon, t_lexicon, source_type):
    """
    Given two graphs, one of which is the source and the other target, and the alignments between their nodes
    finds alignments between paths such that on the source side the path includes two nodes and a link between them,
    and on the target side two nodes aligned to the source ones connected by a path of arbitrary length.
    :param alignments: previously found node and edge alignments
    :param source_graph
    :param target_graph
    :param source_type: "AMR" or "DG"
    :return:
    """
    node_pairs = create_node_pairs(alignments, limited=False)
    for amr1, dep1, amr2, dep2 in node_pairs:
        t_high = None
        t_low = None
        s_path = ()
        if source_type == 'DG' and (dep1, dep2) in s_graph:
            s_path = (dep1, s_graph[(dep1, dep2)], dep2)
            t_high = amr1
            t_low = amr2
        if source_type == 'AMR' and (amr1, amr2) in s_graph:
            s_path = (amr1, s_graph[(amr1, amr2)], amr2)
            t_high = dep1 if isinstance(dep1,int) else dep1[0]
            t_low = dep2 if isinstance(dep2,int) else dep2[0]
        if t_high and t_low:
            raw_paths = find_shortest_path_through_parents(t_high, t_low, t_graph)
            raw_normalized = [tuple(normalize_edges(parse_path(r_p, [], []), t_graph)) if
                              len(r_p) > 1 else tuple(r_p) for r_p in raw_paths]
            for t_p in raw_normalized:
                ### NE CHANGES HERE ##
                if source_type == 'DG':
                    t_p = check_name(t_p, t_graph, t_lexicon)
                    if not is_node(t_p):
                        t_p = extend_to_entity_node(t_p, t_graph, t_lexicon, alignments['nodes'])
                else:
                    s_path = check_name(s_path, s_graph, s_lexicon)
                    s_path = extend_to_entity_node(s_path, s_graph, s_lexicon, alignments['nodes'])
                new_s_path, t_path = extend_path((s_path,), t_p, s_graph, alignments, source_type)
                if new_s_path and t_path:
                    if source_type == 'DG':
                        newer_s_path = check_case(new_s_path, s_graph)
                        if isinstance(dep1, tuple) and dep1 not in newer_s_path:
                            newer_s_path.append(dep1)
                        if isinstance(dep2, tuple) and dep2 not in newer_s_path:
                            newer_s_path.append(dep2)
                        if is_node(t_path):
                            insert_to_dict(alignments['nodes'], t_path, tuple(newer_s_path))
                        else:
                            insert_to_dict(alignments['paths'], t_path, tuple(newer_s_path))
                    else:
                        new_t_path = check_case(t_path, t_graph)
                        if isinstance(dep1, tuple) and dep1 not in new_t_path:
                            if isinstance(new_t_path, list):
                                new_t_path.append(dep1)
                            else:
                                new_t_path = dep1
                        if isinstance(dep2, tuple) and dep2 not in new_t_path:
                            if isinstance(new_t_path, list):
                                new_t_path.append(dep2)
                            else:
                                new_t_path = dep2
                        insert_to_dict(alignments['paths'], new_s_path, tuple(new_t_path))


def extend_path(source_p, target_p, source_g, alignments, source_type):
    aligned_n = alignments['nodes']
    target_n = set(extract_nodes(target_p))
    source_n = set(extract_nodes(source_p))
    if source_type == 'DG':
        aligned_to_target_n = [aligned_n[(t,)] if (t,) in aligned_n else set([]) for t in target_n]
    else:
        aligned_to_target_n = [[amr for amr, dep_list in aligned_n.items() if (t,) in dep_list] for t in target_n]
    aligned_to_target_n = [[x[0] for x in sublist] for sublist in aligned_to_target_n if sublist]

    extra = set([])
    acceptable_leftovers_lists = []
    for a_list in aligned_to_target_n:
        if not any([a_node in source_n for a_node in a_list]):
            extra.update(set(a_list))
            if len(a_list) > 1:
                acceptable_leftovers_lists.append(a_list)
    if not extra:
        # but that shouldn't happen
        return source_p, target_p
    else:
            temp_source_p = copy.deepcopy(list(source_p))
            new_source, leftovers = extend_source(temp_source_p, source_g, list(extra))
            disqualified = False
            for lo in leftovers:
                if not any([lo in sublist for sublist in acceptable_leftovers_lists]):
                    disqualified = True
            for l_list in acceptable_leftovers_lists:
                if all([l in leftovers for l in l_list]):
                    disqualified = True
            if not disqualified:
                return new_source, target_p
            else:
                return None, None


def extend_source(source_p, source_g, extra):
    if len(extra) == 1:
        source_n = extract_nodes(source_p)
        for n in source_n:
            if (extra[0], n) in source_g:
                source_p.append((extra[0], source_g[(extra[0],n)], n))
                return tuple(source_p), []
            elif (n, extra[0]) in source_g:
                source_p.append((n, source_g[n, (extra[0])], extra[0]))
                return tuple(source_p), []
        return tuple(source_p), extra
    else:
        source_n = extract_nodes(source_p)
        extends = {}
        for e in extra:
            for n in source_n:
                if (e,n) in source_g or (n,e) in source_g:
                    extends[e] = n
        if extends:
            e, n = extends.items()[0]
            if (e,n) in source_g:
                source_p.append((e, source_g[(e,n)], n))
            else:
                source_p.append((n, source_g[(n,e)], e))
            return extend_source(source_p, source_g, [x for x in extra if x != e])
        else:
            return tuple(source_p), extra


def check_case(d_path, d_graph):
    if is_node(d_path):
        return d_path
    new_edges = []
    path = list(d_path)
    nmod_edges = [e for e in d_path if e[1].startswith(":nmod-")]
    for edge in nmod_edges:
        tail = edge[2]
        new_edges = [(e1, lab, e2) for (e1,e2), lab in d_graph.items() if e1 == tail and lab in [":case", ":mark"]]
    for edge in path:
        tail = edge[2]
        new_edges.extend([(e1, lab, e2) for (e1,e2), lab in d_graph.items() if e1 == tail and lab == ":mwe"])
    if new_edges:
        path.extend(new_edges)
    return path


def check_name(amr_path, amr_graph, amr_concepts):
    path = list(amr_path)
    nodes = extract_nodes(amr_path)
    name_nodes = [n for n in nodes if amr_concepts[n] == 'name']
    for n in name_nodes:
        children = get_children(n, amr_graph)
        for c in children:
            if c not in nodes:
                path.append((n, amr_graph[(n, c)], c))
    return tuple(path)


def extend_to_entity_node(amr_path, amr_graph, amr_concepts, node_alignments):
    path = list(amr_path)
    top = get_top_nodes(path, amr_graph)
    all_parents = [get_parents(n, amr_graph) for n in top]
    entity_parents = [[x for x in p if amr_concepts[x] in ontology and not node_alignments[(x,)]]
                      for p in all_parents]
    # changed = False
    for t, parents in zip(top, entity_parents):
        for p in parents:
                path.append((p, amr_graph[(p, t)], t))
                # changed = True
    # if changed:
    #     return extend_to_entity_node(tuple(path), amr_graph, amr_concepts, node_alignments)
    # else:
    return tuple(path)


def enforce_conjunction_constraint(alignments, amr_graph, dep_graph, amr_concepts):
    amr_paths = copy.deepcopy(alignments['paths'])
    for amr_path in amr_paths:
        conjunction_expansion(amr_path, amr_graph, dep_graph, amr_concepts, alignments)


def conjunction_expansion(amr_path, amr_graph, dep_graph, amr_concepts, alignments):
    node_alignments = alignments['nodes']
    path_aligments = alignments['paths']
    for amr_path, dep_paths in path_aligments.items():
        changed_amr = False
        path = list(amr_path)
        nodes = extract_nodes(amr_path)
        conjunction_nodes = [n for n in nodes if amr_concepts[n] in ['and', 'or']]
        for n in conjunction_nodes:
            children = [x for x in get_children(n, amr_graph) if x not in nodes and amr_graph[(n, x)].startswith(":op")]
            if children:
                changed_amr = True
            for c in children:
                # missing child is lexically aligned
                # add the corresponding dep structure to dep paths aligned to amr path
                if (c, ) in node_alignments:
                    path.append((n, amr_graph[(n, c)], c))
                    aligned_to_c = node_alignments[(c, )]
                    new_dep_paths = copy.deepcopy(dep_paths)
                    if aligned_to_c:
                        new_dep_paths = expand_dep_structures(aligned_to_c, dep_paths, dep_graph)
                    path_aligments[tuple(path)] = new_dep_paths
                # missing child is a subgraph
                else:
                    candidate_children = paths_containing_node_as_root((c, ), amr_graph, path_aligments)
                    candidate_dep_children = [path_aligments[p] for p in candidate_children]
                    for dep_path in dep_paths:
                        good_children = pick_optimal_child(amr_path, dep_path, candidate_children, candidate_dep_children,
                                                           amr_graph, dep_graph, c)
                        for amr_child, dep_children in good_children:
                            expand_with_children(amr_path, dep_path, amr_child, dep_children, amr_graph, dep_graph,
                                                 c, alignments)
        if changed_amr:
            path_aligments.pop(amr_path)


def expand_with_children(amr_path, dep_path, amr_extra, dep_extras, amr_graph, dep_graph, amr_child_root, alignments):
    full_amr_path = combine_paths(amr_path, amr_extra, amr_child_root, amr_graph)
    for dep_extra in dep_extras:
        full_dep_path = combine_paths(dep_path, dep_extra, None, amr_graph)
        alignments['paths'][full_amr_path].add(full_dep_path)


def combine_paths(path1, path2, nominated_node2, graph):
    path = list(path1)
    nodes1 = extract_nodes(path1)
    if nominated_node2:
        for n in nodes1:
            if (n, nominated_node2) in graph:
                path.append((n, graph[(n, nominated_node2)], nominated_node2))
                path.extend(list(path2))
                return tuple(path)
    else:
        nodes2 = extract_nodes(path2)
        for n1 in nodes1:
            for n2 in nodes2:
                if (n1, n2) in graph:
                    path.append((n1, graph[(n1, n2)], n2))
                    path.extend(list(path2))
                    return tuple(path)
                elif (n2, n1) in graph:
                    path.append((n2, graph[(n2, n1)], n1))
                    path.extend(list(path2))
                    return tuple(path)


def pick_optimal_child(amr_path, dep_path, candidate_amrs, candidate_deps, amr_graph, dep_graph, amr_child_root):
    good = []
    for c_amr, c_deps in zip(candidate_amrs, candidate_deps):
        if can_add(c_amr, amr_path, amr_graph, amr_child_root):
            good_deps = []
            for c_dep in c_deps:
                if can_add(c_dep, dep_path, dep_graph):
                    good_deps.append(c_dep)
            good.append((c_amr, good_deps))
    return good


def can_add(extra, structure, graph, extra_root=None):
    nodes = extract_nodes(structure)
    if is_node(extra):
        return any([(n, extra[0]) in graph or (extra[0], n) in graph for n in nodes])
    else:
        return are_paths_linkable(structure, extra, extra_root, graph)


def are_paths_linkable(path1, path2, nominated_node2, graph):
    nodes1 = extract_nodes(path1)
    nodes2 = extract_nodes(path2)
    if nominated_node2:
        for n1 in nodes1:
            if (n1, nominated_node2) in graph or (nominated_node2, n1) in graph:
                return True
    else:
        for n1 in nodes1:
            for n2 in nodes2:
                if (n1, n2) in graph or (n2, n1) in graph:
                    return True
    return False


def paths_containing_node_as_root(node, amr_graph, path_alignments):
    paths = []
    for path in path_alignments:
        if node in get_top_nodes(path, amr_graph):
            paths.append(path)
    return paths


def is_node_in_path_alignments(node, path_algnments):
    for path in path_algnments:
        if node in extract_nodes(path):
            return True
    return False


def expand_dep_structures(extras, structures, dep_graph):
    all_expanded_structures = []
    for extra in extras:
        expanded_structures = []
        for structure in structures:
            s = list(structure)
            nodes = extract_nodes(s)
            if is_node(extra):
                attached = False
                for n in nodes:
                    parents = get_parents(n, dep_graph)
                    children = get_children(n, dep_graph)
                    if extra[0] in parents:
                        s.append((extra[0], dep_graph[(extra[0], n)], n))
                        attached = True
                    if extra[0] in children:
                        s.append((n, dep_graph[(n, extra[0])], extra[0]))
                        attached = True
                if attached:
                    expanded_structures.append(tuple(s))
            else:
                extra_nodes = extract_nodes(extra)
                if any([n in extra_nodes for n in nodes]):
                    s.extend(list(extra))
                    expanded_structures.append(tuple(s))
        all_expanded_structures.extend(expanded_structures)
    return set(all_expanded_structures)


def overlapping_structures(manual_alignments, verbose=True):
    overlaps = {}
    for s_id in manual_alignments:
        alignments = manual_alignments[s_id]['alignments']
        node_dict = manual_alignments[s_id]['amr_concepts']
        lemmas = manual_alignments[s_id]['lemmas']
        overlap_list = overlaps_in_sentence(alignments, node_dict, lemmas, verbose)
        overlaps[s_id] = overlap_list
    if verbose:
        for s_id in overlaps:
            if overlaps[s_id]:
                print str(s_id)
                for o in overlaps[s_id]:
                    print o[0]
                    print o[1]
                    print '\n'
                    print '\n'
    return overlaps


def amr_overlaps_in_sentence(p_alignments, node_dict, verbose):
    amr_structures = p_alignments.keys()
    return overlaps_in_sentence_helper(amr_structures, node_dict, verbose)


def dep_overlaps_in_sentence(n_alignments, p_alignments, lemmas, verbose):
    dep_structures = [x for x in flatten_list(n_alignments.values()) if not is_node(x)]
    dep_structures.extend([x for x in flatten_list(p_alignments.values()) if not is_node(x)])
    return overlaps_in_sentence_helper(dep_structures, lemmas, verbose)


def overlaps_in_sentence_helper(structures, label_dict, verbose):
    overlap_list = []
    for s1 in structures:
        for s2 in structures:
            if s1 != s2:
                if overlap(s1, s2):
                    common_part = overlap_part(s1, s2)
                    if not any([equal_paths(common_part, s) for s in structures]):
                        if verbose:
                            overlap_list.append((show_as_string(s1, label_dict),
                                                show_as_string(s2, label_dict)))
                        else:
                            overlap_list.append((s1, s2))
    return overlap_list


def overlaps_in_sentence(alignments, node_dict, lemmas, verbose):
    amr_overlaps = amr_overlaps_in_sentence(alignments['paths'], node_dict, verbose)
    dep_overlaps = dep_overlaps_in_sentence(alignments['nodes'], alignments['paths'], lemmas, verbose)
    if verbose:
        overlap_list = amr_overlaps
        overlap_list.extend(dep_overlaps)
    else:
        overlap_list = {'amr': amr_overlaps, 'dep': dep_overlaps}
    return overlap_list


def overlap_part(structure1, structure2):
    return [edge for edge in structure1 if edge in structure2]


def overlap(structure1, structure2):
    if contained_within(structure1, structure2) or contained_within(structure2, structure1):
        return False
    else:
        if is_node(structure1) or is_node(structure2):
            return False
        else:
            return any([(edge in structure2) for edge in structure1])


def contained_within(structure1, structure2):
    return structure1 == structure2 or properly_contained_within(structure1, structure2)


def properly_contained_within(structure1, structure2):
    # is structure1 contained within structure2?
    if is_node(structure2):
        return False
    if is_node(structure1) and not is_node(structure2):
        return structure1[0] in extract_nodes(structure2)
    if len(structure1) >= len(structure2):
        return False
    else:
        return all([(edge in structure2) for edge in structure1])


def remove_overlaps(alignments, overlaps):
    for s_id in alignments:
        n_align = alignments[s_id]['alignments']['nodes']
        p_align = alignments[s_id]['alignments']['paths']
        for amr_overlap in overlaps[s_id]['amr']:
            for amr in amr_overlap:
                if amr in p_align:
                    del p_align[amr]
        for dep_overlap in overlaps[s_id]['dep']:
            for dep in dep_overlap:
                amr_sides_n = [(amr, dep_side) for (amr, dep_side) in n_align.items() if dep in dep_side]
                amr_sides_p = [(amr_p, dep_side) for (amr_p, dep_side) in p_align.items() if dep in dep_side]
                for amr_side, dep_side in amr_sides_n:
                    if len(dep_side) == 1:
                        del n_align[amr_side]
                    else:
                        n_align[amr_side] = [d for d in dep_side if d != dep]
                for amr_side, dep_side in amr_sides_p:
                    if len(dep_side) == 1:
                        del p_align[amr_side]
                    else:
                        p_align[amr_side] = [d for d in dep_side if d != dep]
    return alignments


# def remove_amr_overlaps(offending_amrs, p_align, amr_dict):
# 	if not offending_amrs:
# 		return p_align
# 	for o_a in offending_amrs:
# 		del p_align[o_a]
# 		new_overlaps = amr_overlaps_in_sentence(p_align, amr_dict, False)
# 		if not new_overlaps:
# 			return p_align
# 		else:
# 			remaining_amrs = [a for a in offending_amrs if a != o_a]
# 			return remove_amr_overlaps(remaining_amrs, p_align, amr_dict)
#
#
# def remove_dep_overlaps(offending_deps, n_align, p_align, lemmas):
# 	if not offending_deps:
# 		return n_align, p_align
# 	for o_d in offending_deps:
# 		amr_sides_n = [(amr, dep_side) for (amr, dep_side) in n_align.items() if o_d in dep_side]
# 		amr_sides_p = [(amr_p, dep_side) for (amr_p, dep_side) in p_align.items() if o_d in dep_side]
# 		for amr_side, dep_side in amr_sides_n:
# 			if len(dep_side) == 1:
# 				del n_align[amr_side]
# 			else:
# 				n_align[amr_side] = [d for d in dep_side if d != o_d]
# 		for amr_side, dep_side in amr_sides_p:
# 			if len(dep_side) == 1:
# 				del p_align[amr_side]
# 			else:
# 				p_align[amr_side] = [d for d in dep_side if d != o_d]
# 		new_overlaps = dep_overlaps_in_sentence(n_align, p_align, lemmas, False)
# 		if not new_overlaps:
# 			return n_align, p_align
# 		else:
# 			remaining_deps = [d for d in offending_deps if d != o_d]
# 			return remove_dep_overlaps(remaining_deps, n_align, p_align, lemmas)
#
#
# def remove_overlaps(alignments, overlaps):
# 	for s_id in alignments:
# 		n_align = alignments[s_id]['alignments']['nodes']
# 		temp_n_align = copy.deepcopy(n_align)
# 		p_align = alignments[s_id]['alignments']['paths']
# 		temp_p_align = copy.deepcopy(p_align)
# 		amr_dict = alignments[s_id]['amr_concepts']
# 		lemmas = alignments[s_id]['lemmas']
# 		offending_amrs = set(flatten_list([list(pair) for pair in overlaps[s_id]['amr']]))
# 		offending_deps = set(flatten_list([list(pair) for pair in overlaps[s_id]['dep']]))
# 		temp_p_align = remove_amr_overlaps(offending_amrs, temp_p_align, amr_dict)
# 		new_n_align, new_p_align = remove_dep_overlaps(offending_deps, temp_n_align, temp_p_align, lemmas)
# 		alignments[s_id]['alignments']['nodes'] = new_n_align
# 		alignments[s_id]['alignments']['paths'] = new_p_align
# 	return alignments


def find_shortest_path(higher_node, lower_node, graph, ignore_direction=False):
    nodewise_path_dict = populate_path_dict(higher_node, lower_node, graph, 'none')
    meeting_points = check_for_meeting_points(nodewise_path_dict)
    paths = extract_paths(meeting_points, nodewise_path_dict, higher_node, lower_node, ignore_direction)
    min_length = min([len(x) for x in paths])
    shortest = [p for p in paths if len(p) == min_length]
    return shortest


def find_paths_through_parents(higher_node, lower_node, graph, ignore_direction=False):
    nodewise_path_dict = populate_path_dict(higher_node, lower_node, graph, 'only_up')
    meeting_points = check_for_meeting_points(nodewise_path_dict)
    paths = extract_paths(meeting_points, nodewise_path_dict, higher_node, lower_node, ignore_direction)
    return paths


def find_shortest_path_through_parents(higher_node, lower_node, graph, ignore_direction=False):
    nodewise_path_dict = populate_path_dict(higher_node, lower_node, graph, 'only_up')
    meeting_points = check_for_meeting_points(nodewise_path_dict)
    paths = extract_paths(meeting_points, nodewise_path_dict, higher_node, lower_node, ignore_direction)
    if not paths:
        pass
    min_length = min([len(x) for x in paths])
    shortest = [p for p in paths if len(p) == min_length]
    return shortest


def find_paths(higher_node, lower_node, graph, ignore_direction=False):
    """
    Finds all paths between two nodes.
    Treats the graph as if it was not directed to allow for finding monotonic and non-monotonic paths
    Monotonic path proceeds from higher/lower node through its children to lower/higher node (respectively)
    Non-monotonic path proceeds from one node to the other through a mixture of children and parents
    :param higher_node: node in the target graph aligned to the higher node in the source graph
    :param lower_node: node in the target graph aligned to the lower node in the source graph
    :param graph: the target graph
    :return: all paths between higher_node and lower_node
    """
    nodewise_path_dict = populate_path_dict(higher_node, lower_node, graph, 'none')
    meeting_points = check_for_meeting_points(nodewise_path_dict)
    paths = extract_paths(meeting_points, nodewise_path_dict, higher_node, lower_node, ignore_direction)
    return paths


def populate_path_dict(higher_node, lower_node, graph, direction_restriction):
    nodewise_path_dict = collections.defaultdict(lambda: {'from_1_up': set([]), 'from_1_down': set([]),
                                                          'from_2_up': set([]), 'from_2_down': set([])})
    nodewise_path_dict[higher_node]['from_1_up'].add(tuple([higher_node]))
    nodewise_path_dict[lower_node]['from_2_up'].add(tuple([lower_node]))
    percolate(higher_node, [], graph, nodewise_path_dict, 'up', 1)
    percolate(lower_node, [], graph, nodewise_path_dict, 'up', 2)
    if direction_restriction == 'none':
        nodewise_path_dict[higher_node]['from_1_down'].add(tuple([higher_node]))
        nodewise_path_dict[lower_node]['from_2_down'].add(tuple([lower_node]))
        percolate(higher_node, [], graph, nodewise_path_dict, 'down', 1)
        percolate(lower_node, [], graph, nodewise_path_dict, 'down', 2)
    return nodewise_path_dict


def percolate(source, node_buffer, graph, nodewise_path_dict, direction, source_id):
    if direction == 'up':
        next_nodes = get_parents(source, graph)
    else:
        next_nodes = get_children(source, graph)
    next_nodes = [n for n in next_nodes if nodewise_path_dict[n]['from_{}_{}'.format(source_id, direction)] == set([])]
    node_buffer.extend(zip([source for _ in xrange(len(next_nodes))], next_nodes))
    if node_buffer:
        next = node_buffer.pop()
        next_node = next[1]
        predecessor = next[0]
        paths_to_next_node = [list(x) for x in copy.deepcopy(nodewise_path_dict[predecessor]['from_{}_{}'.format(source_id, direction)])]
        extension = graph[(next_node, predecessor)] if direction == 'up' else graph[(predecessor, next_node)]
        for p in paths_to_next_node:
            p.append(extension)
            p.append(next_node)
        nodewise_path_dict[next_node]['from_{}_{}'.format(source_id, direction)].update([tuple(x) for x in paths_to_next_node])
        percolate(next_node, node_buffer, graph, nodewise_path_dict, direction, source_id)


def check_for_meeting_points(nodewise_path_dict):
    return [n for n in nodewise_path_dict.keys() if (nodewise_path_dict[n]['from_1_up'] or nodewise_path_dict[n]['from_1_down'])
            and (nodewise_path_dict[n]['from_2_up'] or nodewise_path_dict[n]['from_2_down'])]


def extract_paths(meeting_points, nodewise_path_dict, higher_node, lower_node, ignore_direction):
    all_valid_paths = set()
    for m in meeting_points:
        from_1_to_m_down = [list(x) for x in nodewise_path_dict[m]['from_1_down']]
        from_1_to_m_up = [list(x) for x in nodewise_path_dict[m]['from_1_up']]
        from_2_to_m_down = [list(x) for x in nodewise_path_dict[m]['from_2_down']]
        from_2_to_m_up = [list(x) for x in nodewise_path_dict[m]['from_2_up']]
        if m == higher_node:
            if from_2_to_m_up:
                paths = from_2_to_m_up
            elif ignore_direction:
                paths = from_2_to_m_down
            else:
                paths = [reverse_edges_in_path(p) for p in from_2_to_m_down]
            for p in paths:
                p.reverse()
        elif m == lower_node:
            if from_1_to_m_up and not ignore_direction:
                paths = [reverse_edges_in_path(p) for p in from_1_to_m_up]
            elif from_1_to_m_up:
                paths = from_1_to_m_up
            else:
                paths = from_1_to_m_down
        else:
            if from_1_to_m_down:
                paths_part_1 = from_1_to_m_down
            elif ignore_direction:
                paths_part_1 = from_1_to_m_up
            else:
                paths_part_1 = [reverse_edges_in_path(p) for p in from_1_to_m_up]
            if from_2_to_m_up:
                paths_part_2 = from_2_to_m_up
            elif ignore_direction:
                paths_part_2 = from_2_to_m_down
            else:
                paths_part_2 = [reverse_edges_in_path(p) for p in from_2_to_m_down]
            for p in paths_part_2:
                p.reverse()
            paths = []
            for p1 in paths_part_1:
                for p2 in paths_part_2:
                    full_path = copy.deepcopy(p1)
                    full_path.extend(p2[1:])
                    paths.append(full_path)
        all_valid_paths.update([tuple(p) for p in paths if check_path_validity(p)])
    return list([list(p) for p in all_valid_paths])


def check_path_validity(path):
    edges = set([])
    for edge in [path[x:x+3] for x in range(0, len(path), 2)]:
        reverse_edge = reverse_edges_in_path(edge)
        reverse_edge.reverse()
        if tuple(edge) in edges or tuple(reverse_edge) in edges:
            return False
        else:
            edges.add(tuple(edge))
            edges.add(tuple(reverse_edge))
    return True


def get_next_parameters(nodewise_path_dict, parents_1, parents_2, meeting_points, direction):
    next_endpoints_1 = [n for sublist in parents_1 for n in sublist if n not in meeting_points]
    next_endpoints_2 = [n for sublist in parents_2 for n in sublist if n not in meeting_points]
    next_paths_1 = [nodewise_path_dict[n]['from_1_{}'.format(direction)] for n in next_endpoints_1]
    next_paths_2 = [nodewise_path_dict[n]['from_2_{}'.format(direction)] for n in next_endpoints_2]
    return next_endpoints_1, next_endpoints_2, next_paths_1, next_paths_2


def normalize_edges(edges, graph):
    normalized_edges = []
    for parent, relation, child in edges:
        if (parent, child) in graph:
            normalized_edges.append((parent, relation, child))
        else:
            normalized_edges.append((child, reverse_edge_label(relation), parent))
    return normalized_edges


def parse_path(path, nodes_and_edges, parent_buffer):
    if not path:
        if not nodes_and_edges and parent_buffer:
            nodes_and_edges.append(parent_buffer[0])
        return nodes_and_edges
    elif is_edge_label(path[0]):
        relation = path[0]
        parent = parent_buffer.pop()
        child = path[1]
        nodes_and_edges.append((parent, relation, child))
        parent_buffer.append(child)
        return parse_path(path[2:], nodes_and_edges, parent_buffer)
    elif path[0] == '(':
        parent_buffer.append(parent_buffer[-1])
        return parse_path(path[1:], nodes_and_edges, parent_buffer)
    elif path[0] == ')':
        parent_buffer.pop()
        return parse_path(path[1:], nodes_and_edges, parent_buffer)
    elif path[0] == '|':
        return parse_path(path[1:], nodes_and_edges, parent_buffer)
    else:
        parent_buffer.append(path[0])
        return parse_path(path[1:], nodes_and_edges, parent_buffer)


def print_alignment(sentences, filename, structure_type='both'):
    out_file = open(filename, "w")
    for s_id in sorted(sentences.keys()):
        sentence_dict = sentences[s_id]
        amr_concepts = sentence_dict['amr_concepts']
        words = sentence_dict['words']
        out_file.write(str(s_id) + ": " + sentence_dict['sentence'] + '\n')
        out_file.write("AMR    :    DEP\n")
        structures = ['nodes', 'paths'] if structure_type == 'both' else [structure_type]
        for s_type in structures:
            alignment_dict = sentence_dict['alignments'][s_type]
            for a in alignment_dict:
                a_string = show_as_edges(a, amr_concepts)
                for b in alignment_dict[a]:
                    b_string = show_as_edges(b, words)
                    out_file.write('; '.join(a_string) + "    #    " + '; '.join(b_string) + "\n")
        out_file.write('\n')


ontology = ["thing", "person", "family", "animal", "language", "nationality", "ethnic-group", "regional-group",
            "religious-group", "political-movement", "organization", "company", "government-organization", "military",
            "criminal-organization", "political-party", "market-sector", "school", "university", "research-institute",
            "team", "league", "location", "city", "city-district", "county", "state", "province", "territory",
            "country", "local-region", "country-region", "world-region", "continent", "ocean", "sea", "lake", "river",
            "gulf", "bay", "strait", "canal", "peninsula", "mountain", "volcano", "valley", "canyon", "island",
            "desert", "forest", "moon", "planet", "star", "constellation", "facility", "airport", "station", "port",
            "tunnel", "bridge", "road", "railway-line", "canal", "building", "theater", "museum", "palace", "hotel",
            "worship-place", "sports-facility", "market", "park", "zoo", "amusement-park", "event", "incident",
            "natural-disaster", "earthquake", "war", "conference", "game", "festival", "product", "vehicle", "ship",
            "aircraft", "aircraft-type", "spaceship", "car-make", "work-of-art", "picture", "music", "show",
            "broadcast-program", "publication", "book", "newspaper", "magazine", "journal", "natural-object", "law",
            "treaty", "award", "food-dish", "music-key", "musical-note", "variable", "molecular-physical-entity",
            "small-molecule", "protein", "protein-family", "protein-segment", "amino-acid", "macro-molecular-complex",
            "enzyme", "nucleic-acid", "pathway", "gene", "dna-sequence", "cell", "cell-line", "organism", "disease",
            "medical-condition"]

reified_relations = ['be-compared-to-91', 'have-concession-91', 'have-condition-91', 'be-destined-for-91',
                     'have-org-role-91', 'have-extent-91', 'have-frequency-91', 'have-instrument-91', 'have-li-91',
                     'be-located-at-91', 'have-manner-91', 'have-mod-91', 'have-name-91', 'have-part-91',
                     'have-polarity-91', 'have-purpose-91', 'have-quant-91', 'have-org-role-91', 'be-from-91',
                     'have-subevent-91', 'include-91', 'include-91', 'be-temporally-at-91']
