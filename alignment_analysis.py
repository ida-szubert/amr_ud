# from alignment import *
from alignment import normalize_edges, parse_path, properly_contained_within, contained_within
from utils import *
from parse import process_amr
from collections import defaultdict
import collections
import copy


# Read in my alignments
def read_in_hand_alignments(filename, dict_file, direction='AMR-DG'):
    parsed_sentences = unpickle(dict_file)
    n = 1
    alignments = {'nodes': collections.defaultdict(set), 'paths': collections.defaultdict(set)}
    for line in open(filename, 'r'):
        if line.strip():
            split_line = line.split('    #    ')
            if line.startswith('###'):
                pass
            elif len(split_line) == 2:
                amr_list = split_line[0].split()
                dep_list = split_line[1].split()
                amr_side = [contract_node_label(x) if not is_edge_label(x) else x for x in amr_list]
                amr_path = (amr_side[0],) if len(amr_side) == 1 else \
                    tuple(normalize_edges(parse_path(amr_side, [], []), parsed_sentences[n]['amr_graph']))
                dep_side = tuple([contract_node_label(x) if not (is_edge_label(x) or x in ['(', ')', '|'])
                                  else x for x in dep_list])
                # try:
                dep_path = (dep_side[0],) if len(dep_side) == 1 else \
                    tuple(normalize_edges(parse_path(dep_side, [], []), parsed_sentences[n]['dep_graph']))
                # except TypeError:
                #     pass
                if direction == 'AMR-DG':
                    if is_node(amr_path):
                        alignments['nodes'][amr_path].add(dep_path)
                    else:
                        alignments['paths'][amr_path].add(dep_path)
                else:
                    if is_node(dep_path):
                        alignments['nodes'][dep_path].add(amr_path)
                    else:
                        alignments['paths'][dep_path].add(amr_path)
            # else:
            #     print(parsed_sentences[n]['sentence'] + '\n'+ line + '\n')
        else:
            parsed_sentences[n]['alignments'] = alignments
            n += 1
            alignments = {'nodes': collections.defaultdict(set), 'paths': collections.defaultdict(set)}
    return parsed_sentences


# Read in ISI alignments
def read_in_isi_alignments_bare(file_name, parses):
    alignments = parse_isi(file_name)
    return alignments


def read_in_isi_alignments_part(file_name, parses):
    alignments_full = parse_isi(file_name)
    alignments = {k - 100: v for k, v in alignments_full.items() if 99 < k < 200}
    return read_in_isi(alignments, parses)


def read_in_isi_alignments(file_name, parses):
    alignments = parse_isi(file_name)
    return read_in_isi(alignments, parses)


def read_in_isi_gold_alignments(file_name, parses):
    alignments = parse_isi(file_name, gold=True)
    return read_in_isi(alignments, parses)


def read_in_isi(gold, parses):
    sent_dict = unpickle(parses)
    for s_id in sent_dict:
        sent_dict[s_id]['alignments'] = {'nodes': defaultdict(set), 'paths': {}}
        amr2isi = sent_dict[s_id]['amr2isi']
        isi2amr = {}
        for amr_name in amr2isi:
            for isi_name in amr2isi[amr_name]:
                isi2amr[isi_name] = amr_name
        gold_align = gold[s_id-1]['align']
        new_gold = sent_dict[s_id]['alignments']['nodes']
        for amr, dep in gold_align:
            try:
                new_gold[(isi2amr[amr],)].add((dep,))
            except KeyError:
                pass
    return sent_dict


def parse_isi(alignments, gold=False):
    with open(alignments, "r") as in_f:
        temp_lines = in_f.readlines()
    align_dict = {}
    lines = [l for l in temp_lines if l != "\n"] if gold else temp_lines
    gid = 0
    for i in xrange(0, len(lines), 2):
        align_strings = []
        sentence = ""
        if gold:
            sen_line = lines[i].strip().replace("# ::id ","")
            align_strings.append(lines[i+1].strip().replace("# ::alignments ","").split(" "))
            sentence = sen_line
        else:
            align_strings = [lines[i].strip().split(" "), lines[i + 1].strip().split(" ")]
        for n in range(len(align_strings)):
            align_dict[gid + n] = {}
            align_tups = []
            if sentence:
                align_dict[gid]["sentence"] = sentence
            for align_entry in align_strings[n]:
                align_pair = align_entry.split("-")
                align_dg = int(align_pair[0].strip()) + 1
                align_amr = align_pair[1].strip()
                if not align_amr.endswith(".r"):
                    align_tups.append((align_amr, align_dg))
            align_dict[gid + n]["align"] = align_tups
        gid = gid + 2 if not sentence else gid + 1
    return align_dict


# Read in JAMR alignments
def read_in_jamr_alignments(file_name):
    sent_dict = {}
    sent_id = 0
    jamr_alignments = collections.defaultdict(set)
    words = []
    tokens = []
    full_amr = []
    for line in open(file_name, "r"):
        split_line = line.split()
        if split_line:
            if split_line[1] == '::snt':
                words = flatten_list([split_word(w) for w in split_line[2:]])
            if split_line[1] == '::tok':
                tokens = split_line[2:]
                sent_id += 1
            if split_line[1] == '::alignments':
                alignments = split_line[2:]
                for item in alignments:
                    if '|' in item:
                        aligned_to = item.split('|')[0]
                        node_ids = item.split('|')[1]
                        if '+' in node_ids:
                            pass
                            # for node_id in node_ids.split('+'):
                            #     jamr_alignments[node_id].add(aligned_to)
                        else:
                            jamr_alignments[node_ids].add(aligned_to)
            # if split_line[1] == '::node':
            #     node_id = split_line[2]
            #     concept = split_line[3]
            #     aligned_to = split_line[4] if len(split_line) > 4 else '-1'
            #     jamr_alignments[node_id] = [concept, aligned_to]
            if split_line[0].startswith('(') or split_line[0].startswith(':'):
                # full_amr.extend(line.replace('(', ' ( ').replace(')', ' ) ').split())
                full_amr.extend(process_amr_line(line))
        else:
            sent_dict[sent_id] = {'alignments': {'nodes': {}, 'paths': {}}}
            tokens2words = get_token2words(1, 1, words, tokens, {})
            if sent_id == 20:
                pass
            concept_dict, concept_dict_jamr, concept_dict_isi, graph = process_amr(full_amr, isi=False, jamr=True)
            renamed_jamr_alignments = {}
            for node_id in jamr_alignments:
                try:
                    other_id = [k for k,v in concept_dict_jamr.items() if v == node_id][0]
                except IndexError:
                    pass
                renamed_jamr_alignments[other_id] = jamr_alignments[node_id]
            normal_alignments = collections.defaultdict(set)
            for jamr_a, entries in renamed_jamr_alignments.items():
                for entry in entries:
                    if entry != '-1':
                        dep_range = [int(x.lstrip('*')) for x in entry.split('-')]
                        for i in range(int(dep_range[0]), int(dep_range[1])):
                            normal_alignments[(jamr_a,)].add((tokens2words[i + 1],))
            sent_dict[sent_id]['alignments']['nodes'] = normal_alignments
            jamr_alignments = collections.defaultdict(set)
            tokens = []
            full_amr = []
    return sent_dict


def process_amr_line(line):
    if '"' in line:
        final_tokens = []
        tokens = line.split()
        for t in tokens:
            if '"' in t:
                endpoints = [n for n in range(len(t)) if t[n] == '"']
                quoted = t[endpoints[0]:endpoints[1]+1]
                final_tokens.append(quoted)
                rest = t[endpoints[1]+1:].split(')')[1:]
                for _ in rest:
                    final_tokens.append(')')
            elif t.startswith('('):
                final_tokens.extend(['(', t[1:]])
            elif t.endswith(')'):
                final_tokens.append(t.split(')')[0])
                for _ in t.split(')')[1:]:
                    final_tokens.append(')')
            else:
                final_tokens.append(t)
    else:
        final_tokens = line.replace('(', ' ( ').replace(')', ' ) ').split()
    return final_tokens


def split_word(w):
    if w not in ['.', ',', ';', ':', '?', '!', '!!', 'i.e.', 'e.g.', 'U.S.', '...', 'D.', 'W.R.']:
        if (w.endswith(':') or w.endswith(',') or w.endswith(';')
            or w.endswith('.') or w.endswith('?') or w.endswith('!')):
            return [w[:-1], w[-1]]
        elif (w.startswith('(') and w.endswith(')')):
            return [w[0], w[1:-1], w[-1]]
        elif w != "'s" and w.endswith("'s"):
            return [w[:-2], w[-2:]]
        else:
            return[w]
    else:
        return [w]


def get_token2words(current_i, current_j, words, tokens, tokens2words):
    if len(words) == len(tokens):
        for i in range(len(words)):
            tokens2words[current_j + i] = current_i + i
        return tokens2words
    else:
        for i in range(len(words)):
            words[i] == tokens[i]
            if words[i] == tokens[i]:
                tokens2words[current_j + i] = current_i + i
            else:
                rest_of_words = words[i+1:]
                rest_of_tokens = tokens[i+1:]
                if len(rest_of_words) == 0 and len(rest_of_tokens) == 1:
                    tokens2words[current_j+i] = current_i+i
                    tokens2words[current_j + i + 1] = current_i+i
                    return tokens2words
                else:
                    next_word = rest_of_words[0]
                    matching_token_index = rest_of_tokens.index(next_word)
                    rest_of_tokens = rest_of_tokens[matching_token_index:]
                    for j in range(0, matching_token_index+1):
                        tokens2words[current_j + i + j] = current_i + i
                    return get_token2words((current_i + i + 1),
                                           (current_j + i + matching_token_index+1),
                                           rest_of_words, rest_of_tokens, tokens2words)


# Finding non-path aligned structures
def get_non_paths(sent_dict, print_out=False):
    non_path_amr = []
    path_amr = []
    amr_sent_list = set([])
    non_path_dep = []
    path_dep = []
    dep_sent_list = set([])
    for s_id in sent_dict:
        alignments_n = sent_dict[s_id]['alignments']['paths']
        alignments_p = sent_dict[s_id]['alignments']['paths']
        for amr_p in alignments_p:
            if not is_path(amr_p):
                non_path_amr.append(amr_p)
                amr_sent_list.add(s_id)
                if print_out:
                    print sent_dict[s_id]['sentence']
                    print amr_p
            else:
                path_amr.append(amr_p)
            for dep_p in alignments_p[amr_p]:
                if not (is_node(dep_p) or is_path(dep_p)):
                    non_path_dep.append(dep_p)
                    dep_sent_list.add(s_id)
                    if print_out:
                        print sent_dict[s_id]['sentence']
                        print dep_p
                else:
                    path_dep.append(dep_p)
        for amr_n in alignments_n:
            for dep_n in alignments_n[amr_n]:
                if not (is_node(dep_n) or is_path(dep_n)):
                    non_path_dep.append(dep_n)
                    dep_sent_list.add(s_id)
                    if print_out:
                        print sent_dict[s_id]['sentence']
                        print dep_n
                else:
                    path_dep.append(dep_n)
    count_non_path_amr = len(non_path_amr)
    count_path_amr = len(path_amr)
    count_non_path_dep = len(non_path_dep)
    count_path_dep = len(path_dep)
    print "Non path amr structures: {}".format(count_non_path_amr)
    print "As proportion of al amr structures: {}".format((float(count_non_path_amr) / (count_non_path_amr+count_path_amr))*100)
    print "Non path dep structures: {}".format(count_non_path_dep)
    print "As proportion of al dep structures: {}".format((float(count_non_path_dep) / (count_non_path_dep+count_path_dep))*100)


# Analysing sized of aligned structure
def structure_sizes(manual):
    freq_dict = {'amr': collections.defaultdict(int),
                 'dep': collections.defaultdict(int)}
    for s_id in manual:
        for amr in manual[s_id]['alignments']['nodes']:
            for dep in manual[s_id]['alignments']['nodes'][amr]:
                freq_dict['amr']['n'] += 1
                if is_node(dep):
                    freq_dict['dep']['n'] += 1
                else:
                    freq_dict['dep'][len(dep)] += 1
        for amr in manual[s_id]['alignments']['paths']:
            for dep in manual[s_id]['alignments']['paths'][amr]:
                freq_dict['amr'][len(amr)] += 1
                if is_node(dep):
                    freq_dict['dep']['n'] += 1
                else:
                    freq_dict['dep'][len(dep)] += 1
    return freq_dict


def alignment_type_frequency(sent_dict, raw=False, amr_filter=None):
    freq_dict = collections.defaultdict(int)
    for s_id in sent_dict:
        alignments_n = sent_dict[s_id]['alignments']['nodes']
        alignments_p = sent_dict[s_id]['alignments']['paths']
        for a_n in alignments_n:
            if structure_of_interest(a_n, sent_dict[s_id]['amr_concepts'], sent_dict[s_id]['amr_graph'],
                                     alignments_n, amr_filter):
                for dep in alignments_n[a_n]:
                    a_type = alignment_type(a_n, dep)
                    freq_dict[a_type] += 1
        for a_p in alignments_p:
            if structure_of_interest(a_p, sent_dict[s_id]['amr_concepts'], sent_dict[s_id]['amr_graph'],
                                     alignments_n, amr_filter):
                for dep in alignments_p[a_p]:
                    if is_node(dep):
                        a_type = alignment_type(a_p, dep)
                        freq_dict[a_type] += 1
                    else:
                        if raw:
                            amr_part_size = 0
                            dep_part_size = 0
                        else:
                            amr_part, dep_part = aligned_substructures(a_p, dep, alignments_p, alignments_n)
                            amr_part_size = len(amr_part) if not is_node(amr_part) else 0
                            dep_part_size = len(dep_part) if not is_node(dep_part) else 0
                        a_type = '{}-{}'.format(len(a_p) - amr_part_size, len(dep) - dep_part_size)
                        freq_dict[a_type] += 1
    # total = sum(freq_dict.values())
    # for a_type in freq_dict:
    #     freq_dict[a_type] = (float(freq_dict[a_type]) / total) * 100
    return freq_dict


def structure_of_interest(structure, amr_concepts, amr_graph, node_alignments, amr_filter):
    if amr_filter:
        return amr_filter(structure, amr_concepts, amr_graph, node_alignments)
    else:
        return True


def all_other_filter(structure, amr_concepts, amr_graph, node_alignments):
    quant_exp = quantity_expression_filter(structure, amr_concepts, amr_graph, node_alignments)
    conj = conjunction_filter(structure, amr_concepts, amr_graph, node_alignments)
    ne = named_entity_filter(structure, amr_concepts, amr_graph, node_alignments)
    de = decomposed_nominal_filter(structure, amr_concepts, amr_graph, node_alignments)
    return not (quant_exp or conj or ne or de)


def quantity_expression_filter(structure, amr_concepts, amr_graph, node_alignments):
    if is_node(structure):
        return False
    nodes = extract_nodes(structure)
    concepts = [amr_concepts[x] for x in nodes]
    parents = [[y for y in get_parents(x, amr_graph) if y in nodes] for x in nodes]
    for n, c, ps in zip(nodes, concepts, parents):
        if c in quantities and not ps:
            if not node_alignments[(n,)]:
                print(show_as_string(structure, amr_concepts))
                return True
    return False


def conjunction_filter(structure, amr_concepts, amr_graph, node_alignments):
    if is_node(structure):
        return False
    else:
        for edge in structure:
            if amr_concepts[edge[0]] == 'and' or amr_concepts[edge[2]] == 'and':
                print(show_as_string(structure, amr_concepts))
                return True
    return False


def named_entity_filter(structure, amr_concepts, amr_graph, node_alignments):
    return entity_filter(structure, amr_concepts, amr_graph, node_alignments, named=True)


def decomposed_nominal_filter(structure, amr_concepts, amr_graph, node_alignments):
    return entity_filter(structure, amr_concepts, amr_graph, node_alignments, named=False)


def entity_filter(structure, amr_concepts, amr_graph, node_alignments, named=True):
    if is_node(structure):
        return False
    nodes = extract_nodes(structure)
    concepts = [amr_concepts[x] for x in nodes]
    parents = [[y for y in get_parents(x, amr_graph) if y in nodes] for x in nodes]
    for n, c, ps in zip(nodes, concepts, parents):
        if c in ontology and not ps:
            if not node_alignments[(n,)]:
                is_named_entity = any([(edge[0] == n and edge[1] == ':name') for edge in structure])
                if named:
                    if is_named_entity:
                        print(show_as_string(structure, amr_concepts))
                        return True
                else:
                    if not is_named_entity:
                        print(show_as_string(structure, amr_concepts))
                        return True
    return False


def aligned_substructures(amr, dep, path_alignments, node_alignments):
    largest_amr = ()
    largest_dep = ()
    for a_p in path_alignments:
        if properly_contained_within(a_p, amr) and len(a_p) > len(largest_amr):
            for d_p in path_alignments[a_p]:
                if properly_contained_within(d_p, dep) and len(d_p) > len(largest_dep):
                    largest_amr = a_p
                    largest_dep = d_p
    if not largest_amr:
        for a_n in node_alignments:
            if properly_contained_within(a_n, amr):
                for d_n in node_alignments[a_n]:
                    if properly_contained_within(d_n, dep) and len(d_n) > len(largest_dep):
                        largest_amr = a_n
                        largest_dep = d_n
    return largest_amr, largest_dep


def alignment_type(amr_side, dep_side):
    if is_node(amr_side) and is_node(dep_side):
        return 'n-n'
    if is_node(amr_side):
        return 'n-{}'.format(len(dep_side))
    if is_node(dep_side):
        return '{}-n'.format(len(amr_side))
    else:
        return '{}-{}'.format(len(amr_side), len(dep_side))


def alignment_neatness(manual_alignments, verbose=False):
    neatness_dict = collections.defaultdict(list)
    for s_id in manual_alignments:
        sent_length = len(manual_alignments[s_id]['lemmas'])
        alignment_types = alignment_type_frequency({s_id: manual_alignments[s_id]}, raw=False)
        highest_alignment = highest_alignment_type(alignment_types.keys())
        neatness_dict[highest_alignment].append((manual_alignments[s_id]['sentence'], sent_length))
    if not verbose:
        count_dict = {}
        for alignment_type in neatness_dict:
            freq = len(neatness_dict[alignment_type])
            avg_len = float(sum([x[1] for x in neatness_dict[alignment_type]]))/freq
            count_dict[alignment_type] = (freq, avg_len)
        neatness_dict = count_dict
    return neatness_dict


def highest_alignment_type(types):
    highest_amr = 'n'
    highest_dep = 'n'
    for t in types:
        amr_side = int(t.split('-')[0]) if t.split('-')[0].isdigit() else t.split('-')[0]
        dep_side = int(t.split('-')[1]) if t.split('-')[1].isdigit() else t.split('-')[1]
        if involves_n(highest_amr, highest_dep):
            if not involves_n(amr_side, dep_side):
                highest_amr = amr_side
                highest_dep = dep_side
            else:
                new_amr, new_dep = compare_n_types(highest_amr, highest_dep, amr_side, dep_side)
                highest_amr = new_amr
                highest_dep = new_dep
        else:
            if not involves_n(amr_side, dep_side):
                new_amr, new_dep = compare_int_types(highest_amr, highest_dep, amr_side, dep_side)
                highest_amr = new_amr
                highest_dep = new_dep
    return '-'.join([str(highest_amr), str(highest_dep)])


def involves_n(amr_side, dep_side):
    return amr_side == 'n' or dep_side == 'n'


def compare_n_types(amr1, dep1, amr2, dep2):
    if amr1 == amr2 and dep1 == dep2:
        return amr1, dep1
    else:
        maybe_non_n1 = [x for x in [amr1, dep1] if x != 'n']
        non_n1 = maybe_non_n1[0] if maybe_non_n1 else None
        maybe_non_n2 = [x for x in [amr2, dep2] if x != 'n']
        non_n2 = maybe_non_n2[0] if maybe_non_n2 else None
        if non_n1 and non_n2:
            if non_n1 > non_n2:
                return amr1, dep1
            else:
                return amr2, dep2
        elif non_n2:
            return amr2, dep2
        elif non_n1:
            return amr1, dep1


def compare_int_types(amr1, dep1, amr2, dep2):
    if (amr1 + dep1) > (amr2 + dep2):
        return amr1, dep1
    else:
        return amr2, dep2


# Checking coverage
def coverage(file_name, dict_file):
    manual_alignments = read_in_hand_alignments(file_name, dict_file)
    unaligned(manual_alignments)
    for s, inf in manual_alignments.items():
        if 'alignments' in inf:
            if inf['unaligned']['amr_edges']:
                print(inf['sentence']+"\t")
                print('; '.join(["  ".join(x) for x in inf['unaligned']['amr_edges']]))
                # print('; '.join([x+"/"+inf['amr_concepts'][x] for x in inf['unaligned']['amr_nodes']]))
                print("\n")
    # coverage_counts = collections.defaultdict(int)
    for side in ['amr', 'dep']:
        coverage_counts = collect_coverage_counts(manual_alignments, side)
        print_dict_for_table(coverage_counts)
        print("\n\n\n")
    #     for s_id in manual_alignments:
    #         nodes = len(manual_alignments[s_id]['unaligned']['amr_nodes'])
    #         edges = len(manual_alignments[s_id]['unaligned']['amr_edges'])
    #         if nodes == 0 and edges == 0:
    #             coverage_counts['full'] += 1
    #         elif nodes == 0:
    #             coverage_counts['not covered {} edges'.format(str(edges))] +=1
    #         else:
    #             coverage_counts['not covered {} nodes, {} edges'.format(str(nodes), str(edges))] +=1
    # print_dict_for_table(coverage_counts)


def collect_coverage_counts(manual_alignments, side):
    coverage_counts = collections.defaultdict(int)
    for s_id in manual_alignments:
        if 'alignments' in manual_alignments[s_id]:
            nodes = len(manual_alignments[s_id]['unaligned']['{}_nodes'.format(side)])
            edges = len(manual_alignments[s_id]['unaligned']['{}_edges'.format(side)])
            if nodes == 0 and edges == 0:
                coverage_counts['full'] += 1
            elif nodes == 0:
                coverage_counts['not covered {} edges'.format(str(edges))] +=1
            else:
                coverage_counts['not covered {} nodes, {} edges'.format(str(nodes), str(edges))] +=1
    return coverage_counts


# Analyse node alignment types
def node_alignment_types(manual_alignments, source='amr'):
    total = 0
    one_to_many = 0
    one_to_one = 0
    one_to_no = 0
    only_in_group = 0
    for s_id in manual_alignments:
        node_alignments = manual_alignments[s_id]['alignments']['nodes']
        path_alignments = manual_alignments[s_id]['alignments']['paths']
        aligned_amr_paths = path_alignments.keys()
        amr_nodes = manual_alignments[s_id]['amr_concepts']
        dep_nodes = manual_alignments[s_id]['lemmas']
        if source == 'amr':
            total += len(amr_nodes)
            for amr_node in amr_nodes:
                if amr_node == 'top':
                    continue
                if (amr_node, ) in node_alignments:
                    aligned_dg = [dg for dg in node_alignments[(amr_node,)] if is_node(dg)]
                    if len(aligned_dg) == 1:
                        one_to_one += 1
                    else:
                        one_to_many += 1
                else:
                    one_to_no += 1
                    if any([path_contains_node(p, amr_node) for p in aligned_amr_paths]):
                        only_in_group += 1
                    else:
                        print(amr_node+"\\"+amr_nodes[amr_node])
        else:
            total += len(dep_nodes)
            aligned_deps = flatten_list([[x for x in align_list if is_node(x)] for align_list in node_alignments.values()])
            for dep_node in dep_nodes:
                if dep_node == 0:
                    continue
                if (dep_node, ) in aligned_deps:
                    if sum(map(lambda d: (dep_node, ) == d, aligned_deps)) == 1:
                        one_to_one += 1
                    else:
                        one_to_many += 1
                else:
                    one_to_no += 1
    print "Total nodes: {}".format(total)
    print "One - to - one: {}, {}%".format(one_to_one, (float(one_to_one)/total)*100)
    print "One - to - many: {}, {}%".format(one_to_many, (float(one_to_many)/total)*100)
    print "One - to - no: {}, {}%".format(one_to_no, (float(one_to_no)/total)*100)
    print "Only in a subgraph: {}, {}%".format(only_in_group, (float(only_in_group)/total)*100)


# Should probability distribution over alignments be uniform?
def proportion_of_order_preserving_alignments(manual_alignments):
    preserves = 0
    not_preserves = 0
    cyclic_examples = 0
    for s_id in manual_alignments:
        pairs = []
        amr_graph = manual_alignments[s_id]['amr_graph']
        dep_graph = manual_alignments[s_id]['dep_graph']
        if contains_cycles(amr_graph) or contains_cycles(dep_graph):
            cyclic_examples += 1
        else:
            node_alignments = manual_alignments[s_id]['alignments']['nodes']
            for amr_node, aligned_dep in node_alignments.items():
                dep_nodes = [n for n in aligned_dep if is_node(n)]
                for dep_node in dep_nodes:
                    pairs.append((amr_node[0], dep_node[0]))
            for amr1, dep1 in pairs:
                for amr2, dep2 in pairs:
                    if amr1 != amr2 and dep1 != dep2:
                        amr_status = dominates(amr1, amr2, amr_graph)
                        dep_status = dominates(dep1, dep2, dep_graph)
                        if amr_status == dep_status:
                            preserves += 1
                        else:
                            not_preserves += 1
    total = preserves + not_preserves
    print "Number of graphs with cycles: {}".format(cyclic_examples)
    print "Aligned node pairs in which ordering is preserved: {}, {}%".format(preserves, (float(preserves)/total)*100)
    print "Aligned node pairs in which ordering is not preserved: {}, {}%".format(not_preserves, (float(not_preserves)/total)*100)


def contains_cycles(graph):
    for (head, tail) in graph:
        try:
            get_all_children(head, graph)
        except RuntimeError:
            return True
    return False


def dominates(node1, node2, graph):
    if (node1, node2) in graph:
        return True
    elif (node2, node1) in graph:
        return False
    else:
        children1 = get_all_children(node1, graph)
        return node2 in children1


def get_all_children(node, graph):
    children = [child for parent, child in graph if parent == node]
    if not children:
        return []
    else:
        offspring = flatten_list([get_all_children(child, graph) for child in children])
        offspring.extend(children)
        return offspring


# Finding unaligned nodes and edges
def unaligned_analysis(manual_alignments):
    unaligned(manual_alignments)
    amr_nodes_proportion, dep_nodes_proportion = proportion_of_unaligned(manual_alignments, 'nodes')
    print "Proportion of unaligned AMR nodes: " + str(amr_nodes_proportion)
    print "Proportion of unaligned dependency nodes: " + str(dep_nodes_proportion)
    amr_edges_proportion, dep_edges_proportion = proportion_of_unaligned(manual_alignments, 'edges')
    print "Proportion of unaligned AMR edges: " + str(amr_edges_proportion)
    print "Proportion of unaligned dependency edges: " + str(dep_edges_proportion)
    frequencies = unaligned_items_frequency(manual_alignments)
    for k in frequencies:
        print k
        print_dict_for_table(frequencies[k])
        print '\n'
    connecting_edges_amr, connecting_edges_dep = edges_to_unaligned_nodes(manual_alignments)
    print "Edges incoming to unaligned AMR nodes: \n"
    print_dict_for_table(connecting_edges_amr)
    print "Edges incoming to unaligned dependency nodes: \n"
    print_dict_for_table(connecting_edges_dep)


def individually_unaligned(manual_alignments):
    for s_id in manual_alignments:
        manual_alignments[s_id]['individually_unaligned'] = []
        node_alignments = manual_alignments[s_id]['alignments']['nodes']
        path_alignments = manual_alignments[s_id]['alignments']['paths']
        aligned_dep = flatten_list([extract_nodes(x) for x in flatten_list(node_alignments.values()) if is_node(x)])
        aligned_dep.extend(flatten_list([extract_nodes(x) for x in flatten_list(path_alignments.values()) if is_node(x)]))
        for dep_node in manual_alignments[s_id]['lemmas']:
            if dep_node not in aligned_dep:
                manual_alignments[s_id]['individually_unaligned'].append(dep_node)
    return manual_alignments


def unaligned(manual_alignments, structure_type='both'):
    for s_id in manual_alignments:
        manual_alignments[s_id]['unaligned'] = {}
    if structure_type in ['nodes', 'both']:
        unaligned_nodes(manual_alignments)
    if structure_type in ['edges', 'both']:
        unaligned_edges(manual_alignments)
    return manual_alignments


def unaligned_nodes(manual_alignments):
    for s_id in manual_alignments:
        if 'alignments' in manual_alignments[s_id]:
            unaligned_amr_nodes = []
            unaligned_dep_nodes = []
            node_alignments = manual_alignments[s_id]['alignments']['nodes']
            path_alignments = manual_alignments[s_id]['alignments']['paths']
            amr_nodes = manual_alignments[s_id]['amr_concepts']
            dep_nodes = manual_alignments[s_id]['lemmas']
            amr_nodes_in_alignments = flatten_list([extract_nodes(p) for p in path_alignments.keys()])
            amr_nodes_in_alignments.extend([n[0] for n in node_alignments.keys()])
            for amr_node in amr_nodes:
                if amr_node not in amr_nodes_in_alignments and amr_node != 'top':
                    unaligned_amr_nodes.append(amr_node)
            dep_nodes_in_alignments = flatten_list([extract_nodes(x) for x in flatten_list(node_alignments.values())])
            dep_nodes_in_alignments.extend(flatten_list([extract_nodes(x) for x in flatten_list(path_alignments.values())]))
            for dep_node in dep_nodes:
                if dep_node not in dep_nodes_in_alignments and dep_node != 0:
                    unaligned_dep_nodes.append(dep_node)
            manual_alignments[s_id]['unaligned']['amr_nodes'] = unaligned_amr_nodes
            manual_alignments[s_id]['unaligned']['dep_nodes'] = unaligned_dep_nodes
    return manual_alignments


def unaligned_edges(manual_alignments):
    for s_id in manual_alignments:
        if 'alignments' in manual_alignments[s_id]:
            amr_graph = manual_alignments[s_id]['amr_graph']
            dep_graph = manual_alignments[s_id]['dep_graph']
            unaligned_amr_edges = []
            unaligned_dep_edges = []
            node_alignments = manual_alignments[s_id]['alignments']['nodes']
            path_alignments = manual_alignments[s_id]['alignments']['paths']
            amr_edges_in_alignments = [edge for path in path_alignments.keys() for edge in path]
            dep_paths_in_alignments = [x for x in flatten_list(node_alignments.values()) if not is_node(x)]
            dep_paths_in_alignments.extend([x for x in flatten_list(path_alignments.values()) if not is_node(x)])
            dep_edges_in_alignments = flatten_list([list(x) for x in dep_paths_in_alignments])
            for amr_parent, amr_child in amr_graph:
                edge_label = amr_graph[(amr_parent, amr_child)]
                if (amr_parent, edge_label, amr_child) not in amr_edges_in_alignments and edge_label != ':focus':
                    unaligned_amr_edges.append((amr_parent, edge_label, amr_child))
            for dep_parent, dep_child in dep_graph:
                edge_label = dep_graph[(dep_parent, dep_child)]
                if (dep_parent, edge_label, dep_child) not in dep_edges_in_alignments and edge_label != ':ROOT':
                    unaligned_dep_edges.append((dep_parent, edge_label, dep_child))
            manual_alignments[s_id]['unaligned']['amr_edges'] = unaligned_amr_edges
            manual_alignments[s_id]['unaligned']['dep_edges'] = unaligned_dep_edges


def edges_to_unaligned_nodes(manual_alignments):
    if 'unaligned' not in manual_alignments[1] or 'amr_nodes' not in manual_alignments[1]['unaligned']:
        unaligned(manual_alignments, 'nodes')
    amr_edges = collections.defaultdict(int)
    dep_edges = collections.defaultdict(int)
    for s_id in manual_alignments:
        s = manual_alignments[s_id]
        amr_graph = s['amr_graph']
        dep_graph = s['dep_graph']
        if 'alignments' in manual_alignments[s_id]:
            for node in s['unaligned']['amr_nodes']:
                for edge_label in [amr_graph[(p, ch)] for (p, ch) in amr_graph if ch == node]:
                    amr_edges[edge_label] += 1
            for node in s['unaligned']['dep_nodes']:
                for edge_label in [dep_graph[(p, ch)] for (p, ch) in dep_graph if ch == node]:
                    dep_edges[edge_label] += 1
    return amr_edges, dep_edges


def proportion_of_unaligned(manual_alignments, structure_type):
    if 'unaligned' not in manual_alignments[1] or 'amr_{}'.format(structure_type) not in manual_alignments[1]['unaligned']:
        unaligned(manual_alignments, structure_type)
    amr_key = 'amr_concepts' if structure_type == 'nodes' else 'amr_graph'
    dep_key = 'lemmas' if structure_type == 'nodes' else 'dep_graph'
    for s_id in manual_alignments:
        s = manual_alignments[s_id]
        # amr_count = len(s[amr_key]) - 1
        # dep_count = len(s[dep_key]) - 1
        unaligned_amr_count = len(s['unaligned']['amr_{}'.format(structure_type)])
        unaligned_dep_count = len(s['unaligned']['dep_{}'.format(structure_type)])
        # dep_proportion = (float(unaligned_dep_count) / dep_count) if dep_count else 0
        # amr_proportion = (float(unaligned_amr_count) / amr_count) if amr_count else 0
        dep_proportion = unaligned_dep_count
        amr_proportion = unaligned_amr_count
        s['unaligned']['proportions_{}'.format(structure_type)] = [amr_proportion, dep_proportion]
    # sum_amr = sum([len(manual_alignments[s_id][amr_key]) - 1 for s_id in manual_alignments])
    # sum_dep = sum([len(manual_alignments[s_id][dep_key]) - 1 for s_id in manual_alignments])
    sum_unaligned_amr = sum([len(v['unaligned']['amr_{}'.format(structure_type)]) for v in manual_alignments.values()])
    sum_unaligned_dep = sum([len(v['unaligned']['dep_{}'.format(structure_type)]) for v in manual_alignments.values()])
    # return (float(sum_unaligned_amr / sum_amr), (float(sum_unaligned_dep) / sum_dep)
    return sum_unaligned_amr, sum_unaligned_dep


def unaligned_items_frequency(manual_alignments):
    if 'unaligned' not in manual_alignments[1]:
        unaligned(manual_alignments)
    frequencies = {}
    for type in ['nodes', 'edges']:
        for type2 in ['amr', 'dep']:
            if type == 'nodes':
                id_to_word_key = 'amr_concepts' if type2 == 'amr' else 'lemmas'
            unaligned_freq = collections.defaultdict(int)
            for s_id in manual_alignments:
                if 'alignments' in manual_alignments[s_id]:
                    for item in manual_alignments[s_id]['unaligned']['{}_{}'.format(type2, type)]:
                        actual_item = manual_alignments[s_id][id_to_word_key][item] if type == 'nodes' else item
                        actual_item = actual_item[1] if isinstance(actual_item, tuple) else actual_item
                        actual_item = ':nmod' if actual_item.startswith(':nmod') else actual_item
                        unaligned_freq[actual_item] += 1
            frequencies['{}_{}'.format(type2, type)] = unaligned_freq
    return frequencies


# Finding superfluous nodes
def extra_nodes(manual_alignments):
    extra_amr_nodes = collections.defaultdict(int)
    extra_dep_nodes = collections.defaultdict(int)
    for s_id in manual_alignments:
        node_alignments = manual_alignments[s_id]['alignments']['nodes']
        path_alignments = manual_alignments[s_id]['alignments']['paths']
        extra_amr_per_sentence = set([])
        extra_dep_per_sentence = set([])
        alignment_pairs = path_alignments.items()
        alignment_pairs.extend(node_alignments.items())
        for amr, dep_list in alignment_pairs:
            for dep in dep_list:
                unexpected_amr = unexpected_nodes(amr, dep, node_alignments, path_alignments, 'amr')
                unexpected_dep = unexpected_nodes(dep, amr, node_alignments, path_alignments, 'dep')
                extra_amr_per_sentence.update([manual_alignments[s_id]['amr_concepts'][a] for a in unexpected_amr])
                extra_dep_per_sentence.update([manual_alignments[s_id]['lemmas'][d] for d in unexpected_dep])
        for a in extra_amr_per_sentence:
            extra_amr_nodes[a] += 1
        for d in extra_dep_per_sentence:
            extra_dep_nodes[d] += 1
    return extra_amr_nodes, extra_dep_nodes


def unexpected_nodes(to_inspect, counterpart, node_alignments, path_alignments, side):
    if is_node(counterpart) and is_node(to_inspect):
        return set([])
    expected_nodes = set([])
    if side == 'dep':
        aligned_to_amr = [x for x in (node_alignments[counterpart] if is_node(counterpart) else path_alignments[counterpart]) if not is_node(x)]
        validating_amrs = [a for a in node_alignments if contained_within(a, counterpart)]
        validating_amrs2 = [a for a in path_alignments if contained_within(a, counterpart)]
        expected_structures = flatten_list([node_alignments[a] for a in validating_amrs])
        expected_structures.extend(flatten_list([path_alignments[a] for a in validating_amrs2]))
        expected_structures = remove_paths(expected_structures, aligned_to_amr)
    else:
        aligned_to_dep = [a for (a, dep_list) in node_alignments.items() if counterpart in dep_list if not is_node(a)]
        aligned_to_dep.extend([a for (a, dep_list) in path_alignments.items() if counterpart in dep_list])
        expected_structures = flatten_list([[a for d in dep_list if contained_within(d, counterpart)] for (a, dep_list) in node_alignments.items()])
        expected_structures.extend(flatten_list([[a for d in dep_list if contained_within(d, counterpart)] for (a, dep_list) in path_alignments.items()]))
        expected_structures = remove_paths(expected_structures, aligned_to_dep)
    if is_node(to_inspect):
        if to_inspect in expected_structures:
            expected_nodes.add(to_inspect[0])
    else:
        for edge in to_inspect:
            if (edge[0], ) in expected_structures:
                expected_nodes.add(edge[0])
            if (edge[2], ) in expected_structures:
                expected_nodes.add(edge[2])
            candidate_structures = select_paths_given_edge(expected_structures, edge)
            if any([properly_contained_within(c, to_inspect) for c in candidate_structures]):
                expected_nodes.add(edge[0])
                expected_nodes.add(edge[2])
    return set(extract_nodes(to_inspect)).difference(expected_nodes)


# Find lexically aligned nodes which do not participate in structural alignments
def find_only_lexically_aligned_nodes(manual_alignments):
    for s_id in manual_alignments:
        identified_nodes_amr = []
        node_alignments = manual_alignments[s_id]['alignments']['nodes']
        structure_alignments = manual_alignments[s_id]['alignments']['paths']
        for (amr_n,) in node_alignments:
            if not any([path_contains_node(s, amr_n) for s in structure_alignments]):
                identified_nodes_amr.append(amr_n)
                print(str(s_id)+":   "+amr_n+"/"+manual_alignments[s_id]['amr_concepts'][amr_n])
        manual_alignments[s_id]['alignments']['only_lexical_amr'] = identified_nodes_amr
        identified_nodes_dep = []
        dep_node_alignments = flatten_list([list(v) for k,v in node_alignments.items()])
        dep_node_alignments.extend(flatten_list([list(v) for k,v in structure_alignments.items()]))
        dep_structure_alignments = copy.deepcopy(dep_node_alignments)
        dep_node_alignments = [x for x in dep_node_alignments if is_node(x)]
        dep_structure_alignments = [x for x in dep_structure_alignments if not is_node(x)]
        for (dep_n,) in dep_node_alignments:
            if not any([path_contains_node(s, dep_n) for s in dep_structure_alignments]):
                identified_nodes_dep.append(dep_n)
                print(str(s_id)+":   "+str(dep_n)+"/"+manual_alignments[s_id]['lemmas'][dep_n])
        manual_alignments[s_id]['alignments']['only_lexical_dep'] = identified_nodes_dep
    print("Number of only lexically aligned AMR nodes:")
    print(sum([len(manual_alignments[s_id]['alignments']['only_lexical_amr']) for s_id in manual_alignments]))
    print("Number of only lexically aligned DG nodes:")
    print(sum([len(manual_alignments[s_id]['alignments']['only_lexical_dep']) for s_id in manual_alignments]))


# Find named entity nodes from outside the ontology
def non_ontology_entities(sent_dict):
    non_ontology = []
    in_ontology = []
    for s_id in sent_dict:
        concepts = sent_dict[s_id]['amr_concepts']
        for (parent, child), relation in sent_dict[s_id]['amr_graph'].items():
            if relation == ':name':
                if concepts[parent] not in ontology:
                    non_ontology.append((s_id, parent + '/' + concepts[parent]))
                else:
                    in_ontology.append((s_id, parent + '/' + concepts[parent]))
    return non_ontology, in_ontology


def non_ontology_entities_freq(sent_dict):
    non_ontology, in_ontology = non_ontology_entities(sent_dict)
    freq_dict = collections.defaultdict(int)
    for entity in non_ontology:
        freq_dict[entity[1].split('/')[1]] += 1
    print_dict_for_table(freq_dict)


def ontology_entities_freq(sent_dict):
    non_ontology, in_ontology = non_ontology_entities(sent_dict)
    freq_dict = {}
    for e in ontology:
        freq_dict[e] = 0
    for entity in in_ontology:
        freq_dict[entity[1].split('/')[1]] += 1
    print_dict_for_table(freq_dict)


def tricky_ne_alignments(sent_dict):
    tricky = []
    for s_id in sent_dict:
        concepts = sent_dict[s_id]['amr_concepts']
        for (parent, child), relation in sent_dict[s_id]['amr_graph'].items():
            if relation == ':name':
                for (parent2, child2) in sent_dict[s_id]['amr_graph']:
                    if parent2 == child:
                        if concepts[parent].lower() == concepts[child2].lower():
                            tricky.append((s_id, parent + '/' + concepts[parent], child2 + '/' + concepts[child2],))
    return tricky


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

quantities = ["monetary-quantity", "distance-quantity", "area-quantity", "volume-quantity", "temporal-quantity",
              "frequency-quantity", "speed-quantity", "acceleration-quantity", "mass-quantity", "force-quantity",
              "pressure-quantity", "energy-quantity", "power-quantity", "voltage-quantity", "charge-quantity",
              "potential-quantity", "resistance-quantity", "inductance-quantity", "magnetic-field-quantity",
              "magnetic-flux-quantity", "radiation-quantity", "concentration-quantity", "temperature-quantity",
              "score-quantity", "fuel-consumption-quantity", "seismic-quantity", "date-entity", "date-interval",
              "percentage-entity", "phone-number-entity", "email-address-entity", "url-entity"]


# Utility functions
def remove_paths(path_list, paths_to_remove):
    already_removed = []
    for p in path_list:
        if p in paths_to_remove and p not in already_removed:
            path_list.remove(p)
            already_removed.append(p)
    return path_list


def select_paths_given_edge(paths, edge):
    possible_paths = []
    for p in paths:
        if edge in p:
            possible_paths.append(p)
    return possible_paths


def extract_aligned_targets(source_nodes, alignments_nodes):
    aligned_nodes = []
    for s in source_nodes:
        for amr, dep_list in alignments_nodes.items():
            if amr[0] == s:
                aligned_nodes.append(set(flatten_list([extract_nodes(x) for x in dep_list])))
    return aligned_nodes


# Analysis display
def see_unaligned(file_name, dict_file):
    manual = read_in_hand_alignments(file_name, dict_file)
    unaligned_items = unaligned_items_frequency(manual)
    for k in unaligned_items:
        print k
        print_dict_for_table(unaligned_items[k])
        print '\n\n'
    amr_edges, dep_edges = edges_to_unaligned_nodes(manual)
    for k in [amr_edges, dep_edges]:
        print "Labels of edges leading to unaligned nodes"
        print_dict_for_table(k)
        print'\n\n'


def see_individually_unaligned_dg_nodes(file_name, dict_file):
    manual = read_in_hand_alignments(file_name, dict_file)
    individually_unaligned(manual)
    freq_dict = collections.defaultdict(int)
    for s_id in manual:
        lemmas = manual[s_id]['lemmas']
        for dep in manual[s_id]['individually_unaligned']:
            freq_dict[lemmas[dep]] += 1
    print_dict_for_table(freq_dict)


def see_unexpected_nodes(file_name, dict_file):
    manual = read_in_hand_alignments(file_name, dict_file)
    extra_amr_nodes, extra_dep_nodes = extra_nodes(manual)
    for k in [extra_amr_nodes, extra_dep_nodes]:
        print_dict_for_table(k)
        print '\n\n'


def see_proportions_of_unaligned(manual):
    # manual = read_in_hand_alignments(file_name, dict_file)
    amr_nodes, dep_nodes = proportion_of_unaligned(manual, 'nodes')
    amr_edges, dep_edges = proportion_of_unaligned(manual, 'edges')
    total_amr_nodes = sum([len(x['amr_concepts']) - 1 for x in manual.values()])
    total_dep_nodes = sum([len(x['lemmas']) - 1 for x in manual.values()])
    total_amr_edges = sum([len(x['amr_graph']) - 1 for x in manual.values()])
    total_dep_edges = sum([len(x['dep_graph']) - 1 for x in manual.values()])
    print "Number of AMR nodes: {}".format(str(total_amr_nodes))
    print "Number of DG nodes: {}".format(str(total_dep_nodes))
    print "Number of AMR edges: {}".format(str(total_amr_edges))
    print "Number of DG edges: {}".format(str(total_dep_edges))
    print "Unaligned AMR nodes : {}, {}%".format(str(amr_nodes), (float(amr_nodes)/total_amr_nodes)*100)
    print "Unaligned DG nodes : {}, {}%".format(str(dep_nodes), (float(dep_nodes)/total_dep_nodes)*100)
    print "Unaligned AMR edges : {}, {}%".format(str(amr_edges), (float(amr_edges)/total_amr_edges)*100)
    print "Unaligned DG edges : {}, {}%".format(str(dep_edges), (float(dep_edges)/total_dep_edges)*100)


def see_neatness_distribution(file_name, dict_file):
    manual = read_in_hand_alignments(file_name, dict_file)
    print_dict_for_table(alignment_neatness(manual))


def see_neatness_examples(max_alignment_type, file_name, dict_file):
    manual = read_in_hand_alignments(file_name, dict_file)
    neat_dict = alignment_neatness(manual, verbose=True)
    for s in neat_dict[max_alignment_type]:
        print s


def see_non_ontology_entities():
    sent_dict = unpickle('parsed_datasets/parsed_aligned.p')
    print "Tricky NE alignments:"
    for x in tricky_ne_alignments(sent_dict):
        print x
    print "\n Non-ontology NE frequency :"
    non_ontology_entities_freq(sent_dict)
    print "\n In-ontology NE frequency :"
    ontology_entities_freq(sent_dict)
