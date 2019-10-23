from pycorenlp import StanfordCoreNLP
import re
import networkx as nx
import matplotlib.pyplot as plt
import collections
from utils import *


def read_and_parse(amr_file, max_sent_length=100, isi=False, jamr=False):
    """
    Read in AMRs from a text file and store them as lists of nodes and edges.
    Uses a CoreNLP dependency parser to parse the sentences and stores resulting information as lists of dependency graph
     nodes and edges.
    :param amr_file: text file holding AMR data; no empty lines allowed within a block of text representing an AMR graph
    :param max_sent_length: optional limit on length of sentences that will be processed
    :return: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary of AMR nodes,
    a list of AMR edges, dictionary of dependency nodes and a list of dependency edges
    """
    return process_amr_input_file(amr_file, max_sent_length, parse=True, isi=isi, jamr=jamr)


def read(amr_file, max_sent_length=100, isi=False, jamr=False):
    """
    Read in AMRs from a text file and store them as lists of nodes and edges.
    :param amr_file: text file holding AMR data; no empty lines allowed within a block of text representing an AMR graph
    :param max_sent_length: optional limit on length of sentences that will be processed
    :return: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary of AMR nodes,
     and a list of AMR edges
    """
    return process_amr_input_file(amr_file, max_sent_length, parse=False, isi=isi, jamr=jamr)


def process_amr_input_file(amr_file, max_sent_length, parse=False, isi=False, jamr=False):
    """
    Reads in AMRs from a text file and, depending on parameter values, may also parse the source sentences to extract
    dependency graphs.
    :param amr_file: text file holding AMR data; no empty lines allowed within a block of text representing an AMR graph
    :param max_sent_length: optional limit on length of sentences that will be processed
    :param parse: specifies whether parsing should be performed
    :return: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary of AMR nodes,
    a list of AMR edges, and optionally also dictionary of dependency nodes and a list of dependency edges
    """
    sent_id = 0
    amr_id = ""
    processed = {}
    sentence = ""
    full_amr = []
    ignore = False
    for line in open(amr_file, "r"):
        split_line = line.split()
        if split_line:
            if split_line[1] == "::id":
                amr_id = " ".join(split_line[1:3])
            elif split_line[1] == "::snt":
                if len(split_line[2:]) <= max_sent_length:
                    ignore = False
                    sentence = ' '.join(split_line[2:])
                    sent_id += 1
                else:
                    ignore = True
            if not ignore and (split_line[0][0] == "(" or split_line[0][0] == ":"):
                full_amr.extend(process_amr_line(line))
                # full_amr.extend(line.replace('(', ' ( ').replace(')', ' ) ').split())
        elif not ignore:
            # Empty line signals the end of an AMR
            amr_concepts, amr2jamr, amr2isi, amr_relations = process_amr(full_amr, jamr=jamr, isi=isi)
            if parse:
                lemmas, words, dependencies = dep_parse(sentence)
                processed[sent_id] = {'sentence': sentence, 'amr_id': amr_id, 'amr_graph': amr_relations,
                                      'amr_concepts': amr_concepts, 'amr2isi': amr2isi, 'amr2jamr': amr2jamr,
                                      'dep_graph': dependencies, 'lemmas': lemmas, 'words': words}
            else:
                processed[sent_id] = {'sentence': sentence, 'amr_id': amr_id, 'amr_graph': amr_relations,
                                      'amr_concepts': amr_concepts, 'amr2isi': amr2isi, 'amr2jamr': amr2jamr}
            amr_id = ""
            sentence = ""
            full_amr = []
    return processed


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


def dep_parse(sentence):
    """
    Parse a sentence using CoreNLP dependency parser and extract lemmatization and dependencies in the extradependencies
    mode. The function depends on CoreNLP server being set up.
    See http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
    :param sentence: sentence to be parsed
    :return: a dictionary whose keys are word indeces and values are lemmas;
    a dictionary whose keys are (parent, child) tuples and values are edge labels
    """
    nlp = StanfordCoreNLP('http://localhost:9000')
    annotation = nlp.annotate((sentence), properties={'annotators': 'tokenize,ssplit,lemma,pos,depparse',
                                                      'outputFormat': 'json',
                                                      'depparse.extradependencies': 'MAXIMAL'})
    lemmas = {}
    words = {}
    for t in annotation['sentences'][0]['tokens']:
        lemmas[t['index']] = t['lemma']
        words[t['index']] = t['word']
    lemmas[0] = 'ROOT'
    words[0] = 'ROOT'
    dependencies = rename_dependencies(annotation['sentences'][0]['collapsed-ccprocessed-dependencies'])
    dep_edge_dict = collections.defaultdict(str)
    for dep in dependencies:
            dep_edge_dict[(dep['governor'], dep['dependent'])] = dep['dep']
    return lemmas, words, dep_edge_dict


def rename_dependencies(dependencies):
    """
    For the benefit of the graph drawing tool, rename dependency labels so that instead of ':' they use '-', e.g.
    'nmod:to' changes to 'nmod-to'
    :param dependencies: a list of dependency edges
    :return: the same list with edge labels modified
    """
    for dep in dependencies:
        relation = dep['dep']
        if ':' in relation:
            new_relation = ''.join(['-' if char == ':' else char for char in relation])
            dep['dep'] = ':' + new_relation
        else:
            dep['dep'] = ':' + relation
    return dependencies


def process_amr(amr_string, jamr=False, isi=False):
    """
    Given an AMR represented as a string, extract all nodes and edges between them.
    :param amr_string: AMR represented as a string
    :return: a dictionary whose keys are node ids and values are node concepts; a list of edges
    """
    amr_string = [token.lstrip('"').rstrip('"') for token in amr_string]
    concept_dict = extract_all_concepts(amr_string)
    # concept_dict_combined = concept_dict.copy()
    concept_dict_jamr = {}
    concept_dict_isi = {}
    if jamr and isi:
        concept_dict_combined, graph = parse_amr(amr_string, concept_dict.copy(), [], {}, [], 0, jamr_counter={}, jamr=True)
        isi_concept_dict_combined, _ = parse_amr(amr_string, concept_dict.copy(), [], {}, [], 0, jamr_counter={}, isi=True)
        concept_dict = {k: v[0] for k, v in concept_dict_combined.items()}
        concept_dict_jamr = {k: v[1][0] for k, v in concept_dict_combined.items()}
        concept_dict_isi = {k: v[1] for k, v in isi_concept_dict_combined.items()}
    if jamr:
        concept_dict_combined, graph = parse_amr(amr_string, concept_dict, [], {}, [], 0, jamr_counter={}, jamr=True)
        concept_dict = {k: v[0] for k, v in concept_dict_combined.items()}
        concept_dict_jamr = {k: v[1][0] for k, v in concept_dict_combined.items()}
    elif isi:
        concept_dict_combined, graph = parse_amr(amr_string, concept_dict, [], {}, [], 0, jamr_counter={}, isi=True)
        concept_dict = {k: v[0] for k, v in concept_dict_combined.items()}
        concept_dict_isi = {k: v[1] for k, v in concept_dict_combined.items()}
    else:
        concept_dict, graph = parse_amr(amr_string, concept_dict, [], {}, [], 0)
    concept_dict['top'] = 'TOP'
    graph[('top', amr_string[1])] = ':focus'
    return concept_dict, concept_dict_jamr, concept_dict_isi, graph


def extract_all_concepts(full_amr):
    """
    Given an AMR represented as a string, extract all nodes.
    Nodes which are instantiations of a concept have an unique ids. Nodes which are constants are assigned a mock id
    of the form noneX where X is an integer, starting from 0 and going up to how many constant-nodes there are in
    the AMR.
    :param full_amr: AMR represented as a string
    :return: a dictionary whose keys are node ids and values are node concepts
    """
    concept_dict = {}
    for i in range(0, len(full_amr)):
        if full_amr[i - 1] == '(' and full_amr[i + 1] == '/':
            concept_name = full_amr[i + 2]
            if re.match('-[0-9]+', concept_name[-3:]):
                concept_name = concept_name[:-3]
            concept_dict[full_amr[i]] = concept_name
    return concept_dict


def parse_amr(tokens, concept_dict, relations, current_relation, open_concepts, none_counter, jamr_counter = {},
              jamr=False, isi=False, nesting_level=0):
    """
    Given an AMR represented as a string, extract all edges between nodes.
    :param tokens: as yet unparsed portion of the AMR represented as a string
    :param concept_dict: a dictionary whose keys are node ids and values are node concepts
    :param relations: list of already extracted edges
    :param current_relation: the edge that is currently being extracted (i.e. some, but not all, of its elements have
    been seen)
    :param open_concepts: nodes that have been seen in the already processed portion of the AMR and may have children
    in the yet unprocessed portion
    :param none_counter: the numeric part of next noneX id that can be assigned to a constant-node when encountered
    :return:  dictionary whose keys are node ids and values are node concepts;
     dictionary whose keys are (parent, child) tuples and values are edge labels
     if jamr_ids=True, every node is represented as a tuple of ids, the standard one ant the jamr one
    """
    base_level = '1' if isi else '0'
    if not tokens:
        amr_edge_dict = {}
        for edge in relations:
            amr_edge_dict[(edge['parent'], edge['child'])] = edge['relation']
        return concept_dict, amr_edge_dict
    else:
        first = tokens[0]
        if first == '(':
            if jamr or isi:
                if nesting_level == 0:
                    concept_dict[tokens[1]] = (concept_dict[tokens[1]], [base_level])
                    jamr_counter[0] = base_level
            open_concepts.append(tokens[1])
            return parse_amr(tokens[4:], concept_dict, relations, {'relation': '', 'parent': tokens[1], 'child': ''},
                             open_concepts, none_counter, jamr_counter, jamr, isi, nesting_level)
        if first == ')':
            open_concepts.pop()
            if jamr or isi:
                cleared_counter = {level: v for level,v in jamr_counter.items() if level <= nesting_level}
                nesting_level -= 1
            else:
                cleared_counter = jamr_counter
            latest_concept = open_concepts[len(open_concepts) - 1] if open_concepts else ''
            return parse_amr(tokens[1:], concept_dict, relations, {'relation': '', 'parent': latest_concept, 'child': ''},
                             open_concepts, none_counter, cleared_counter, jamr, isi, nesting_level)
        if first.startswith(':'):
            if jamr or isi:
                nesting_level += 1
                if nesting_level in jamr_counter:
                    last_on_level = jamr_counter[nesting_level]
                    jamr_id = last_on_level[:-1] + str(int(last_on_level[-1]) + 1)
                else:
                    jamr_id = jamr_counter[nesting_level - 1] + '.' + base_level
            current_relation['relation'] = first
            if tokens[1] == '(':
                current_relation['child'] = tokens[2]
                if jamr or isi:
                    if isinstance(concept_dict[tokens[2]], tuple):
                        concept_dict[tokens[2]][1].append(jamr_id)
                    else:
                        concept_dict[tokens[2]] = (concept_dict[tokens[2]], [jamr_id])
                    jamr_counter[nesting_level] = jamr_id
                relations.append(current_relation)
                return parse_amr(tokens[1:], concept_dict, relations, {}, open_concepts, none_counter, jamr_counter,
                                 jamr, isi, nesting_level)
            else:
                if tokens[1] in concept_dict:
                    concept_id = tokens[1]
                    if isi:
                        if isinstance(concept_dict[concept_id], tuple):
                            concept_dict[concept_id][1].append(jamr_id)
                        else:
                            concept_dict[concept_id] = (concept_dict[concept_id], [jamr_id])
                        jamr_counter[nesting_level] = jamr_id
                else:
                    concept_id = 'none' + str(none_counter)
                    none_counter += 1
                    concept_dict[concept_id] = tokens[1]
                    if jamr or isi:
                        concept_dict[concept_id] = (tokens[1], [jamr_id])
                        jamr_counter[nesting_level] = jamr_id
                if jamr or isi:
                    nesting_level -= 1
                current_relation['child'] = concept_id
                relations.append(current_relation)
                latest_concept = open_concepts[len(open_concepts) - 1]
                return parse_amr(tokens[2:], concept_dict, relations, {'relation': '', 'parent': latest_concept, 'child': ''},
                                 open_concepts, none_counter, jamr_counter, jamr, isi, nesting_level)


def draw_amr(sent_dict, output_dir, save_graphs=False):
    """
    Draw AMR graphs and save them to individual png files.
    :param sent_dict: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary
    of AMR nodes, a list of AMR edges, and optionally other information
    :param output_dir: directory in which to save the drawings
    :param save_graphs: whether to pickle networx graphs created for the purpose of drawing
    :return: none
    """
    draw_graphs(sent_dict, output_dir, save_graphs, True, False)


def draw_dep(sent_dict, output_dir, save_graphs=False):
    """
    Draw dependency graphs and save them to individual png files.
    :param sent_dict: a dictionary whose keys are integers and values are dictionaries holding the sentence, a dictionary
    of dependency nodes, a list of dependency edges, and optionally other information
    :param output_dir: directory in which to save the drawings
    :param save_graphs: whether to pickle networx graphs created for the purpose of drawing
    :return: none
    """
    draw_graphs(sent_dict, output_dir, save_graphs, False, True)


def draw_amr_and_dep(sent_dict, output_dir, save_graphs=False):
    """
    Draw both AMR and dependency graphs and save them to individual png files.
    :param sent_dict: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary
    of AMR nodes, a list of AMR edges, dictionary of dependency nodes and a list of dependency edges
    :param output_dir: directory in which to save the drawings
    :param save_graphs: whether to pickle networx graphs created for the purpose of drawing
    :return: none
    """
    draw_graphs(sent_dict, output_dir, save_graphs)


def draw_graphs(sent_dict, output_dir, save_graphs=False, want_amr=True, want_dep=True, special_subgraphs=None):
    """
    Draw graphs and save them to individual png files.
    Uses pyplot (through networx library) to draw the graphs and grapviz to find optimal layout.
    If required, pickles individual graphs in the networx format.
    :param sent_dict:
    :param output_dir:
    :param save_graphs:
    :param want_amr:
    :param want_dep:
    :return:
    """
    plt.figure(figsize=(35, 20))
    for s in sent_dict:
        # if s not in [14]:
        #     continue
        sentence = sent_dict[s]['sentence']
        title = sentence
        word_list = sentence.split(' ')
        if len(word_list) > 15:
            title = '\n'.join([' '.join(segment) for segment in [word_list[i:i+15] for i in xrange(0, len(word_list), 15)]])
        if want_dep:
            dependencies = sent_dict[s]['dep_graph']
            word_dict = sent_dict[s]['words']
            lemmas = [str(l) + '/ ' + word_dict[l] for l in word_dict.keys()]
            dep_graph = nx.MultiDiGraph()
            dep_graph.add_nodes_from(lemmas)
            dep_edge_labels = {}
            for parent, child in dependencies:
                gov = str(parent) + '/ ' + word_dict[parent]
                dep = str(child) + '/ ' + word_dict[child]
                relation = dependencies[(parent, child)]
                dep_graph.add_edge(gov, dep, link=relation)
                dep_edge_labels[(gov, dep)] = relation
            dep_pos = nx.nx_pydot.graphviz_layout(dep_graph, prog='dot')
            nodes = nx.draw_networkx_nodes(dep_graph, dep_pos, node_size=1600, alpha=1, node_color='white')
            nodes.set_edgecolor('white')
            if special_subgraphs:
                general_colour = 'grey'
                colour = 'deeppink'
                edge_colours = [colour if e in special_subgraphs[s][3] else 'black' for e in dep_graph.edges()]
                special_edge_labels = {tup: rel for (tup, rel) in dep_edge_labels.items() if tup in special_subgraphs[s][3]}
                other_edge_labels = {tup: rel for (tup, rel) in dep_edge_labels.items() if tup not in special_subgraphs[s][3]}
                special_nodes = {n: n for n in dep_graph.nodes() if n in special_subgraphs[s][2]}
                other_nodes = {n: n for n in dep_graph.nodes() if n not in special_subgraphs[s][2]}
            else:
                general_colour = 'blue'
                edge_colours = 'blue'
                colour = edge_colours
                special_edge_labels = {}
                other_edge_labels = dep_edge_labels
                special_nodes = {}
                other_nodes = {n: n for n in dep_graph.nodes()}
            nx.draw_networkx_edges(dep_graph, dep_pos, width=2, alpha=0.5, edge_color=edge_colours, arrows=False)
            nx.draw_networkx_edge_labels(dep_graph, dep_pos, special_edge_labels, font_color=colour, label_pos=0.3, font_size=12)
            nx.draw_networkx_edge_labels(dep_graph, dep_pos, other_edge_labels, font_color=general_colour, label_pos=0.3, font_size=12)
            nx.draw_networkx_labels(dep_graph, dep_pos, special_nodes, font_size=15, font_family='sans-serif', font_weight='bold', font_color=colour)
            nx.draw_networkx_labels(dep_graph, dep_pos, other_nodes, font_size=15, font_family='sans-serif', font_weight='bold', font_color=general_colour)

            dep_file_name = output_dir + "/" + str(s) + "dep.png"#"dep.svg"
            plt.axis('off')
            plt.title("***UD*** "+title, y=.95)
            plt.savefig(dep_file_name, bbox_inches='tight', pad_inches=0)#, format='svg')
            plt.clf()
            if save_graphs:
                nx.write_gpickle(dep_graph, "./pickles/" + str(s) + "_dep.gpickle")

        if want_amr:
            relations = sent_dict[s]['amr_graph']
            amr_concept_dict = sent_dict[s]['amr_concepts']
            amr_concepts = [d + "/ " + amr_concept_dict[d] for d in amr_concept_dict.keys()]
            amr_graph = nx.MultiDiGraph()
            amr_graph.add_nodes_from(amr_concepts)
            amr_edge_labels = {}
            for parent_id, child_id in relations:
                parent = parent_id + "/ " + amr_concept_dict[parent_id]
                child = child_id + "/ " + amr_concept_dict[child_id]
                relation = relations[(parent_id, child_id)]
                amr_graph.add_edge(parent, child, link=relation)
                amr_edge_labels[(parent, child)] = relation
            amr_pos = nx.nx_pydot.graphviz_layout(amr_graph, prog='dot')
            nodes = nx.draw_networkx_nodes(amr_graph, amr_pos, node_size=1600, alpha=1, node_color='white', edgecolor='white')
            nodes.set_edgecolor('white')
            if special_subgraphs:
                general_colour = 'grey'
                colour = 'c'
                edge_colours = [colour if e in special_subgraphs[s][1] else 'black' for e in amr_graph.edges()]
                special_edge_labels = {tup: rel for (tup, rel) in amr_edge_labels.items() if tup in special_subgraphs[s][1]}
                other_edge_labels = {tup: rel for (tup, rel) in amr_edge_labels.items() if tup not in special_subgraphs[s][1]}
                special_nodes = {n: n for n in amr_graph.nodes() if n in special_subgraphs[s][0]}
                other_nodes = {n: n for n in amr_graph.nodes() if n not in special_subgraphs[s][0]}
            else:
                general_colour = 'black'
                edge_colours = 'black'
                colour = edge_colours
                special_edge_labels = {}
                other_edge_labels = amr_edge_labels
                special_nodes = {}
                other_nodes = {n: n for n in amr_graph.nodes()}
            nx.draw_networkx_edges(amr_graph, amr_pos, width=2, alpha=0.5, edge_color=edge_colours, arrows=False)
            nx.draw_networkx_edge_labels(amr_graph, amr_pos, special_edge_labels, font_color=colour, label_pos=0.3, font_size=12)
            nx.draw_networkx_edge_labels(amr_graph, amr_pos, other_edge_labels, font_color=general_colour, label_pos=0.3, font_size=12)
            nx.draw_networkx_labels(amr_graph, amr_pos, special_nodes, font_size=15, font_family='sans-serif', font_weight='bold', font_color=colour)
            nx.draw_networkx_labels(amr_graph, amr_pos, other_nodes, font_size=15, font_family='sans-serif', font_weight='bold', font_color=general_colour)

            # nx.draw_networkx_edges(amr_graph, amr_pos, width=2, alpha=0.5, edge_color='red', arrows=False)
            # nx.draw_networkx_labels(amr_graph, amr_pos, font_size=15, font_family='sans-serif', font_weight='bold')
            # nx.draw_networkx_edge_labels(amr_graph, amr_pos, amr_edge_labels, font_color='red', label_pos=0.3, font_size=12)

            amr_file_name = output_dir + "/" + str(s) + "amr_auto.png"#"amr.svg"
            plt.axis('off')
            plt.title("***automatic AMR*** "+title, y=.95)
            plt.savefig(amr_file_name, bbox_inches='tight', pad_inches=0)#, format='svg')
            plt.clf()
            if save_graphs:
                nx.write_gpickle(amr_graph, "./pickles/" + str(s) + "_amr.gpickle")


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
                           '\t_\t_\t_\t' + str(parent) + '\t' + clean_edge_label(dependencies[(parent, child)]) + '\t_\t_\n')
        out_file.write('\n')
    out_file.close()


def read_conllu(dep_file):
    """
    Read in dependency graphs from a text file in the CONLL-U format and store them as lists of nodes and edges
    :param dep_file: CONLL-U format text file storing dependency trees
    :return: a dictionary whose keys are integers and values are dictionaries holding the sentence, dictionary of
    dependency nodes and a list of dependency edges
    """
    parses = {}
    sent_id = 1
    lemmas = {}
    words = {}
    dependencies = {}
    sentence = []
    for line in open(dep_file, "r"):
        split_line = line.split()
        if split_line:
            word_index = split_line[0]
            word = split_line[1]
            lemma = split_line[2]
            parent_index = split_line[6]
            edge_label = ':' + ''.join(['-' if char == ':' else char for char in split_line[7]])
            lemmas[int(word_index)] = lemma
            words[int(word_index)] = word
            # if parent_index == 'nmod-against':
            #     print split_line
            dependencies[(int(parent_index), int(word_index))] = edge_label
            sentence.append(word)
        else:
            lemmas[0] = 'ROOT'
            words[0] = 'ROOT'
            parses[sent_id] = {'sentence': ' '.join(sentence), 'dep_graph': dependencies,
                               'lemmas': lemmas, 'words': words}
            sent_id += 1
            lemmas = {}
            words = {}
            dependencies = {}
            sentence = []
    return parses


def reorder_conllu_file(parse_file, out_file):
    sent_dict = collections.defaultdict(list)#{}
    with open(out_file, "w") as out_f, open(parse_file, "r") as in_f:
        for line in in_f:
            l = line.split('\t')
            if len(l) > 1:
                sent_dict[int(l[0])].append(line)
            else:
                for key in sorted(sent_dict):
                    for out_line in sent_dict[key]:
                        out_f.write(out_line)
                out_f.write('\n')
                sent_dict = collections.defaultdict(list)


def read_in_fixed_parses(sent_dict, filename):
    fixed_parses = read_conllu(filename)
    for s_id in sent_dict:
        sent_dict[s_id]['dep_graph'] = fixed_parses[s_id]['dep_graph']
        sent_dict[s_id]['lemmas'] = fixed_parses[s_id]['lemmas']
        sent_dict[s_id]['words'] = fixed_parses[s_id]['words']
    return sent_dict


def count_corenlp_mistakes(amr_file, fixed_parses, automatic_parse_dict=None, manual_parse_dict=None, verbose=False,
                           norm_dep=False):
    if automatic_parse_dict and manual_parse_dict:
        sent_dict = automatic_parse_dict
        fixed_dict = manual_parse_dict
    else:
        sent_dict = read_and_parse(amr_file)
        fixed_dict = read_conllu(fixed_parses)
    mistake_dict = {}
    for s_id in sent_dict:
        corenlp = sent_dict[s_id]['dep_graph']
        fixed = fixed_dict[s_id]['dep_graph']
        right = []
        label_change = []
        deleted_edges = []
        reversed_edges = []
        added_edges = []
        # right = 0
        # label_change = 0
        # deleted_edges = 0
        # reversed_edges = 0
        # added_edges = 0
        for parent, child in corenlp:
            corenlp_dep = corenlp[(parent, child)]
            if (parent, child) in fixed:
                fixed_dep = normalize_dep_label(fixed[(parent, child)], total=True) if norm_dep else fixed[(parent, child)]
                if corenlp_dep == fixed_dep:
                    right.append((parent, fixed_dep, child))
                    # right += 1
                else:
                    label_change.append((parent, fixed_dep, child))
                    # label_change += 1
            else:
                deleted_edges.append((parent, corenlp_dep, child))
                # deleted_edges += 1
                if (child, parent) in fixed and (child, parent) not in corenlp:
                    reversed_edges.append((parent, fixed[(child, parent)], child))
                    # reversed_edges += 1
        for parent, child in fixed:
            if (parent, child) not in corenlp:
                added_edges.append((parent, fixed[(parent, child)], child))
                # added_edges += 1
        if verbose:
            mistake_dict[s_id] = {'right': len(right),
                                  'right_v': right,
                                  'label_change': len(label_change),
                                  'label_change_v': label_change,
                                  'deleted': len(deleted_edges),
                                  'deleted_v': deleted_edges,
                                  'added': len(added_edges),
                                  'added_v': added_edges,
                                  'reversed': len(reversed_edges),
                                  'reversed_v': reversed_edges,
                                  'total_fixed': len(fixed),
                                  'total_corenlp': len(corenlp)}
        else:
            mistake_dict[s_id] = {'right': len(right),
                                  'label_change': len(label_change),
                                  'deleted': len(deleted_edges),
                                  'added': len(added_edges),
                                  'reversed': len(reversed_edges),
                                  'total_fixed': len(fixed),
                                  'total_corenlp': len(corenlp)}
    return mistake_dict


def normalize_dep_label(dep, total=False):
    if not total and dep in [':compound-prt', ':acl-relcl', ':nmod-tmod']:
        return dep
    else:
        return dep.split('-')[0]


def compare_corenlp_to_fixed_parses(amr_file, fixed_parses, automatic_parse_dict=None, manual_parse_dict=None):
    if automatic_parse_dict and manual_parse_dict:
        mistakes = count_corenlp_mistakes(amr_file, fixed_parses, automatic_parse_dict, manual_parse_dict, verbose=True)
    else:
        mistakes = count_corenlp_mistakes(amr_file, fixed_parses)
    total_fixed = sum([mistakes[s_id]['total_fixed'] for s_id in mistakes])
    total_corelp = sum([mistakes[s_id]['total_corenlp'] for s_id in mistakes])
    right = sum([mistakes[s_id]['right'] for s_id in mistakes])
    label_change = sum([mistakes[s_id]['label_change'] for s_id in mistakes])
    deleted_edges = sum([mistakes[s_id]['deleted'] for s_id in mistakes])
    reversed_edges = sum([mistakes[s_id]['reversed'] for s_id in mistakes])
    added_edges = sum([mistakes[s_id]['added'] for s_id in mistakes])
    print "total corenlp edge count: " + str(total_corelp)
    print "total fixed edge count: " + str(total_fixed)
    print "corenlp got right: " + str(right)
    print "corenlp got wrong: " + str(deleted_edges)
    print "corenlp didn't get: " + str(added_edges)
    print "out of interest, the fix was edge reversal {} times".format(str(reversed_edges))
    print "\n"
    print "recall: " + str(float(right) / total_fixed)
    print "precision: " + str(float(right) / total_corelp)
    print "proportion of corenlp edges needing label change: " + str(float(label_change) / total_corelp)
    print "proportion of corenlp edges which were deleted: " + str(float(deleted_edges) / total_corelp)
    print "proportion of correct edges which were added: " + str(float(added_edges) / total_fixed)
    print "\n"
    no_mistakes = len([s_id for s_id in mistakes if mistakes[s_id]['deleted'] == 0 and mistakes[s_id]['added'] == 0])
    print "In {} sentences corenlp made no mistakes".format(str(no_mistakes))
    print "\n"
    change_counts = [(mistakes[s_id]['deleted'], mistakes[s_id]['added']) for s_id in mistakes]
    freq_delte = collections.defaultdict(int)
    freq_add = collections.defaultdict(int)
    for delete, add in change_counts:
        freq_delte[delete] += 1
        freq_add[add] += 1
    print "How many deletions were made per sentence?"
    print_dict_ordered_by_value(freq_delte)
    print "\n"
    print "How many additions were made per sentence?"
    print_dict_ordered_by_value(freq_add)


# reorder_conllu_file('../data/aligned_parses_messy.txt', '../data/aligned_parses.txt')
# pickle(read_and_parse('../data/aligned_amrs.txt', isi=True, jamr=True), 'parsed_datasets/auto_parsed.p')
# pickle(read_in_fixed_parses(read_and_parse('../data/aligned_amrs.txt', isi=True, jamr=True), '../data/aligned_parses.txt'),
#        'parsed_datasets/parsed.p')
# draw_dep(unpickle('parsed_datasets/parsed.p'), '../data/graphs/new', save_graphs=False)

# pickle(read_and_parse('../data/jamr_gold_amrs.txt', isi=True, jamr=True), 'parsed_datasets/jamr_test_parsed.p')
# pickle(read_in_fixed_parses(read_and_parse('../data/iaa_amrs.txt', isi=True, jamr=True), '../data/iaa_parses.txt'),
#        'parsed_datasets/iaa_parsed.p')
# pickle(read_in_fixed_parses(read_and_parse('../data/aligned_amrs_for_jamr.txt', isi=True, jamr=True), '../data/aligned_parses_for_jamr.txt'),
#        'parsed_datasets/parsed_for_jamr.p')



# old = read('../data/aligned_amrs.txt')



# new = read('../data/extracted_amrs.txt')
# comparison_dict = {}
#
# def make_comparison_dict():
#     for s_id in old:
#         comparison_dict[old[s_id]['amr_id']] = [old[s_id]]
#     for s_id in new:
#         comparison_dict[new[s_id]['amr_id']].append(new[s_id])
#
# make_comparison_dict()
#
# def detect_amr_change(comp_dict):
#     amrs_with_change = collections.defaultdict(list)
#     for amr_id, graphs in comp_dict.items():
#         r1_graph = graphs[0]['amr_graph']
#         r2_graph = graphs[1]['amr_graph']
#         for (parent1, child1), relation1 in r1_graph.items():
#             if (parent1, child1) not in r2_graph:
#                 amrs_with_change[amr_id].append("missing: {}".format(' '.join([parent1,relation1,child1])))
#                 # break
#             elif r2_graph[(parent1, child1)] != relation1:
#                 amrs_with_change[amr_id].append("changed: {}".format(' '.join([parent1,relation1,child1])))
#                 # break
#         for (parent2, child2), relation2 in r2_graph.items():
#             if (parent2, child2) not in r1_graph:
#                 amrs_with_change[amr_id].append("added: {}".format(' '.join([parent2,relation2,child2])))
#                 # break
#             # elif r1_graph[(parent2, child2)] != relation2:
#             #     amrs_with_change[amr_id].append("changed: {}".format(relation2))
#             #     amrs_with_change.add(amr_id)
#             #     break
#     return amrs_with_change
#
# changes = detect_amr_change(comparison_dict)
# for id in changes:
#     print id
#     for change in changes[id]:
#         print("\t"+change)

