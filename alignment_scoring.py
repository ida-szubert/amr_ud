from utils import *
import copy
from alignment_analysis import read_in_hand_alignments, read_in_jamr_alignments,\
    read_in_isi_alignments, read_in_isi_gold_alignments, read_in_isi_alignments_bare, read_in_isi_alignments_part
from alignment import read_neg_polarity_items
from alignment import get_full_alignment, get_full_alignment_oracle, get_node_alignment
import collections


def compare_alignments_per_sentence(automatic_alignment, manual_alignment, structure='both', verbose=False):
    per_sentence = {}
    info_source = manual_alignment if 'sentence' in manual_alignment[1] else automatic_alignment
    for s_id in manual_alignment:
        if s_id == 101:
            pass
        if 'alignments' not in manual_alignment[s_id]:
            continue
        gold = manual_alignment[s_id]['alignments']
        automatic = automatic_alignment[s_id]['alignments']
        match_dict = {'matched_nodes': [], 'unmatched_nodes': [], 'overgenerated_nodes': [],
                          'matched_paths': [], 'unmatched_paths': [], 'overgenerated_paths': []}
        counts = score_sentence(gold, automatic, match_dict)
        recall_n, precision_n, f_n = node_recall_precision(counts)
        recall_p, precision_p, f_p = path_recall_precision(counts)
        if structure == 'both':
            recall, precision, f = total_recall_precision(counts)
        elif structure == 'nodes':
            recall, precision, f = recall_n, precision_n, f_n
        else:
            recall, precision, f = recall_p, precision_p, f_p
        per_sentence[s_id] ={'nodes': [recall_n, precision_n, f_n],
                             'paths': [recall_p, precision_p, f_p],
                             'both': [recall, precision, f],
                             'counts': counts,
                             'sentence': info_source[s_id]['sentence'],
                             'amr_concepts': info_source[s_id]['amr_concepts'],
                             'lemmas': info_source[s_id]['lemmas']}
        if verbose:
            per_sentence[s_id]['detailed'] = match_dict
    return per_sentence


def score_sentence(gold, automatic, match_dict):
    analyse_alignment_dict(gold['nodes'], automatic['nodes'], 'nodes', match_dict)
    analyse_alignment_dict(gold['paths'], automatic['paths'], 'paths', match_dict)
    counts = {k: len(v) for k, v in match_dict.items()}
    return counts


def analyse_alignment_dict(m_dict, a_dict, type, result_dict):
    for amr_m, dep_list_m in m_dict.items():
        amr_a = equivalent_element(a_dict, amr_m)
        dep_list_a = a_dict[amr_a] if amr_a else set([])
        for dep_m in dep_list_m:
            if not amr_a:
                action = 'unmatched'
            else:
                if contains_element(dep_list_a, dep_m):
                    action = 'matched'
                else:
                    action = 'unmatched'
            if type == 'nodes' and is_node(dep_m):
                result_dict['{}_nodes'.format(action)].append([amr_m, dep_m])
            else:
                result_dict['{}_paths'.format(action)].append([amr_m, dep_m])

        for dep_a in dep_list_a:
            try:
                contains_element(dep_list_m, dep_a)
            except IndexError:
                pass
            if not contains_element(dep_list_m, dep_a):
                if type == 'nodes' and is_node(dep_a):
                    result_dict['overgenerated_nodes'].append([amr_a, dep_a])
                else:
                    result_dict['overgenerated_paths'].append([amr_a, dep_a])
    for amr_a in a_dict:
        if not contains_element(m_dict, amr_a):
            for dep_a in a_dict[amr_a]:
                result_dict['overgenerated_{}'.format(type)].append([amr_a, dep_a])


def node_recall_precision(counts):
    return calculate_recall_precision(counts['matched_nodes'], counts['unmatched_nodes'], counts['overgenerated_nodes'])


def path_recall_precision(counts):
    return calculate_recall_precision(counts['matched_paths'], counts['unmatched_paths'], counts['overgenerated_paths'])


def total_recall_precision(counts):
    return calculate_recall_precision(counts['matched_nodes'] + counts['matched_paths'],
                                      counts['unmatched_nodes'] + counts['unmatched_paths'],
                                      counts['overgenerated_nodes'] + counts['overgenerated_paths'])


def calculate_recall_precision(matched, unmatched, overgenerated):
    recall = float(matched) / (matched + unmatched) if matched + unmatched != 0 else 0
    precision = float(matched) / (matched + overgenerated) if matched + overgenerated != 0 else 0
    f_score = 2 * (float(precision * recall) / (precision + recall)) if precision + recall != 0 else 0
    return recall, precision, f_score


def corpus_scores(scored_alignments, structure_type='both'):
    total_matched = 0
    total_unmatched = 0
    total_overgenerated = 0
    for s_id in scored_alignments:
        if scored_alignments[s_id]:
            counts = scored_alignments[s_id]['counts']
            if structure_type == 'both':
                total_matched += counts['matched_nodes'] + counts['matched_paths']
                total_unmatched += counts['unmatched_nodes'] + counts['unmatched_paths']
                total_overgenerated += counts['overgenerated_nodes'] + counts['overgenerated_paths']
            else:
                total_matched += counts['matched_{}'.format(structure_type)]
                total_unmatched += counts['unmatched_{}'.format(structure_type)]
                total_overgenerated += counts['overgenerated_{}'.format(structure_type)]
    return calculate_recall_precision(total_matched, total_unmatched, total_overgenerated)


def see_mistakes(scored_alignments, mistake_type='both', structure_type='both'):
    total_count = 0
    sent_count = 0
    if 'detailed' not in scored_alignments[1]:
        print "Cannot show mistakes: need scored alignment dictionary created with the 'verbose' option."
    else:
        for s_id in scored_alignments:
            if scored_alignments[s_id]:
                mistake_list = make_mistake_list(scored_alignments[s_id]['detailed'], mistake_type, structure_type)
                if mistake_list:
                    sent_count += 1
                    total_count += len(mistake_list)
                    print (str(s_id) + ": " + scored_alignments[s_id]['sentence'])
                    amr_concepts = scored_alignments[s_id]['amr_concepts']
                    lemmas = scored_alignments[s_id]['lemmas']
                    for amr, dep in mistake_list:
                        amr_readable = make_structure_readable(amr, amr_concepts)
                        dep_readable = make_structure_readable(dep, lemmas)
                        if isinstance(amr_readable[0], basestring):
                            print 'AMR: ' + amr_readable[0]
                        else:
                            print 'AMR:  ' + ', '.join([' '.join(part) for part in amr_readable])
                        if isinstance(dep_readable[0], basestring):
                            print 'DG:  ' + dep_readable[0]
                        else:
                            print 'DG:  ' + ', '.join([' '.join(part) for part in dep_readable])
                        print("-"*100)
                    scores = scored_alignments[s_id][structure_type]
                    print("recall: {}, precision: {}, f-score: {}".format(scores[0], scores[1], scores[2]))
                    print('\n\n')
        print 'Total number of mistakes = {}, in {} sentences'.format(str(total_count), str(sent_count))


def make_mistake_list(mistakes, mistake_type='both', structure_type='both'):
    if mistake_type == 'both' and structure_type == 'both':
        mistake_list = copy.deepcopy(mistakes['unmatched_nodes'])
        mistake_list.extend(mistakes['unmatched_paths'])
        mistake_list.extend(mistakes['overgenerated_nodes'])
        mistake_list.extend(mistakes['overgenerated_paths'])
    elif structure_type == 'both':
        mistake_list = copy.deepcopy(mistakes['{}_nodes'.format(mistake_type)])
        mistake_list.extend(mistakes['{}_paths'.format(mistake_type)])
    elif mistake_type == 'both':
        mistake_list = copy.deepcopy(mistakes['unmatched_{}'.format(structure_type)])
        mistake_list.extend(mistakes['overgenerated_{}'.format(structure_type)])
    else:
        mistake_list = copy.deepcopy(mistakes['{}_{}'.format(mistake_type, structure_type)])
    return mistake_list


# Evaluation

def evaluate(automatic, gold, structure, verbose, mistake_type):
    performance = compare_alignments_per_sentence(automatic, gold, structure=structure, verbose=verbose)
    if verbose:
        see_mistakes(performance, mistake_type=mistake_type, structure_type=structure)
    total_score = corpus_scores(performance, structure_type=structure)
    print "f-score: {}, recall: {}, precision: {}".format(total_score[2], total_score[0], total_score[1])


def oracle_evaluate(manual, parses, show_mistakes, mistake_type):
    manual_alignments = read_in_hand_alignments(manual, parses)
    for s_id in manual_alignments:
        manual_alignments[s_id]['alignments']['paths'] = collections.defaultdict(set)
    oracle = get_full_alignment_oracle(manual_alignments)
    evaluate(oracle, read_in_hand_alignments(manual, parses), 'paths', verbose=show_mistakes, mistake_type=mistake_type)


def interannotator(manual_file1, manual_file2, parses, structure, show_mistakes, mistake_type):
    annotator1 = read_in_hand_alignments(manual_file1, parses)
    annotator2 = read_in_hand_alignments(manual_file2, parses)
    evaluate(annotator2, annotator1, structure, verbose=show_mistakes, mistake_type=mistake_type)


def interannotator_nodes(manual_file1, manual_file2, parses, show_mistakes, mistake_type):
    interannotator(manual_file1, manual_file2, parses, "nodes", show_mistakes, mistake_type)


def interannotator_paths(manual_file1, manual_file2, parses, show_mistakes, mistake_type):
    interannotator(manual_file1, manual_file2, parses, "paths", show_mistakes, mistake_type)


def interannotator_both(manual_file1, manual_file2, parses, show_mistakes, mistake_type):
    interannotator(manual_file1, manual_file2, parses, "both", show_mistakes, mistake_type)


def my_on_my(manual_file, parses, structure, show_mistakes, mistake_type):
    manual = read_in_hand_alignments(manual_file, parses)
    neg_dict = read_neg_polarity_items('neg-polarity.txt')
    automatic = get_full_alignment(unpickle(parses), neg_dict)
    # automatic = get_node_alignment(unpickle(parses), neg_dict)
    evaluate(automatic, manual, structure, verbose=show_mistakes, mistake_type=mistake_type)


def my_on_my_nodes(manual_file, parses, show_mistakes, mistake_type):
    my_on_my(manual_file, parses, "nodes", show_mistakes, mistake_type)


def my_on_my_paths(manual_file, parses, show_mistakes, mistake_type):
    my_on_my(manual_file, parses, "paths", show_mistakes, mistake_type)


def my_on_my_both(manual_file, parses, show_mistakes, mistake_type):
    my_on_my(manual_file, parses, "both", show_mistakes, mistake_type)


def my_on_isi(isi_gold_file, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_isi_gold_alignments(isi_gold_file, parses)
    neg_dict = read_neg_polarity_items('neg-polarity.txt')
    automatic = get_node_alignment(unpickle(parses), neg_dict)
    evaluate(automatic, gold, 'nodes', show_mistakes, mistake_type)


def my_on_jamr(jamr_gold_file, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_jamr_alignments(jamr_gold_file)
    neg_dict = read_neg_polarity_items('neg-polarity.txt')
    automatic = get_node_alignment(unpickle(parses), neg_dict)
    evaluate(automatic, gold, 'nodes', show_mistakes, mistake_type)


def isi_on_isi(isi_gold_file, isi_alignments, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_isi_gold_alignments(isi_gold_file, parses)
    automatic = read_in_isi_alignments(isi_alignments, parses)
    # automatic = read_in_isi_alignments_part(isi_alignments, parses)
    evaluate(automatic, gold, 'nodes', show_mistakes, mistake_type)


def isi_on_isi_bare(isi_gold_file, isi_alignments, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_isi_alignments_bare(isi_gold_file, parses)
    automatic = read_in_isi_alignments_bare(isi_alignments, parses)
    # automatic_full = read_in_isi_alignments_bare(isi_alignments, parses)
    # automatic = {k - 100: v for k, v in automatic_full.items() if 99 < k < 201}
    matched = 0
    unmatched = 0
    overgenerated = 0
    for s_id in gold:
        gold_align = gold[s_id]['align']
        auto_align = automatic[s_id]['align']
        for amr, dg in gold_align:
            if (amr, dg) in auto_align:
                matched += 1
            else:
                unmatched += 1
        for amr, dg in auto_align:
            if (amr, dg) not in gold_align:
                overgenerated += 1
    recall, precision, f = calculate_recall_precision(matched, unmatched, overgenerated)
    print "recall: {}, precision: {}, f-score: {}".format(recall, precision, f)


def isi_on_my(isi_alignments, manual_file, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_hand_alignments(manual_file, parses)
    isi = read_in_isi_alignments(isi_alignments, parses)
    evaluate(isi, gold, 'nodes', show_mistakes, mistake_type)


def isi_on_jamr(isi_alignments, jamr_gold_file, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_jamr_alignments(jamr_gold_file)
    # isi = read_in_isi_alignments(isi_alignments, parses)
    # neg_dict = read_neg_polarity_items('neg-polarity.txt')
    automatic = read_in_isi_alignments(isi_alignments, parses)
    evaluate(automatic, gold, 'nodes', show_mistakes, mistake_type)


def isi_as_oracle(isi_alignments, manual_file, parses, structure='both', mistake_type='both', show_mistakes=False):
    gold = read_in_hand_alignments(manual_file, parses)
    isi = read_in_isi_alignments(isi_alignments, parses)
    sent_dict = unpickle(parses)
    for s_id in sent_dict:
        sent_dict[s_id]['alignments'] = {'nodes': isi[s_id]['alignments']['nodes'],
                                         'paths': collections.defaultdict(set)}
    oracle = get_full_alignment_oracle(sent_dict)
    evaluate(oracle, gold, 'paths', show_mistakes, mistake_type)


def jamr_on_my(jamr_alignments, manual_file, parses, mistake_type='both', show_mistakes=False):
    jamr = read_in_jamr_alignments(jamr_alignments)
    gold = read_in_hand_alignments(manual_file, parses)
    evaluate(jamr, gold, 'nodes', show_mistakes, mistake_type)


def jamr_as_oracle(jamr_alignments, manual_file, parses, mistake_type='both', show_mistakes=False):
    gold = read_in_hand_alignments(manual_file, parses)
    jamr = read_in_jamr_alignments(jamr_alignments)
    sent_dict = unpickle(parses)
    for i in range(196, 201):
        sent_dict[i-1] = sent_dict[i]
    del sent_dict[200]
    for s_id in sent_dict:
        sent_dict[s_id]['alignments'] = {'nodes': jamr[s_id]['alignments']['nodes'],
                                         'paths': collections.defaultdict(set)}
    oracle = get_full_alignment_oracle(sent_dict)
    evaluate(oracle, gold, 'paths', show_mistakes, mistake_type)


def jamr_on_isi(jamr_alignments, isi_gold_file, parses, show_mistakes=False, mistake_type='both'):
    gold = read_in_isi_gold_alignments(isi_gold_file, parses)
    jamr = read_in_jamr_alignments(jamr_alignments)
    evaluate(jamr, gold, 'nodes', show_mistakes, mistake_type)


def jamr_on_jamr(jamr_alignments, jamr_gold_file, show_mistakes=False, mistake_type='both'):
    gold = read_in_jamr_alignments(jamr_gold_file)
    jamr = read_in_jamr_alignments(jamr_alignments)
    evaluate(jamr, gold, 'nodes', show_mistakes, mistake_type)

