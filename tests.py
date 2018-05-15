from alignment import *
# from inference import alignment_inference, initialize_inference, iterate_inference, update_vocab, run_inference
from alignment_analysis import *
import pickle
from alignment_scoring import *
from parse import *
from utils import unpickle
# from model_training import initialize, find_alignments_to_simplify


def test_morphological_analysis():
    positive = [(['ru', 's', 'ty'], ['ru', 'st']),
             (['hung', 'ry'], ['hung', 'er']),
             (['n', 'ever'], ['ever']),
             (['wear', 'ine', 's', 's'], ['wear', 'y']),
             (['gold'], ['gold', 'en'])]
    negative = [(['form', 'er'], ['hung', 'er'])]
    for p1, p2 in positive:
        if likely_same_root(p1, p2):
            print 'OK'
        else:
            print 'wrong'
    for n1, n2 in negative:
        if not likely_same_root(n1, n2):
            print 'OK'
        else:
            print 'wrong'

# ANALYSIS
# manual = read_in_hand_alignments('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p')
# find_only_lexically_aligned_nodes(manual)
# proportion_of_order_preserving_alignments(manual)
# node_alignment_types(manual, source='dep')
# print_dict_for_table(alignment_type_frequency(manual, raw=True, amr_filter=quantity_expression_filter))
# print_dict_for_table(alignment_neatness(manual))
# see_neatness_examples('2-n', '../alignments/all_alignments.txt', 'parsed_datasets/parsed.p')
# overlapping_structures(manual)
# coverage('../alignments/iaa_alignments2.txt', 'parsed_datasets/iaa_parsed.p')
# see_unaligned('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p')
# see_unexpected_nodes('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p')
# print("COVERAGE - MANUAL ALIGNMENTS:")
# see_proportions_of_unaligned(read_in_hand_alignments('../alignments/all_alignments.txt', './parsed_datasets/parsed.p'))
# print("\nCOVERAGE - AUTOMATIC ALIGNMENTS:")
# automatic = get_full_alignment(unpickle('parsed_datasets/parsed.p'), read_neg_polarity_items('neg-polarity.txt'))
# see_proportions_of_unaligned(automatic)
# find_alignments_to_simplify('C', 'DG-AMR', '../alignments/dev_gold_dg_amr.txt', 'parsed_aligned.p')

# ALIGNMENT
# sentences = unpickle_parsed_data('parsed.p')
# neg_dict = read_neg_polarity_items('neg-polarity.txt')
# al = get_full_alignment(sentences, neg_dict)
# print_alignment(al, '../alignments/selected_test.txt', structure_type='paths')

# PARSING
# data for selected sentences: parsed.p
# data for random sentences: parsed_random.p
# data for both 'parsed_aligned.p'
#
# sent_dict = read_in_fixed_parses(read_and_parse('../data/random1_amrs.txt'), '../data/random1_parses.txt')
# sent_dict = unpickle_parsed_data('parsed.p')
# draw_amr_and_dep({21: sent_dict[21]}, '.')
# for item in sent_dict[13]['dep_graph'].items():
# 	print item
# pickle_parsed_data(sent_dict, 'parsed_random.p')

# EVALUATION
# print("INTER-ANNOTATOR")
# interannotator_both('../alignments/iaa_alignments2.txt', '../alignments/iaa_alignments1.txt',
#                     'parsed_datasets/iaa_parsed.p', mistake_type='both', show_mistakes=True)
# print("\nNode scores:")
# interannotator_nodes('../alignments/iaa_alignments2.txt', '../alignments/iaa_alignments1.txt',
#                      'parsed_datasets/iaa_parsed.p', mistake_type='both', show_mistakes=False)
# print("\nPath scores:")
# interannotator_paths('../alignments/iaa_alignments2.txt', '../alignments/iaa_alignments1.txt',
#                      'parsed_datasets/iaa_parsed.p', mistake_type='both', show_mistakes=False)

# print("MY RULE-BASED ALIGNER")
# print("\nTotal scores, my aligner, my data:")
# my_on_my_both('./alignments/all_alignments.txt', 'parsed_datasets/parsed.p',
#               mistake_type='both', show_mistakes=False)
# print("\nNode scores, my aligner, my data:")
# my_on_my_nodes('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p',
#                mistake_type='both', show_mistakes=False)
print("\nPath scores, my aligner, my data, gold parses:")
my_on_my_paths('./alignments/all_alignments.txt', 'parsed_datasets/parsed.p',
               mistake_type='both', show_mistakes=False)
# print("\nPath scores, my aligner, my data, automatic parses:")
# my_on_my_paths('../alignments/all_alignments.txt', 'parsed_datasets/auto_parsed_new.p',
#                mistake_type='both', show_mistakes=False)
# print("\nOracle path scores, my aligner, my data, gold parses:")
# oracle_evaluate('../alignments/all_alignments.txt', 'parsed_datasets/parsed.p',
#                 mistake_type='unmatched', show_mistakes=False)
# print("\nOracle path scores, my aligner, my data, automatic parses:")
# oracle_evaluate('../alignments/all_alignments.txt', 'parsed_datasets/auto_parsed_new.p',
#                 mistake_type='unmatched', show_mistakes=False)

# print("\nNode scores, my aligner, ISI data:")
# my_on_isi('../alignments/isi_test_gold.txt', 'parsed_datasets/isi_test_parsed.p',
#           mistake_type='both', show_mistakes=False)
# print("\nNode scores, my aligner, JAMR data:")
# my_on_jamr('../alignments/jamr_gold.txt', 'parsed_datasets/jamr_test_parsed.p',
#            mistake_type='both', show_mistakes=False)


# print("\nISI ALIGNER")
# print("\nNode scores, ISI aligner, ISI data:")
# isi_on_isi('../alignments/isi_test_gold.txt', '../alignments/isi_on_isi.txt',
#            'parsed_datasets/isi_test_parsed.p', mistake_type='both', show_mistakes=False)
# isi_on_isi_bare('../alignments/Nimas_data/no-role-test-gold.txt', '../alignments/isi_on_isi_no_roles.txt',
#                 'parsed_datasets/isi_test_parsed.p', mistake_type='both', show_mistakes=False)
# print("\nNode scores, ISI aligner, my data:")
# isi_on_my('../alignments/isi_on_my.txt', '../alignments/all_alignments.txt',
#           'parsed_datasets/parsed.p', mistake_type='both', show_mistakes=False)
# print("\nNode scores, ISI aligner, JAMR data:")
# isi_on_jamr('../alignments/isi_on_jamr.txt', '../alignments/jamr_gold.txt',
#           'parsed_datasets/jamr_test_parsed.p', mistake_type='both', show_mistakes=False)
# print("\nOracle path scores, ISI node alignments, my data, gold parses:")
# isi_as_oracle('../alignments/isi_on_my.txt', '../alignments/all_alignments.txt',
#               'parsed_datasets/parsed.p', mistake_type='both', show_mistakes=False)
# print("\nOracle path scores, ISI node alignments, my data, automatic_parses:")
# isi_as_oracle('../alignments/isi_on_my.txt', '../alignments/all_alignments.txt',
#               'parsed_datasets/auto_parsed.p', mistake_type='both', show_mistakes=False)

# print("\nJAMR ALIGNER")
# print("\nNode scores, JAMR aligner, my data:")
# jamr_on_my('../alignments/jamr_on_my_new.txt', '../alignments/all_alignments_for_jamr.txt',
#            'parsed_datasets/parsed_for_jamr.p', mistake_type='both', show_mistakes=False)
# print("\nNode scores, JAMR aligner, ISI data:")
# jamr_on_isi('../alignments/jamr_on_isi.txt', '../alignments/isi_test_gold.txt',
#             'parsed_datasets/isi_test_parsed.p', mistake_type='both', show_mistakes=False)
# print("\nNode scores, JAMR aligner, JAMR data:")
# jamr_on_jamr('../alignments/jamr_on_jamr.txt', '../alignments/jamr_gold.txt',
#             mistake_type='both', show_mistakes=False)
# print("\nOracle path scores, JAMR node alignments, my data, gold parses:")
# jamr_as_oracle('../alignments/jamr_on_my_new.txt', '../alignments/all_alignments_for_jamr.txt',
#                'parsed_datasets/parsed.p', mistake_type='both', show_mistakes=False)
# print("\nOracle path scores, JAMR node alignments, my data, automatic parses:")
# jamr_as_oracle('../alignments/jamr_on_my_new.txt', '../alignments/all_alignments_for_jamr.txt',
#                'parsed_datasets/auto_parsed.p', mistake_type='both', show_mistakes=False)
