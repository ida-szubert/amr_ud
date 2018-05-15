from data_prep import *

with open("./alignments/amr_ud_alignments_ldc_signatures.txt", "r") as f:
    signature_list = f.read().splitlines()
# prefix = "/Users/ida/Documents/studies/AMR/data/corpus/release1/"
corpus = "data/amr-release-1.0-all.txt"

first_try_amrs = extract_original_amrs(corpus, signature_list)
write_out_annotated_amrs(first_try_amrs, signature_list)
