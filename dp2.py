from data_prep import *

with open("./alignments/amr_ud_alignments_ldc_signatures.txt", "r") as f:
    signature_list = f.read().splitlines()
annotated_amrs = extract_original_amrs("./alignments/aligned_amrs.txt", signature_list)
shell = read_in_shell()
write_out_alignments_parses(shell, annotated_amrs, signature_list)
