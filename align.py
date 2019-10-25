import optparse
from alignment import read_neg_polarity_items, get_full_alignment, print_alignment
from parse import read_in_fixed_parses, read, read_and_parse, write_out_parses

parser = optparse.OptionParser()
parser.add_option('-p', '--parses', dest="ud_file",
                  help="file with UD parses")
parser.add_option('-a', '--amrs', dest="amr_file",
                  help="file with AMRs")
parser.add_option('-o', '--output', dest="output_file",
                  help="alignment file")
parser.add_option('-w', '--write_out', dest="write_out_ud",
                  default="",
                  help="file to which write UD parses; if none, don't write out")
(opts, _) = parser.parse_args()

if opts.ud_file:
    sentences = read_in_fixed_parses(read(opts.amr_file), opts.ud_file)
else:
    sentences = read_and_parse(opts.amr_file)
    if opts.write_out_ud:
        write_out_parses(sentences, opts.write_out_ud)
neg_dict = read_neg_polarity_items('neg-polarity.txt')
alignments = get_full_alignment(sentences, neg_dict)
print_alignment(alignments, opts.output_file)
