from utils import *
from parse import process_amr_line, process_amr


def prepare_corpus():
    signatures = []
    sign = []
    with open("./alignments/amr_ud_alignments_ldc_release.txt", "w") as out_file:
        for line in open("./alignments/amr_ud_alignments_ldc.txt", "r"):
            l = line.rstrip()
            if l.startswith('//'):
                # preped.append(line)
                out_file.write(line)
            elif l.split('.')[0].isdigit():
                # preped.append(line)
                out_file.write(line)
            elif l.startswith('# ::save'):
                sign.append(line)
                signatures.append(sign)
                sign = []
            elif l.startswith('#'):
                sign.append(line)
            elif not l:
                # preped.append(line)
                out_file.write(line)
            elif l.split('\t')[0].isdigit():
                index, word, lemma, rest = l.split('\t', 3)
                out_file.write('\t'.join([index, 'word', 'lemma', rest])+'\n')
            else:
                if len(l.split('    #    ')) < 2:
                    print(line)
                amr, ud = l.split('    #    ')
                amr_side = ' '.join([contract_node_label(x) if not is_edge_label(x) else x for x in amr.split()])
                ud_side = ' '.join([str(contract_node_label(x)) if not (is_edge_label(x) or x in ['(', ')', '|'])
                                          else x for x in ud.split()])
                # preped.append('    #    '.join([amr_side, ud_side]))
                out_file.write('    #    '.join([amr_side, ud_side])+'\n')

    with open("./alignments/amr_ud_alignments_ldc_signatures.txt", "w") as out_file:
        for s in signatures:
            out_file.write(s[0])

# prepare_corpus()


# with open("./alignments/amr_ud_alignments_ldc_signatures.txt", "r") as f:
#     signature_list = f.read().splitlines()
#     signatures = {i: signature_list[i] for i in range(1, len(signature_list))}
# prefix = "/Users/ida/Documents/studies/AMR/data/corpus/release1/"
# corpus = "unsplit/amr-release-1.0-all.txt"


def extract_original_amrs(in_file, signature_list):
    annotated_amrs = {s: [] for s in signature_list}
    include = False
    amr = []
    sig = ''
    for line in open(in_file, "r"):
        if line.startswith("# ::id"):
            if line.strip() in signature_list:
                sig = line.strip()
                include = True
                amr.append(line)
        elif include:
            if line.strip():
                amr.append(line)
            else:
                include = False
                annotated_amrs[sig] = amr
                amr = []
        else:
            pass
    return annotated_amrs


# first_try_amrs = extract_original_amrs(prefix+corpus)


def write_out_annotated_amrs(first_try_amrs, signature_list):
    with open("./alignments/aligned_amrs_reconstructed.txt", "w") as f:
        for s in signature_list:
            amr = first_try_amrs[s]
            for l in amr:
                f.write(l)
            f.write('\n')


# write_out_annotated_amrs()


# annotated_amrs = extract_original_amrs("./alignments/aligned_amrs_patched.txt")

def read_in_shell():
    shell_dict = {}
    index = 0
    alignments = []
    parse = []
    amr_part = True
    sent_ids = []
    for line in open("./alignments/amr_ud_alignments_ldc_release.txt", "r"):
        l = line.rstrip()
        if l.startswith('//'):
            amr_part = False
            index = 0
        elif l.split('.')[0].isdigit():
            index = int(l.split('.')[0])
            shell_dict[index] = []
            sent_ids.append(index)
        elif not l:
            if amr_part:
                shell_dict[index].append(alignments)
                alignments = []
            else:
                shell_dict[sent_ids[index]].append(parse)
                parse = []
                index += 1
        elif l.split('\t')[0].isdigit():
                id, word, lemma, rest = l.split('\t', 3)
                parse.append((int(id), rest))
        else:
            alignments.append(l)
    return shell_dict


def normalize_word(word):
    if word == 'five-cents':
        return ['five', 'cents']
    if word == 'cannot':
        return ['can', 'not']
    if word[0] == '(' and word[-1] == ')':
        merge = ['-LRB-']
        merge.extend(normalize_word(word[1:-1]))
        merge.append('-RRB-')
        return merge
    if word[0] in ["'", '"'] and word[-1] in ["'", '"'] and len(word) > 1:
        merge = [word[0]]
        merge.extend(normalize_word(word[1:-1]))
        merge.append(word[-1])
        return merge
    if len(word) > 3 and word[-3:] == '...':
        main_w = normalize_word(word[:-3])
        main_w.extend(word[-3:])
        return main_w
    if len(word) > 1 and word[-1] in ['.', ',', '!', '?', ';', '"'] and word not in ['W.R.', 'U.S.', 'D.', 'i.e.']:
        main_w = normalize_word(word[:-1])
        main_w.extend(word[-1])
        return main_w
    if len(word) > 1 and word[0] == '"':
        merge = ['"']
        merge.extend(normalize_word(word[1:]))
        return merge
    if word[-2:] in ["'s", "'t", "'m", "'d"] and len(word)>2:
        return [word[:-2], word[-2:]]
    if word[0] =='#':
        merge = ['#']
        merge.extend(normalize_word(word[1:]))
        return merge
    else:
        return [word]


def write_out_alignments_parses(shell, annotated_amrs, signature_list):
    with open("./alignments/amr_ud_alignments_ldc_reconstructed.txt", "w") as out_f,\
        open("./alignments/ud_parses_ldc_reconstructed.txt", "w") as out_parse:
        for num, (id, shells) in enumerate(shell.items()):
            s = signature_list[num]
            out_f.write(str(id)+'.\n')
            for l in annotated_amrs[s][:3]:
                out_f.write(l)
            only_amr = flatten_list([process_amr_line(l) for l in annotated_amrs[s][3:]])
            amr_concepts, amr2jamr, amr2isi, amr_relations = process_amr(only_amr, jamr=False, isi=False)
            sent = annotated_amrs[s][1].strip()[8:].split()
            words = flatten_list([normalize_word(w) for w in sent])
            word_dict = {str(i+1): words[i] for i in range(len(words))}

            for a in shells[0]:
                amr, ud = a.split('    #    ')
                amr_full = [expand_node_label(x, amr_concepts) if (not is_edge_label(x) and x not in ['(', ')', '|']) else x
                            for x in amr.split()]
                ud_full = [expand_node_label(x, word_dict) if (not is_edge_label(x) and x not in ['(', ')', '|']) else x
                            for x in ud.split()]
                out_f.write('    #    '.join([' '.join(amr_full), ' '.join(ud_full)]))
                out_f.write('\n')
            out_f.write('\n')

            for index, rest in shells[1]:
                word = word_dict[str(index)]
                out_parse.write(str(index)+'\n')
                out_parse.write(word+'\n')
                out_parse.write('_'+'\n')
                for x in rest.split('\t')[:-1]:
                    out_parse.write(x+'\n')
                out_parse.write('_')
                out_parse.write('%')
            out_parse.write('@')



