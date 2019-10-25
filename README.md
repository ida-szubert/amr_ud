This resource contains data and code for the paper

* Ida Szubert, Adam Lopez, and Nathan Schneider (2018). A structured syntax-semantics interface for English-AMR alignment. _Proceedings of NAACL-HLT_. <http://people.cs.georgetown.edu/nschneid/p/amr2dep.pdf>

which describes a representation, dataset, and algorithms for aligning nodes and subgraphs of [Abstract Meaning Representation](http://amr.isi.edu/) (AMR) semantic structures with nodes and subgraphs of syntactic parses in the [Universal Dependencies](http://universaldependencies.org/) (UD) framework.

## AUTOMATIC ALIGNER
To run the rule-base aligner described in the paper you'll need an AMR file and a file with UD parse of the same sentences in the same order in CoNLL format.
The aligner does not require any specific version of UD annotation.

Run the align script:
`python2 align.py -p "UD_parse_file" -a "AMR_file" -o "output_alignment_file"`

To run the aligner on and AMR only, you need to start a CoreNLP server (see https://stanfordnlp.github.io/CoreNLP/corenlp-server.html) and run
`python2 align.py -a "AMR_file" -o "output_alignment_file"`

If you'd like to save the automatic UD parses, run:
`python2 align.py -a "AMR_file" -o "output_alignment_file" -w "output_ud_parse_file"`

## MANUAL ALIGNMENTS

### Reconstructing
To recover our manual alignments you will need to have access to [AMR Release 1.0](https://catalog.ldc.upenn.edu/LDC2014T12).
Unzip the release files and cd to corpus/release1/unsplit. From there run:

```
cat amr-release-1.0-bolt.txt amr-release-1.0-consensus.txt  amr-release-1.0-dfa.txt  amr-release-1.0-mt09sdl.txt amr-release-1.0-xinhua.txt amr-release-1.0-proxy.txt > ~/YOUR_PATH/amr_ud/data/amr-release-1.0-all.txt
```

to concatenate all release files into one.


Then cd to amr_ud and run:

```
python2 dp1.py
patch ./alignments/aligned_amrs_reconstructed.txt -i ./alignments/amrs.patch -o ./alignments/aligned_amrs.txt
python2 dp2.py
patch ./alignments/ud_parses_ldc_reconstructed.txt -i ./alignments/ud_parses.patch -o ./alignments/ud_parses_patched.txt
tr '\n%@' '\t\n\n' <./alignments/ud_parses_patched.txt >./alignments/ud_parses.txt
patch ./alignments/amr_ud_alignments_ldc_reconstructed.txt -i ./alignments/alignments.patch -o ./alignments/amr_ud_alignments.txt
rm ./alignments/aligned_amrs_reconstructed.txt ./alignments/ud_parses_patched.txt ./alignments/ud_parses_ldc_reconstructed.txt ./alignments/amr_ud_alignments_ldc_reconstructed.txt
```

The following files should be created in the amr_ud/alignments directory:
* _aligned_amrs.txt_: contains all AMRs for which hand alignments were created
* _ud_parses.txt_: contains hand-corrected UD parses for those AMRs
* _amr_ud_alignments_: contains manual alignments

### Format
The format of the alignments is as follows:

AMR subgraph&nbsp;&nbsp;&nbsp;&nbsp;#&nbsp;&nbsp;&nbsp;&nbsp;UD subgraph

A subgraph might be a single node, or it might contain nodes and edges:


notation | meaning
--- | ----
x/word1 | node x, whose label is _word1_
x/word1 :rel y/word2 | a subgraph consisting of x, its child y, and the edge connecting them labeled rel
x/word1 ( :rel1 y/word2 ) \| :rel2 z/word3 | a subgraph consisting of x, two of its children, and the connecting edges

( ) groups subgraphs, | separates children of one parent node

### Non-LDC alignments
The file _'amr_ud_alignment_nonldc.txt'_ contains alignments and UD parses for AMRs which were not included in the LDC AMR release, and which can be freely shared.
All those AMRs, parses, and alignments are also included in the full alignment corpus reconstructed according to the instructions above.
