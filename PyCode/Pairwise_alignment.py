#Import library
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# Sequences to align
seq1 = "ACTGCTAGCTAG"
seq2 = "ACTGCTAGCTAGG"

# Perform global alignment
alignments = pairwise2.align.globalxx(seq1, seq2)
for alignment in alignments:
    print(format_alignment(*alignment))
