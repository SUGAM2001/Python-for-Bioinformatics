#Import library

from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# Try with random sequence

ntsequence = "ATCGTCAGTCATTAAGCTA"
gc_content = gc_fraction(ntsequence)*100
print(gc_content)

# Set the file path

file_path = "C:/Users/sugam patel/Downloads/seq_fasta.fasta"

for seqobj in SeqIO.parse(file_path,"fasta"):
    sequence = seqobj.seq
    gc_content = gc_fraction(sequence)*100
    print(f"GC content of {seqobj.id} is {gc_content: .2f}%")


for seqobj in SeqIO.parse(file_path,"fasta"):
    sequence = seqobj.seq
    gc_content = gc_fraction(sequence[100:10000])*100
    print(f"GC content of {seqobj.id} is {gc_content: .2f}%")
