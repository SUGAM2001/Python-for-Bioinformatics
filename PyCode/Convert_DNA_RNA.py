# import library
from Bio.Seq import Seq

def dna_to_rna(dna_sequence):
  #seq object
  dna_seq = Seq(dna_sequence)

 # Use the transcribe method to convert dna to rna
  rna_seq = dna_seq.transcribe()
  return ran_seq

dna_seq = "ATGCATCCGATCTGACTGCATC"
rna_sequence = dna_to_rna(dna_sequence)
print(f"DNA sequence: {dna_seq}")
print(f"RNA sequence: {rna_seq}")




