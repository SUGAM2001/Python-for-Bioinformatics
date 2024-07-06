# Import important library
import pandas as pd
import os
from Bio import Entrez, SeqIO



# Define a function to download nucleotide data
def download_nucleotide_data(accession_number):
    handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return record

# You can choose your required accession number
accession_number = "NM_000207"  
nucleotide_data = download_nucleotide_data(accession_number)

# Display the nucleotide data
print(nucleotide_data)


# save this data in fata format
file_name = "NM_000207.fasta"  # you can set the path 
SeqIO.write(nucleotide_data, file_name, "fasta")


#To read the fasta file
seq_obj = SeqIO.read(file_name,"fasta")

# To check the type of file
type(seq_obj)

#To check the seq id
seq_id = seq_obj.id
print(seq_id)

# check the sequence name
seq_name = seq_obj.name
seq_name

# check the description 
print(seq_obj.description)

# TO check the seq.
sequence = seq_obj.seq
print(sequence)

# TO check the length of the sequence
print(len(sequence))

