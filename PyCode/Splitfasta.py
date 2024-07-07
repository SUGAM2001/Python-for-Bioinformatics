#Import library 
from Bio import SeqIO

# Function to reformat a single-line FASTA file to multi-line FASTA with equal parts
def reformat_fasta(input_file, output_file, part_size):
    # Parse the single-line FASTA file
    record = SeqIO.read(input_file, "fasta")
    
    # Extract sequence
    sequence = str(record.seq)
    
    # Break the sequence into equal parts
    with open(output_file, "w") as file:
        # Write the header
        file.write(f">{record.id}\n")
        
        # Write the sequence in parts
        for i in range(0, len(sequence), part_size):
            file.write(sequence[i:i+part_size] + "\n")

# Input and output file names
input_file = "C:/Users/sugam patel/Downloads/sequence_multi.fasta"  # Replace with your single-line FASTA file
output_file = "C:/Users/sugam patel/Downloads/multi_line_fasta.fasta"


part_size = 10000  # Adjust part size as needed

# Reformat the FASTA file
reformat_fasta(input_file, output_file, part_size)

print(f"Multi-line FASTA saved to {output_file}")
