# Levenshtein distance helps to understand the similarities and dissimilarities between genetic sequences. 

#Import library
import Levenshtein

# Example sequences
seq1 = "GTTTACA"
seq2 = "CAUGCA"

# Calculate Levenshtein distance
distance = Levenshtein.distance(seq1, seq2)
print(f"Levenshtein Distance: {distance}")
