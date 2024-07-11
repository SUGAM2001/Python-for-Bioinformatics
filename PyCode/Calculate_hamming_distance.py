
def hamming_distance(seq1,seq2):
    # check if both the seq are same or not
    if len(seq1) != len(seq2):
        raise ValueError("Sequence must be of equal number")
        
    # Initialize the distance 
    distance = 0
    # Iterate over the indices of the sequences
    for i in range(len(seq1)):
        # compare seq1 & seq 2 elements
        if seq1[i] != seq2[i]:
            # Increment the distance counter if they differ
            distance += 1
            
    return distance

seq1 = "ATGCTGACTACG"
seq2 = "AGGTTCATCACC"
distance = hamming_distance(seq1,seq2)
print(f"Hamming distance between two sequence is: {distance}")
