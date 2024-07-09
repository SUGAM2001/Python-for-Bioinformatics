#Import library
from Bio import SeqIO

file_path = "C:/Users/sugam patel/Downloads/sequence.gb"

gb_obj = SeqIO.read(file_path,"gb")
print(gb_obj)

# print genebank id only
gb_id = gb_obj.id
print(gb_id)

# print genebank description 
gb_des = gb_obj.description
print(gb_des)

# print gene bank name
gb_name = gb_obj.name
print(gb_name)

# To check the record sequence length
gb_seq = gb_obj.seq
print(len(gb_seq))

# To check the feature type
gb_feature = gb_obj.features
features = [feature.type for feature in gb_feature]
features = set(features)
features = list(features)
print(features)
