# Import library
import pandas as pd
import numpy as np

#load the file path
vcf_filepath = "C:/Users/sugam patel/Downloads/sample.vcf"

vcf_data = pd.read_csv(vcf_filepath,sep = '\t',comment = '#',header = None)
vcf_data.shape
