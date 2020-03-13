import numpy as np


CNV_path = '/home/neythen/Desktop/Projects/Cancer/data/data-CNV/TCGA-CNV/ACOLD_p_TCGA_Batch17_SNP_N_GenomeWideSNP_6_A01_466074.nocnv_grch38.seg.v2.txt'


with open(CNV_path) as f:
    copy_number_data = {}
    f.readline() # skip headings
    for line in f:
        l = line.split('\t')


        chromosome = l[1].rstrip('\n\r')
        start = int(l[2].rstrip('\n\r'))
        end = int(l[3].rstrip('\n\r'))
        copy_number = float(l[5].rstrip('\n\r'))

        if chromosome in copy_number_data:
            copy_number_data[chromosome].append([start, end, copy_number])
        else:
            copy_number_data[chromosome] = []


#make chromosome 1
