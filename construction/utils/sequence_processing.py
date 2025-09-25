from Bio import SeqIO
import json
import pandas as pd

def read_fasta(fasta_file):
    records = SeqIO.parse(fasta_file, "fasta")
    seqs = []
    names= []
    for i, record in enumerate(records):
        seqs.append(str(record.seq))
        names.append(str(record.id))
    return seqs, names

def read_json(json_file):
    with open(json_file) as f:
        seqs= json.load(f)
    names= list(range(len(seqs)))
    return seqs, names

def read_table(table_file):
    if 'csv' in table_file:
        df = pd.read_csv(table_file)
    else:
        df = pd.read_table(table_file)
    seqs = df.Sequence.to_list()
    names= df.index.to_list()
    return seqs, names

def load_data(input_file):
    if 'fasta' in input_file:
        seq_list, names = read_fasta(input_file)
    
    elif 'tsv' in input_file or 'csv' in input_file:
        seq_list, names = read_table(input_file)

    elif 'json' in input_file:
        seq_list, names = read_json(input_file)
    return seq_list, names