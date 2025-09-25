import pandas as pd

def load_data(file_path, name_col=None):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        seqs = df['gen_seq']
        names = df[name_col]
    elif file_path.endswith('.fasta'):
        seqs = []
        names = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    names.append(line.strip('>')[:-1])
                    # print(line.strip('>')[: -1])
                    # print(line.strip('>'))
                else:
                    seqs.append(line.strip())
                    # print(line.strip()[:-1])
                    # print(line.strip())
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return seqs, names


