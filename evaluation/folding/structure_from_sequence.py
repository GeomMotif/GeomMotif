import os
import torch
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.distributed as dist
import argparse

from utils.setup_ddp import setup_ddp
from utils.set_seed import set_seed
from utils.loading import load_data

from cheap.esmfold import esmfold_v1
    
class ESMFolder:
    def __init__(self, device: str = "cpu", batch_size: int = 8):
        self.model = esmfold_v1()
        self.model = self.model.eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, proteins: List[str], indices: List[int], pdb_path: str, pdb_ids: List[str]) -> List[float]:
        if not proteins:
            return [0]
        
        # Process sequences in batches
        all_plddt_scores = []
        for i in tqdm(range(0, len(proteins), self.batch_size), total = len(proteins)//self.batch_size):
            batch_proteins = proteins[i:i + self.batch_size]
            batch_pdb_ids = pdb_ids[i:i + self.batch_size]
            
            outputs = self.model.infer_pdbs(batch_proteins)
            
            for output, pdb_id in zip(outputs, batch_pdb_ids):
                file_path = os.path.join(pdb_path, f"{pdb_id}.pdb")
                with open(file_path, "w") as f:
                    f.write(output) 
        return 

def fold_sequences(proteins: List[str], index_list: List[int], batch_size: int = 1, device="cuda", pdb_path="", pdb_ids=[]) -> Dict[str, float]:
    fold_fn = ESMFolder(device, batch_size=batch_size)
    os.makedirs(pdb_path, exist_ok=True)
    
    fold_fn(proteins=proteins, indices=index_list, pdb_path=pdb_path, pdb_ids=pdb_ids)
    return


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Fold sequences from meta table")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--name-col", type=str, default='gen_pdb_name')
    parser.add_argument("--pdb_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    setup_ddp()

    predictions, names = load_data(args.input_file, args.name_col)

    if dist.is_available() and dist.is_initialized():
        device_id = dist.get_rank()
        total_device_number = dist.get_world_size()
    else:
        device_id = 0
        total_device_number = 1

    set_seed(device_id)
    start_ind = device_id * (len(predictions) // total_device_number)
    end_ind = (device_id + 1) * (len(predictions) // total_device_number)

    indexes = np.random.default_rng(seed=0).permutation(len(predictions))

    if device_id + 1 == total_device_number:
        predictions_device = [predictions[ind] for ind in indexes[start_ind:]]
        index_list = indexes[start_ind:]
        names_device= [names[ind] for ind in indexes[start_ind:]]
    else:
        predictions_device = [predictions[ind] for ind in indexes[start_ind:end_ind]]
        index_list = indexes[start_ind:end_ind]
        names_device= [names[ind] for ind in indexes[start_ind:end_ind]]
    
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    fold_sequences(
        proteins=predictions_device, 
        index_list=index_list, 
        batch_size=args.batch_size, 
        device=device, 
        pdb_path=args.pdb_path, 
        pdb_ids= names_device
    )
