import numpy as np
from Bio.PDB import PDBParser

def get_plddt(pdb_file):
    """Extract confidence score from PDB file (stored in beta factors)."""
    try:
        p = PDBParser(QUIET=True)
        structure = p.get_structure("PDB", pdb_file)
        
        scores = []
        for res in structure.get_residues():
            for atom in res:
                scores.append(atom.bfactor)
        
        return np.mean(scores) if scores else None
    except Exception as e:
        print(f"Error getting confidence score from {pdb_file}: {e}")
        return None