import os
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms

def calculate_rmsd(ref_universe, query_universe, ref_resid_ids, query_resid_ids, inverse=False):
    """
    Calculate RMSD between reference and query structures for specified residues.
    
    Args:
        ref_universe: pdb path or MDAnalysis Universe for reference structure
        query_universe: pdb path or MDAnalysis Universe for query structure
        ref_resid_ids: List of reference residue IDs
        query_resid_ids: List of query residue IDs
        inverse: If True, calculate RMSD for all residues EXCEPT the specified ones
    
    Returns:
        float: RMSD value if successful, None otherwise
    """

    if isinstance(ref_universe, str):
        ref_universe = mda.Universe(ref_universe)
    if isinstance(query_universe, str):
        query_universe = mda.Universe(query_universe)
    
    # check if residues start at 0
    query_starts_at_zero = query_universe.residues[0].resid == 0
    query_resids = [i-1 for i in query_resid_ids] if query_starts_at_zero else query_resid_ids
    query_resid_str = ' '.join(map(str, query_resids))

    ref_starts_at_zero = ref_universe.residues[0].resid == 0
    ref_resids = [i-1 for i in ref_resid_ids] if ref_starts_at_zero else ref_resid_ids
    ref_resid_str = ' '.join(map(str, ref_resids))
    
    # inverse - is a scRMSD and other - is a motif RMSD
    if inverse:
        ref_sel = f"name CA and not resid {ref_resid_str}"
        query_sel = f"name CA and not resid {query_resid_str}"
    else:
        ref_sel = f"name CA and resid {ref_resid_str}"
        query_sel = f"name CA and resid {query_resid_str}"

    ref_atoms = ref_universe.select_atoms(ref_sel)
    query_atoms = query_universe.select_atoms(query_sel)

    if len(ref_atoms) == 0 or len(query_atoms) == 0:
        region = "non-motif" if inverse else "motif"
        print(f"Warning: No {region} CA atoms found in one or both structures")
        print('ref_sel', ref_sel)
        print('query_sel', query_sel)
        return None

    if len(ref_atoms) != len(query_atoms):
        print(f"Warning: Different number of atoms selected:")
        print(f"Reference: {len(ref_atoms)}, Query: {len(query_atoms)}")
        print(f"Reference starts at: {ref_universe.residues[0].resid}")
        print(f"Query starts at: {query_universe.residues[0].resid}")
        return None

    # Calculate RMSD
    rmsd_val = rms.rmsd(query_atoms.positions,
                        ref_atoms.positions,
                        center=True,
                        superposition=True)
    return rmsd_val

def find_best_self_consistency(gen_pdb, self_cons_base, motif_residues, num_seqs=8):
    """Find the self-consistency file with lowest non-motif RMSD to generated structure."""
    best_rmsd = float('inf')
    best_file = None
    
    try:
        gen = mda.Universe(gen_pdb)
        
        for i in range(1, num_seqs + 1):
            self_cons_pdb = f"{self_cons_base}_seq{i}.pdb"
            if not os.path.exists(self_cons_pdb):
                print(f"Warning: Missing self-consistency file {self_cons_pdb}")
                continue
                
            try:
                self_cons = mda.Universe(self_cons_pdb)
                nonmotif_rmsd = calculate_rmsd(gen, self_cons, motif_residues, motif_residues, inverse=True)
                if nonmotif_rmsd is None:
                    print('nonmotif_rmsd is None', self_cons_pdb, motif_residues)
                
                if nonmotif_rmsd is not None and nonmotif_rmsd < best_rmsd:
                    best_rmsd = nonmotif_rmsd
                    best_file = self_cons_pdb
                    
            except Exception as e:
                print(f"Error processing {self_cons_pdb}: {e}")
                continue
                
        return best_file, best_rmsd
        
    except Exception as e:
        print(f"Error loading generated structure {gen_pdb}: {e}")
        return None, None

def calculate_scrmsd(
        ref_pdb, 
        gen_pdb, 
        self_cons_base, 
        gen_motif_residues, 
        ref_motif_residues, 
        num_cs_seqs=8
        ):
    """Calculate RMSD and best non-motif RMSD, selecting best self-consistency file."""

    # First find best self-consistency file based on non-motif RMSD
    best_self_cons, best_nonmotif_rmsd = find_best_self_consistency(
        gen_pdb, self_cons_base, gen_motif_residues, num_cs_seqs
    )
    
    if best_self_cons is None:
        print(f"Warning: scRMSD calculation problems for {gen_pdb}")
        return None, None
        
    # # Load reference and best self-consistency for motif RMSD
    # ref = mda.Universe(ref_pdb)
    # self_cons = mda.Universe(best_self_cons)
    
    # # Calculate motif RMSD between reference and best self-consistency
    # motif_rmsd = calculate_rmsd(ref, self_cons, ref_motif_residues, gen_motif_residues, inverse=False)
    
    return best_nonmotif_rmsd, best_self_cons