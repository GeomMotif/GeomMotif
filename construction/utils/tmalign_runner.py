import os
import shutil
import subprocess
import logging
from typing import *
import numpy as np
import re


def run_tmalign(query: str, reference: str, fast: bool = False) -> float:
    """
    Run TMalign on the two given input pdb files
    """
    assert os.path.isfile(query), f'file {query} does not exist'
    assert os.path.isfile(reference), f'file {reference} does not exist'

    if os.path.getsize(query) == 0 or os.path.getsize(reference) == 0:
        # logging.warning(f"Empty file detected: {query if os.path.getsize(query) == 0 else reference}")
        return np.nan

    # Check if TMalign is installed
    exec = shutil.which("TMalign")
    if not exec:
        raise FileNotFoundError("TMalign not found in PATH")

    # Build the command
    cmd = f"{exec} {query} {reference}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"Tmalign failed on {query}|{reference}, returning NaN")
        return np.nan

    # Parse the output more carefully
    output_lines = output.decode().split("\n")
    tm_scores = []
    lens= []
    for line in output_lines:
        if line.startswith("TM-score"):
            try:
                # Split the line and look for the score
                parts = line.split()
                for i, part in enumerate(parts):
                    if "=" in part and i + 1 < len(parts):
                        try:
                            score = float(parts[i + 1])
                            tm_scores.append(score)
                            break
                        except ValueError:
                            continue
            except Exception as e:
                logging.warning(f"Error parsing line: {line}, error: {str(e)}")
                continue

        elif line.startswith("Length of "):
            l = re.findall(r":\s+([0-9.]+)", line)[0]
            lens.append(l)
            

    if not tm_scores:
        # logging.warning(f"No valid TM scores found in output for {query}|{reference}")
        return np.nan

    # Return the second TM-score (normalized by length of reference structure)
    return tm_scores[1] if len(tm_scores) > 1 else tm_scores[0], ':'.join(lens), query, reference
    
    # score_lines= []
    # for line in output.decode().split("\n"):
    #     if line.startswith("TM-score"):
    #         score_lines.append(line)

    # if not score_lines:
    #     logging.warning(f"No valid TM scores found in output for {query}|{reference}")
    #     return np.nan

    # # Fetch the chain number
    # key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    # score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    # results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    # return results_dict["Chain_2"]  # Normalize by reference length

