# stdlib
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

# Molecule and molecule graph representations
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

# Utilities
from gpsts.utils import (
    MAX_BENCHMARK_REACTION_NUMATOMS,
    oxygen_edge_extender,
    prepare_reaction_for_input
)


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


# Causes problems with atom mapping
exclude_bh9 = ["02_25"]


def mg_from_bh9(path: Path) -> MoleculeGraph:

    """
    
    Create a MoleculeGraph for a BH9 molecule from an *.xyz file

    Args:
        path (Path): Path to *.xyz file
    
    Returns:
        mg (MoleculeGraph): Graph representation of the molecule based on this *.xyz file

    """

    with open(path) as file_obj:
        lines = file_obj.readlines()
        charge, spin = lines[1].strip().split()
        charge = int(charge)
        spin = int(spin)

    mol = Molecule.from_file(path)
    mol.set_charge_and_spin(charge, spin_multiplicity=spin)
    mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    mg = oxygen_edge_extender(mg)

    return mg


def process_bh9(
    xyz_dir: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    
    """

    Generate benchmark data set from the reactions in the BH9 dataset

    Args:
        xyz_dir (str | Path): Path to directory where BH9 *.xyz files are stored
        clean (bool): If True (default True), process reaction data so that they can be easily dumped as a JSON file

    Returns:
        reaction_data (List[Dict[str, Any]]): List of data points

    """

    logging.info(f"BEGINNING PROCESSING BH9 DATASET WITH ROOT DIRECTORY: {xyz_dir}")

    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    # Identify TS
    # We only do this so that we can effectively grab the reaction reactants and products
    ts_files = glob("*TS.xyz", root_dir=xyz_dir)

    reaction_data = list()

    for ts_file in ts_files:
        rxn_id = ts_file.split("TS")[0]
        
        logging.info(f"\tProcessing reaction: {rxn_id}")

        if rxn_id in exclude_bh9:
            logging.info(f"\t\tSKIPPING: atom mapping problematic")
            continue

        ts_mol = Molecule.from_file(xyz_dir / ts_file)

        rct_files = glob(f"{rxn_id}R*.xyz", root_dir=xyz_dir)
        rct_mgs = list()
        for rct_file in rct_files:
            rct_path = xyz_dir / rct_file
            rct_mgs.append(mg_from_bh9(rct_path))

        pro_files = glob(f"{rxn_id}P*.xyz", root_dir=xyz_dir)
        pro_mgs = list()
        for pro_file in pro_files:
            pro_path = xyz_dir / pro_file
            pro_mgs.append(mg_from_bh9(pro_path))

        total_length = sum([len(x.molecule) for x in rct_mgs])

        if total_length > MAX_BENCHMARK_REACTION_NUMATOMS:
            logging.info(f"\t\tSKIPPING: reaction too large")
            continue

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"BH9:{rxn_id}", clean=clean))

    return reaction_data
