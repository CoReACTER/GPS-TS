# stdlib
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Molecule and molecule graph representations
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

# Utility functions
from gpsts.utils import (
    construct_molecule_from_adjacency_matrix,
    oxygen_edge_extender,
    prepare_reaction_for_input,
    read_adjacency_matrix,
)


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


# Reactions from BH2O-36 that can be included in this benchmark
# For now, ignore all water-assisted reactions
# TODO: eventually, would be nice if we could handle water-assisted mechanisms, water wires, etc.
reactions = [
    "aceticanhydride",
    # "amide_2_1",
    # "amide_2_2",
    "basic_epoxide_1",
    "basic_epoxide_2",
    "borohydride",
    "carbonate",
    "cl2co",
    "diazonium",
    "epoxide1",
    "epoxide2",
    "ester",
    # "furan1",
    # "furan2",
    # "furan3",
    "imine",
    "iminium",
    "lactone"
]


def process_bh2o(
    xyz_dir: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:

    """

    Generate benchmark data set from the reactions in the BH2O-36 dataset

    Args:
        xyz_dir (str | Path): Path to directory where BH2O-36 *.xyz files are stored
        clean (bool): If True (default True), process reaction data so that they can be easily dumped as a JSON file

    Returns:
        reaction_data (List[Dict[str, Any]]): List of data points

    """

    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    logging.info(f"BEGINNING PROCESSING BH2O-36 DATASET WITH ROOT DIRECTORY: {xyz_dir}")

    reaction_data = list()

    for reaction in reactions:
        logging.info(f"\tProcessing reaction: {reaction}")

        rct_file = glob(f"{reaction}_rct*.xyz", root_dir=xyz_dir)[0]
        pro_file = glob(f"{reaction}_pro*.xyz", root_dir=xyz_dir)[0]

        charge = int(rct_file.split(".")[0].split("_")[-1])

        # Construct molecule graphs
        rct = Molecule.from_file(xyz_dir / rct_file)
        rct.set_charge_and_spin(charge)
        rct_mg = oxygen_edge_extender(MoleculeGraph.with_local_env_strategy(rct, OpenBabelNN()))
        
        pro = Molecule.from_file(xyz_dir / pro_file)
        pro.set_charge_and_spin(charge)
        pro_mg = oxygen_edge_extender(MoleculeGraph.with_local_env_strategy(pro, OpenBabelNN()))

        # In some cases, there are multiple reactants or products - need to check for disconnected subgraphs
        rct_mgs = rct_mg.get_disconnected_fragments()
        pro_mgs = pro_mg.get_disconnected_fragments()

        # Certain molecules (within this dataset) are always neutral, and others are always ionic
        neutral_formulas = ["H2 O1", "N2", "C1 H5 N1"]
        negative_formulas = ["O1 H1"]

        # Make sure charge is distributed properly
        if len(rct_mgs) == 2 and charge != 0:
            for ii, mg in enumerate(rct_mgs):
                if mg.molecule.composition.alphabetical_formula in neutral_formulas:
                    mg.molecule.set_charge_and_spin(0)
                    if ii == 0:
                        rct_mgs[1].molecule.set_charge_and_spin(charge)
                    else:
                        rct_mgs[0].molecule.set_charge_and_spin(charge)

                    break
                elif mg.molecule.composition.alphabetical_formula in negative_formulas:
                    mg.molecule.set_charge_and_spin(-1)
                    if ii == 0:
                         rct_mgs[1].molecule.set_charge_and_spin(0)
                    else:
                        rct_mgs[0].molecule.set_charge_and_spin(0)
                    break

        if len(pro_mgs) == 2 and charge != 0:
            for ii, mg in enumerate(pro_mgs):
                if mg.molecule.composition.alphabetical_formula in neutral_formulas:
                    mg.molecule.set_charge_and_spin(0)
                    if ii == 0:
                        pro_mgs[1].molecule.set_charge_and_spin(charge)
                    else:
                        pro_mgs[0].molecule.set_charge_and_spin(charge)

                    break
                elif mg.molecule.composition.alphabetical_formula in negative_formulas:
                    mg.molecule.set_charge_and_spin(-1)
                    if ii == 0:
                        pro_mgs[1].molecule.set_charge_and_spin(0)
                    else:
                        pro_mgs[0].molecule.set_charge_and_spin(0)
                    break

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"BH2O-36: {reaction}", clean=clean))
    
    return reaction_data
