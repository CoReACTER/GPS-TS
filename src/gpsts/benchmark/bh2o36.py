import logging
from glob import glob
from pathlib import Path

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import oxygen_edge_extender, OpenBabelNN

from gpsts.utils import (
    construct_molecule_from_adjacency_matrix,
    prepare_reaction_for_input,
    read_adjacency_matrix,
)


reactions = [
    "aceticanhydride",
    "amide_2_1",
    "amide_2_2",
    "basic_epoxide_1",
    "basic_epoxide_2",
    "borohydride",
    "carbonate",
    "cl2co",
    "diazonium",
    "epoxide1",
    "epoxide2",
    "ester",
    "furan1",
    "furan2",
    "furan3",
    "imine",
    "iminium",
    "lactone"
]


def process_bh2o(
    xyz_dir: str | Path
) -> List[Dict[str, Any]]:

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    logging.info(f"BEGINNING PROCESSING BH2O-36 DATASET WITH ROOT DIRECTORY: {xyz_dir}")

    reaction_data = list()

    for reaction in reactions:
        logging.info(f"\tProcessing reaction: {reaction}")

        rct_file = glob(f"{reaction}_rct*.xyz", root_dir=xyz_dir)[0]
        pro_file = glob(f"{reaction}_pro*.xyz", root_dir=xyz_dir)[0]

        charge = int(rct_file.split(".")[0].split("_")[-1])

        # Construct molecule graphs
        rct = Molecule.from_file(base_dir / rct_file)
        rct.set_charge_and_spin(charge)
        rct_mg = oxygen_edge_extender(Moleculegraph.with_local_env_strategy(rct, OpenBabelNN()))
        
        pro = Molecule.from_file(base_dir / pro_file)
        pro.set_charge_and_spin(charge)
        pro_mg = oxygen_edge_extender(Moleculegraph.with_local_env_strategy(pro, OpenBabelNN()))

        # In some cases, there are multiple reactants or products - need to check for disconnected subgraphs
        rct_mgs = rct_mg.get_disconnected_fragments()
        pro_mgs = pro_mg.get_disconnected_fragments()

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"BH2O-36: {reaction}"))
    
    return reaction_data
