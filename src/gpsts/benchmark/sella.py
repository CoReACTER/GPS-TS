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


SELLA_EXCLUDE_RXNS = [
    15, 17, 22, 43, 50, 53, 61, 72, 75, 80, 102, 130, 131 136, 142, 144, 147, 168, 208, 216, 217, 228, 230, 233, 241
]


def process_sella_benchmark(
    base_dir: str | Path,
    exclude: List[int] = SELLA_EXCLUDE_RXNS,
):

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    logging.info(f"BEGINNING PROCESSING `sella` BENCHMARK DATASET WITH ROOT DIRECTORY: {base_dir}")

    # Identify TS
    ts_files = glob("*.xyz", root_dir=base_dir / "molecules_fromscratch_renamed")

    reaction_data = list()

    for ts_file in ts_files:
        rxn_id = ts_file.split(".")[0]

        logging.info(f"\tProcessing reaction: {rxn_id}")

        # Exclude some reactions that overlap with other datasets
        if int(rxn_id) in exclude:
            continue

        initial_structure = Molecule.from_file(base_dir / "molecules_fromscratch_renamed" / ts_file)

        # Build molecules from adjacency matrices
        rct_adjacency_matrix = read_adjacency_matrix(base_dir / "molecules_kinbotprod_renamed" / f"{rxn_id}_R.bond")
        pro_adjacency_matrix = read_adjacency_matrix(base_dir / "molecules_kinbotprod_renamed" / f"{rxn_id}_P.bond")

        rct = construct_molecule_from_adjacency_matrix(initial_structure, rct_adjacency_matrix)
        rct_mg = oxygen_edge_extender(MoleculeGraph.with_local_env_strategy(rct, OpenBabelNN()))
        pro = construct_molecule_from_adjacency_matrix(initial_structure, pro_adjacency_matrix)
        pro_mg = oxygen_edge_extender(MoleculeGraph.with_local_env_strategy(pro, OpenBabelNN()))

        # In some cases, there are multiple reactants or products - need to check for disconnected subgraphs
        rct_mgs = rct_mg.get_disconnected_fragments()
        pro_mgs = pro_mg.get_disconnected_fragments()

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"Sella: {rxn_id}"))
    
    return reaction_data
