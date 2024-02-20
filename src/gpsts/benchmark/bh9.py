import os
from glob import glob
from pathlib import Path

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, oxygen_edge_extender, OpenBabelNN

from gpsts.utils import (
    METAL_EDGE_EXTENDER_PARAMS,
    prepare_reaction_for_input
)


def mg_from_bh9(path: Path) -> MoleculeGraph:
    with open(path) as file_obj:
        lines = file_obj.readlines()
        charge, spin = lines[1].strip().split()
        charge = int(charge)
        spin = int(spin)

    mol = Molecule.from_file(path)
    mol.set_charge_and_spin(charge, spin_multiplicity=spin)
    mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    mg = metal_edge_extender(mg, **METAL_EDGE_EXTENDER_PARAMS)
    mg = oxygen_edge_extender(mg)

    return mg


def process_bh9(
    xyz_dir: str | Path
):
    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    # Identify TS
    # We only do this so that we can effectively grab the reaction reactants and products
    ts_files = glob("*TS.xyz", root_dir=xyz_dir)

    reaction_data = list()

    for ts_file in ts_files:
        rxn_id = ts_file.split("TS")[0]

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

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs))

    return reaction_data