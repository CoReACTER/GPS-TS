import logging
from glob import glob
from pathlib import Path

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import oxygen_edge_extender, OpenBabelNN

from gpsts.utils import prepare_reaction_for_input


MOBH_35_REACTIONS = {
    "R1": {"reactant": ["r1+"], "product": ["p1+"]},
    "R2": {"reactant": ["r2+"], "product": ["p2+"]},
    "R3": {"reactant": ["r3"], "product": ["p3"]},
    "R4": {"reactant": ["r4"], "product": ["p4"]},
    "R5": {"reactant": ["r5"], "product": ["p5"]},
    "R6": {"reactant": ["r6"], "product": ["p6_r7"]},
    "R7": {"reactant": ["p6_r7"], "product": ["p7"]},
    "R8": {"reactant": ["r8"], "product": ["p8_r9"]},
    "R9": {"reactant": ["p8_r9"], "product": ["p9"]},
    "R10": {"reactant": ["r10"], "product": ["p10", "CO"]},
    "R11": {"reactant": ["r11"], "product": ["p11"]},
    "R12": {"reactant": ["r12"], "product": ["p12"]},
    "R13": {"reactant": ["r13"], "product": ["p13"]},
    "R14": {"reactant": ["r14"], "product": ["p14"]},
    "R15": {"reactant": ["r15"], "product": ["p15"]},
    "R16": {"reactant": ["r16"], "product": ["p16"]},
    "R17": {"reactant": ["r17"], "product": ["p17", "PEt3"]},
    "R18": {"reactant": ["r18"], "product": ["p18", "PEt3"]},
    "R19": {"reactant": ["r19"], "product": ["p19", "PEt3"]},
    "R20": {"reactant": ["r20"], "product": ["p20", "PEt3"]},
    "R21": {"reactant": ["r21"], "product": ["p21"]},
    "R22": {"reactant": ["r22"], "product": ["p22"]},
    "R23": {"reactant": ["r23"], "product": ["p23"]},
    "R24": {"reactant": ["r24"], "product": ["p24"]},
    "R25": {"reactant": ["r25"], "product": ["p25"]},
    "R26": {"reactant": ["r26"], "product": ["p26"]},
    "R27": {"reactant": ["r27"], "product": ["p27"]},
    "R28": {"reactant": ["r28"], "product": ["p28"]},
    "R29": {"reactant": ["r29"], "product": ["p29"]},
    "R30": {"reactant": ["r30_r31"], "product": ["p30"]},
    "R31": {"reactant": ["r30_r31"], "product": ["p31"]},
    "R32": {"reactant": ["r32+"], "product": ["p32+"]},
    "R33": {"reactant": ["r33+"], "product": ["p33+"]},
    "R34": {"reactant": ["r34"], "product": ["p34_r35"]},
    "R35": {"reactant": ["p34_r35"], "product": ["p35", "CH4"]}
}


def process_mobh35(
    xyz_dir: str | Path
) -> List[Dict[str, Any]]:

    reaction_data = list()

    logging.info(f"BEGINNING PROCESSING MOBH35 DATASET WITH ROOT DIRECTORY: {xyz_dir}")

    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    # First, grab all non-TS *.xyz files
    structure_paths = [
        p.resolve() for p in xyz_dir.iterdir()
        if p.name.endswith(".xyz")
        and not p.name.startswith("ts")
    ]

    logging.info(f"GENERATING MOLECULE GRAPHS")

    # Construct molecule graphs, which we'll then use to build reactions
    mgs = dict()
    for sp in structure_paths:
        name = sp.name.split(".")[0]
        mol = Molecule.from_file(sp)
        if "+" in name:
            mol.set_charge_and_spin(1)
        mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
        mg = metal_edge_extender(mg, **METAL_EDGE_EXTENDER_PARAMS)
        mg = oxygen_edge_extender(mg)
        mgs[name] = mg

    # After this, should be very straightforward to generate reaction data
    for rxn_id, end_names in MOBH_35_REACTIONS.items():
        logging.info(f"\tProcessing reaction: {rxn_id}")
        rct_mgs = [mgs[x] for x in end_names["reactant"]]
        pro_mgs = [mgs[x] for x in end_names["product"]]

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"MOBH35:{rxn_id}"))

    return reaction_data
