import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, OpenBabelNN

from gpsts.utils import (
    MAX_BENCHMARK_REACTION_NUMATOMS,
    METAL_EDGE_EXTENDER_PARAMS,
    oxygen_edge_extender,
    prepare_reaction_for_input
)


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


ROST_61_REACTIONS = {
    "R1": {"reactant": ["m1", "m2"], "product": ["m3"]},
    "R2": {"reactant": ["m3"], "product": ["m4"]},
    "R3": {"reactant": ["m1", "m5"], "product": ["m4", "m6"]},
    "R4": {"reactant": ["m7", "m8"], "product": ["m9"]},
    "R5": {"reactant": ["m8", "m9"], "product": ["m10"]},
    "R6": {"reactant": ["m10"], "product": ["m11"]},
    "R7": {"reactant": ["m8", "m11"], "product": ["m12"]},
    "R8": {"reactant": ["m12"], "product": ["m13"]},
    "R9": {"reactant": ["m14"], "product": ["m13"]},
    "R10": {"reactant": ["m14"], "product": ["m15"]},
    "R11": {"reactant": ["m13"], "product": ["m15"]},
    "R12": {"reactant": ["m9", "m16"], "product": ["m8", "m15"]},
    "R13": {"reactant": ["m19", "m20", "m21"], "product": ["m22", "m23"]},
    "R14": {"reactant": ["m24", "m25"], "product": ["m26"]},
    "R15": {"reactant": ["m26", "m27"], "product": ["m28", "m29"]},
    "R16": {"reactant": ["m30", "m31"], "product": ["m32"]},
    "R17": {"reactant": ["m31", "m33"], "product": ["m34"]},
    "R18": {"reactant": ["m31", "m35"], "product": ["m36"]},
    "R19": {"reactant": ["m37", "1/2 m38"], "product": ["m39"]},
    "R20": {"reactant": ["m40", "m41"], "product": ["m42"]},
    "R21": {"reactant": ["m43", "m44"], "product": ["m45", "m46"]},
    "R22": {"reactant": ["m43", "m46"], "product": ["m47"]},
    "R23": {"reactant": ["m49", "m50"], "product": ["m51"]},
    "R24": {"reactant": ["m51", "m52"], "product": ["m53"]},
    "R25": {"reactant": ["m54", "2 m55"], "product": ["2 m41", "2 m17", "m56"]},
    "R26": {"reactant": ["m56", "m57"], "product": ["m18", "m58", "m59"]},
    "R27": {"reactant": ["m48", "m60"], "product": ["m61", "m62"]},
    "R28": {"reactant": ["m48", "m63"], "product": ["m62", "m64"]},
    "R29": {"reactant": ["m66"], "product": ["m65"]},
    "R30": {"reactant": ["m41", "m67"], "product": ["m66"]},
    "R31": {"reactant": ["m31", "m68"], "product": ["m69"]},
    "R32": {"reactant": ["m20", "m70"], "product": ["m41", "m71"]},
    "R33": {"reactant": ["m72"], "product": ["m73", "m74"]},
    "R34": {"reactant": ["m75", "m76"], "product": ["m17", "m77"]},
    "R35": {"reactant": ["m78"], "product": ["m77"]},
    "R36": {"reactant": ["m79", "4 m80"], "product": ["4 m21", "m81"]},
    "R37": {"reactant": ["m82", "m83"], "product": ["m84"]},
    "R38": {"reactant": ["m85", "m86"], "product": ["m87"]},
    "R39": {"reactant": ["m88", "m89"], "product": ["m90", "m91"]},
    "R40": {"reactant": ["m92", "m93"], "product": ["m91", "m94"]},
    "R41": {"reactant": ["m94", "m95"], "product": ["m93", "m96"]},
    "R42": {"reactant": ["m96", "m97"], "product": ["m98", "1/2 m99"]},
    "R43": {"reactant": ["m100", "m101", "2 m17", "m102"], "product": ["m103", "m104", "m105"]},
    "R44": {"reactant": ["m101", "m102", "m106", "m107"], "product": ["m104", "m105", "m108"]},
    "R45": {"reactant": ["m83", "m109"], "product": ["m110"]},
    "R46": {"reactant": ["m111", "m112"], "product": ["m113"]},
    "R47": {"reactant": ["m114", "m115"], "product": ["2 m41", "m107", "m116"]},
    "R48": {"reactant": ["m91", "m117"], "product": ["m118"]},
    "R49": {"reactant": ["m119", "m120"], "product": ["m41", "m121"]},
    "R50": {"reactant": ["m121"], "product": ["m91", "m122"]},
    "R51": {"reactant": ["m122", "m123"], "product": ["m124"]},
    "R52": {"reactant": ["m125"], "product": ["m126"]},
    "R53": {"reactant": ["m120", "m127"], "product": ["m41", "m128"]},
    "R54": {"reactant": ["m129", "m130"], "product": ["m131"]},
    "R55": {"reactant": ["m132"], "product": ["m133"]},
    "R56": {"reactant": ["m134", "m135"], "product": ["m136", "m137"]},
    "R57": {"reactant": ["m138"], "product": ["m139"]},
    "R58": {"reactant": ["m140"], "product": ["m141"]},
    "R59": {"reactant": ["m142", "m143"], "product": ["m144"]},
    "R60": {"reactant": ["m145", "m146"], "product": ["m147"]},
    "R61": {"reactant": ["m148", "m149"], "product": ["m150"]},
}

ROST61_CHARGE_SPIN = {
    1: (0, 2),
    2: (0, 1),
    3: (0, 2),
    4: (0, 2),
    5: (0, 1),
    6: (0, 1),
    7: (1, 6),
    8: (0, 1),
    9: (1, 6),
    10: (1, 6),
    11: (1, 4),
    12: (1, 4),
    13: (1, 4),
    14: (1, 4),
    15: (1, 6),
    16: (0, 1),
    17: (0, 1),
    18: (0, 1),
    19: (1, 2),
    20: (-1, 1),
    21: (0, 1),
    22: (0, 2),
    23: (0, 1),
    24: (0, 4),
    25: (0, 1),
    26: (0, 4),
    27: (0, 1),
    28: (0, 4),
    29: (0, 1),
    30: (1, 2),
    31: (-1, 1),
    32: (0, 2),
    33: (1, 2),
    34: (0, 2),
    35: (1, 2),
    36: (0, 2),
    37: (0, 4),
    38: (0, 1),
    39: (0, 3),
    40: (0, 2),
    41: (-1, 1),
    42: (-1, 2),
    43: (0, 3),
    44: (0, 1),
    45: (0, 2),
    46: (0, 2),
    47: (0, 2),
    48: (0, 1),
    49: (2, 2),
    50: (0, 1),
    51: (2, 2),
    52: (0, 1),
    53: (2, 2),
    54: (0, 4),
    55: (-1, 1),
    56: (0, 4),
    57: (1, 1),
    58: (1, 4),
    59: (0, 1),
    60: (0, 2),
    61: (0, 2),
    62: (0, 1),
    63: (0, 2),
    64: (0, 2),
    65: (0, 2),
    66: (0, 2),
    67: (1, 2),
    68: (0, 2),
    69: (-1, 2),
    70: (0, 2),
    71: (0, 2),
    72: (0, 2),
    73: (0, 2),
    74: (0, 1),
    75: (0, 2),
    76: (0, 1),
    77: (0, 2),
    78: (0, 2),
    79: (2, 2),
    80: (0, 1),
    81: (2, 2),
    82: (0, 2),
    83: (0, 1),
    84: (0, 2),
    85: (1, 2),
    86: (0, 2),
    87: (1, 3),
    88: (0, 3),
    89: (0, 1),
    90: (0, 3),
    91: (0, 1),
    92: (0, 4),
    93: (0, 1),
    94: (0, 4),
    95: (0, 1),
    96: (0, 4),
    97: (0, 1),
    98: (0, 3),
    99: (0, 1),
    100: (-1, 3),
    101: (0, 1),
    102: (1, 1),
    103: (0, 3),
    104: (0, 1),
    105: (0, 1),
    106: (-1, 3),
    107: (0, 1),
    108: (0, 3),
    109: (1, 2),
    110: (1, 2),
    111: (0, 2),
    112: (0, 1),
    113: (0, 2),
    114: (0, 3),
    115: (-2, 1),
    116: (0, 3),
    117: (0, 2),
    118: (0, 2),
    119: (0, 2),
    120: (-1, 1),
    121: (0, 2),
    122: (0, 2),
    123: (0, 1),
    124: (0, 2),
    125: (0, 2),
    126: (0, 2),
    127: (0, 2),
    128: (0, 2),
    129: (1, 2),
    130: (0, 1),
    131: (1, 2),
    132: (0, 2),
    133: (0, 2),
    134: (0, 2),
    135: (0, 1),
    136: (0, 2),
    137: (0, 1),
    138: (2, 2),
    139: (2, 2),
    140: (2, 2),
    141: (2, 2),
    142: (0, 2),
    143: (0, 1),
    144: (0, 2),
    145: (0, 2),
    146: (-1, 1),
    147: (-1, 2),
    148: (0, 2),
    149: (-1, 1),
    150: (-1, 2)
}

ROST61_REACTIONS_TO_PURSUE = [
    1, 2, 3, 6, 8, 9, 10, 11, 14, 15, 21, 24, 30, 33, 35, 39, 40, 41, 45, 46, 48, 50, 52, 54, 55,  # Straightforward
    4, 5, 7, 12, 16, 17, 18, 20, 22, 23, 27, 28, 29, 31, 32, 34, 37, 38, 49, 51, 53, 56, 57, 58, 59, 60, 61  # Hard
]


def process_rost61(
    xyz_dir: str | Path,
    chosen_reactions: List[int] = ROST61_REACTIONS_TO_PURSUE,
    clean: bool = True
) -> List[Dict[str, Any]]:

    reaction_data = list()

    logging.info(f"BEGINNING PROCESSING ROST61 DATASET WITH ROOT DIRECTORY: {xyz_dir}")

    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    logging.info(f"GENERATING MOLECULE GRAPHS")

    # First, construct all molecule graphs
    mgs = dict()
    for subdir in [
        p.resolve() for p in xyz_dir.iterdir() if p.is_dir() 
    ]:
        name = subdir.name
        molid = int(name.replace("m", ""))
        mol = Molecule.from_file(subdir / "mol.xyz")
        charge, spin = ROST61_CHARGE_SPIN[molid]
        mol.set_charge_and_spin(charge, spin_multiplicity=spin)
        mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
        mg = metal_edge_extender(mg, **METAL_EDGE_EXTENDER_PARAMS)
        mg = oxygen_edge_extender(mg)
        mgs[name] = mg

    # After this, should be very straightforward to generate reaction data
    for rxn_id_int in chosen_reactions:
        rxn_id = f"R{rxn_id_int}"

        logging.info(f"\tProcessing reaction: {rxn_id}")

        end_names = ROST_61_REACTIONS[rxn_id]
        rct_mgs = [mgs[x] for x in end_names["reactant"]]
        pro_mgs = [mgs[x] for x in end_names["product"]]

        total_length = sum([len(x.molecule) for x in rct_mgs])

        if total_length > MAX_BENCHMARK_REACTION_NUMATOMS:
            logging.info(f"\t\tSKIPPING: reaction too large")
            continue

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ROST61:{rxn_id}", clean=clean))

    return reaction_data
