import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

from monty.serialization import loadfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, oxygen_edge_extender, OpenBabelNN

from gpsts.utils import (
    prepare_reaction_for_input,
    METAL_EDGE_EXTENDER_PARAMS
)


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


# From 10.1021/acs.jpclett.3c03279
# Format: {name: {"molecule": mol, ...}, ...}
REACTIONS_OX = [
    {"reactants": ["M1"], "products": ["M2"]},
    {"reactants": ["M3"], "products": ["M4"]},
    {"reactants": ["M4"], "products": ["M5"]},
    {"reactants": ["M1"], "products": ["M6"]},
    {"reactants": ["M7"], "products": ["M8"]},
    {"reactants": ["M8"], "products": ["M9"]},
    {"reactants": ["M7"], "products": ["M10"]},
    {"reactants": ["M11"], "products": ["M11"]},
    {"reactants": ["M11"], "products": ["M12"]},
    {"reactants": ["M11"], "products": ["M13"]},
]

# From 10.1021/acsenergylett.2c02351
REACTIONS_PF6 = [
    {"reactants": ["M1"], "products": ["M2"]},
    {"reactants": ["M2-HF"], "products": ["M3"]},
    {"reactants": ["M4"], "products": ["M5"]},
    {"reactants": ["M6"], "products": ["M7"]},
    {"reactants": ["M7-LiF-CO2"], "products": ["LiF", "POF3"]},
    {"reactants": ["M5"], "products": ["M8"]},
    {"reactants": ["M8-LiF"], "products": ["M9"]},
    {"reactants": ["M9"], "products": ["M11"]},
    {"reactants": ["M10"], "products": ["M12"]},
    {"reactants": ["M12-LiF"], "products": ["M13"]},
    {"reactants": ["M14"], "products": ["M15"]},
    {"reactants": ["M15-HF"], "products": ["M16"]},
    {"reactants": ["M17"], "products": ["M18"]},
    {"reactants": ["M19"], "products": ["M20"]},
    {"reactants": ["M21"], "products": ["M22"]},
    {"reactants": ["M23"], "products": ["M24"]},
    {"reactants": ["M24-LiF"], "products": ["M25"]},
    {"reactants": ["PF2OOH", "PF5"], "products": ["M26"]},
    {"reactants": ["M26"], "products": ["M27"]},
    {"reactants": ["M27"], "products": ["M28"]},
    {"reactants": ["LiPF2O2", "PF5"], "products": ["M29"]},
    {"reactants": ["M30"], "products": ["M31"]},
    {"reactants": ["M32"], "products": ["M33"]},
    {"reactants": ["M33"], "products": ["M34"]},
]

# From 10.1021/jacs.3c02222
REACTIONS_MG = [
    {"reactants": ["M2"], "products": ["M3", "M4"]},
    {"reactants": ["M2"], "products": ["M5", "M6"]},
    {"reactants": ["M7"], "products": ["M5", "M9"]},
    {"reactants": ["M8"], "products": ["M9", "M10"]},
    {"reactants": ["M1", "M3"], "products": ["M11", "M12"]},
    {"reactants": ["M1", "M3"], "products": ["M11", "M13"]},
    {"reactants": ["M12"], "products": ["M5", "M14"]},
    {"reactants": ["M13"], "products": ["M10", "M15"]},
    {"reactants": ["M1", "M16"], "products": ["M13", "M17"]},
    {"reactants": ["M2"], "products": ["M18", "M19"]},
    {"reactants": ["M18", "M19"], "products": ["M20", "M21"]},
    {"reactants": ["M18", "M19"], "products": ["M20", "M22"]},
    {"reactants": ["M1", "M18"], "products": ["M20", "M23"]},
    {"reactants": ["M1", "M18"], "products": ["M20", "M24"]},
    {"reactants": ["M18", "M19"], "products": ["M25", "M26"]},
    {"reactants": ["M19", "M27"], "products": ["M25", "M28"]},
    {"reactants": ["M1", "M18"], "products": ["M25", "M29"]},
    {"reactants": ["M1", "M3"], "products": ["M19", "M30"]},
]

# From 10.1039/d2dd00117a
# Format: {reaction: {"reactant": {"molecule": mol, ...}, "product": {"molecule": mol, ...}, 
#                     "transition_state": {"molecule": mol, ...}}}
REACTIONS_HIPRGEN = [
    'li+ec2-_shoulder_ringopen',
    'lvc_ring_open',
    'proton_transfer_lfeo_formation',
    'co2_addition_li+ec2-',
    'carbene_formation_carbonate_elimination',
    'carbene_dimerization',
    'co2_reformation',
    'li+_carbene-_waist',
    'co2-_formation'
]


# From 10.1021/acsenergylett.2c00517
REACTIONS_KMC = [
    {"reactants": ["LiEC_RO1"], "products": ["C2H4", "LiCO3"]},
    {"reactants": ["LEDC_minus_Li_plus"], "products": ["LiEC_RO1", "Li2CO3"]},
    {"reactants": ["LEMC_minus"], "products": ["LiEC_RO1", "OH_minus"]},
    {"reactants": ["LEMC_minus_Li_plus"], "products": ["DLEMC", "H"]},
    {"reactants": ["LEMC_minus_Li_2minus"], "products": ["CO2_minus", "EG_minus_H_minus"]},
    {"reactants": ["tetrahedral_complex"], "products": ["LEDC", "OH_minus"]},
    {"reactants": ["LEDC_minus"], "products": ["LiCO3_minus", "LiEC_RO1"]},
    {"reactants": ["LEDC_minus_Li"], "products": ["LiCO3_minus", "EC"]},
    {"reactants": ["LEMC"], "products": ["OH_minus", "LiEC_plus"]},
    {"reactants": ["LEMC_minus"], "products": ["CO2_minus", "LiEG_minus_H"]},
    {"reactants": ["LEMC_minus"], "products": ["LiCO2", "EG_minus_H_minus"]},
    {"reactants": ["DLEMC_minus"], "products": ["CO2_minus", "(LiOCH2)2"]},
    {"reactants": ["LEDC_minus"], "products": ["LiCO2", "DLEMC_minus_Li"]},
    {"reactants": ["LEDC_minus"], "products": ["CO2_minus", "DLEMC"]},
    {"reactants": ["tetrahedral_complex"], "products": ["HCO3_minus", "DLEMC"]},
    {"reactants": ["tetrahedral_complex"], "products": ["LiHCO3", "DLEMC_minus_Li"]},
    {"reactants": ["HCO3_minus", "DLEMC"], "products": ["LiCO3_minus", "LEMC"]},
    {"reactants": ["LiHCO3", "DLEMC_minus_Li"], "products": ["LiCO3_minus", "LEMC"]},
    {"reactants": ["OH_minus", "LEDC"], "products": ["LiCO3_minus", "LEMC"]},
    {"reactants": ["HCO3_minus", "H"], "products": ["H2", "CO3_minus"]},
    {"reactants": ["LiEC", "OH_minus"], "products": ["LEMC_minus"]},
    {"reactants": ["LEMC_minus", "H"], "products": ["DLEMC_minus_Li", "H2"]},
    {"reactants": ["HCO3_minus", "EC"], "products": ["HLEDC_minus_Li"]},
    {"reactants": ["HCO3_minus", "LiEC_plus"], "products": ["HLEDC"]},
    {"reactants": ["HLEDC_minus"], "products": ["LiEC_RO1", "HCO3_minus"]},
    {"reactants": ["HLEDC", "OH_minus"], "products": ["HCO3_minus", "LEMC"]},
    {"reactants": ["LiEC_RO1", "LiHCO3"], "products": ["HLEDC_minus_Li_plus"]},
    {"reactants": ["tetrahedral_complex_h"], "products": ["HCO3_minus", "LEMC"]},
]

# From 10.1016/j.electacta.2023.143121
REACTIONS_MESOSCALE = [
    # Part 1 - EC
    {"reactants": ["LiEC"], "products": ["LiEC_RO"]},
    {"reactants": ["LiEC_RO_minus"], "products": ["C2H4", "LiCO3_minus"]},
    {"reactants": ["LiCO3_minus", "LiEC_plus"], "products": ["LEDC"]},
    {"reactants": ["LiEC_minus"], "products": ["LiEC_RO_shoulder"]},
    {"reactants": ["LiEC_RO_shoulder"], "products": ["CO", "Li_(OCH2)2_minus"]},
    {"reactants": ["Li_(OCH2)2_minus", "LiEC_plus"], "products": ["PEC_PEO_dimer"]},
    {"reactants": ["Li_(OCH2)2_minus", "EC"], "products": ["PEC_dimer_closed"]},
    {"reactants": ["PEC_dimer_closed"], "products": ["PEC_dimer_open_Li"]},
    # Part 2 - DMC
    {"reactants": ["LiDMC"], "products": ["LMC", "CH3"]},
    {"reactants": ["LiDMC"], "products": ["LiOCH3", "CH3OCO"]},
    {"reactants": ["CH3", "EC"], "products": ["CH4", "EC-H"]},
    {"reactants": ["CH3", "FEC"], "products": ["CH4", "FEC-H"]},
    {"reactants": ["LiFEC-H"], "products": ["LiF", "VC"]},
    # Part 3 - EMC
    {"reactants": ["LiEMC"], "products": ["LEC", "CH3"]},
    {"reactants": ["LiEMC"], "products": ["LMC", "C2H5"]},
    {"reactants": ["LiEMC"], "products": ["LiOCH3", "CH3CH2OCO"]},
    {"reactants": ["LiEMC"], "products": ["LiOCH2CH3", "CH3OCO"]},
    {"reactants": ["C2H5", "EC"], "products": ["C2H6", "EC-H"]},
    {"reactants": ["C2H5", "FEC"], "products": ["C2H6", "FEC-H"]},
    {"reactants": ["LiCH3OCO"], "products": ["CO", "LiOCH3"]},
    {"reactants": ["LiCH3CH2OCO"], "products": ["CO", "LiOCH2CH3"]},
    {"reactants": ["LiOCH3", "EMC"], "products": ["LiTetra112"]},
    {"reactants": ["LiOCH2CH3", "EMC"], "products": ["LiTetra122"]},
    {"reactants": ["LiTetra112"], "products": ["LiOCH2CH3", "DMC"]},
    {"reactants": ["LiTetra122"], "products": ["LiOCH3", "DEC"]},
    # Part 4 - DEC
    {"reactants": ["LiDEC"], "products": ["LEC", "C2H5"]},
    {"reactants": ["LiDEC"], "products": ["LiOCH2CH3", "CH3CH2OCO"]},
    # Part 5 - FEC
    {"reactants": ["LiFEC"], "products": ["LiFEC_RO"]},
    {"reactants": ["LiFEC_RO"], "products": ["LiF", "FEC_RO-LiF"]},
    # {"reactants": ["FEC_RO-LiF"], "products": ["CO2", "OCHCH2"]},  # OCHCH2 missing from data
    {"reactants": ["FEC_RO-LiF_minus"], "products": ["CO2", "OCHCH2_minus"]},
    {"reactants": ["OCHCH2_minus", "FEC"], "products": ["tetrahedral_FEC_minus"]},
    # {"reactants": ["tetrahedral_FEC_minus"], "products": ["FEC_dimer_minus"], "barrier": 0.19},
    {"reactants": ["LiOCH3", "FEC"], "products": ["methoxydioxolanone", "LiF"]},
    {"reactants": ["LiOCH2CH3", "FEC"], "products": ["ethoxydioxolanone", "LiF"]},
    {"reactants": ["LiOCH3", "FEC"], "products": ["moec", "LiF"]},
    {"reactants": ["LiOCH2CH3", "FEC"], "products": ["eoec", "LiF"]},
    # Part 6 - VC
    {"reactants": ["LiVC"], "products": ["LiVC_RO"]},
    {"reactants": ["LiVC_RO_minus"], "products": ["C2H2", "LiCO3_minus"]},
    # Part 7 - PC
    {"reactants": ["LiPC"], "products": ["LiPC_RO"]},
    {"reactants": ["LiPC_RO_minus"], "products": ["C3H6", "LiCO3_minus"]},
    # Part 8 - BC
    {"reactants": ["LiBC"], "products": ["LiBC_RO"]},
    {"reactants": ["LiBC_RO_minus"], "products": ["C4H8", "LiCO3_minus"]},
]


# From 10.1021/acs.jpcc.2c06653
# These are basically all huge and costly - for now, not including
# REACTIONS_BORATE_ALUMINATE = [
#     # Ca_al_hfip_AlObreak
#     (("Ca_al_hfip_AlObreak", "struct1"), ("Ca_al_hfip_AlObreak", "struct2")),
#     (("Ca_al_hfip_AlObreak", "struct2"), ("Ca_al_hfip_AlObreak", "struct3")),
#     (("Ca_al_hfip_AlObreak", "struct3"), ("Ca_al_hfip_AlObreak", "struct4")),
#     # Ca_al_hfip_CObreak
#     (("Ca_al_hfip_CObreak", "struct1"), ("Ca_al_hfip_CObreak", "struct2")),
#     (("Ca_al_hfip_CObreak", "struct2"), ("Ca_al_hfip_CObreak", "struct3")),
#     # Ca_al_hftb_AlObreak
#     (("Ca_al_hftb_AlObreak", "struct1"), ("Ca_al_hftb_AlObreak", "struct2")),
#     (("Ca_al_hftb_AlObreak", "struct2"), ("Ca_al_hftb_AlObreak", "struct3")),
#     (("Ca_al_hftb_AlObreak", "struct3"), ("Ca_al_hftb_AlObreak", "struct4")),
#     # Ca_al_hftb_CFbreak
#     (("Ca_al_hftb_AlObreak", "struct1"), ("Ca_al_hftb_CFbreak", "struct2")),
#     # Ca_al_hftb_CObreak
#     (("Ca_al_hftb_CObreak", "struct1"), ("Ca_al_hftb_CObreak", "struct2")),
#     # Ca_al_pftb_AlObreak
#     (("Ca_al_pftb_AlObreak", "struct1"), ("Ca_al_pftb_AlObreak", "struct2")),
#     (("Ca_al_pftb_AlObreak", "struct2"), ("Ca_al_pftb_AlObreak", "struct3")),
#     (("Ca_al_pftb_AlObreak", "struct3"), ("Ca_al_pftb_AlObreak", "struct4")),
#     # Ca_al_pftb_CFbreak
#     (("Ca_al_pftb_CFbreak", "struct1"), ("Ca_al_pftb_CFbreak", "struct2")),
#     # Ca_al_pftb_CObreak
#     (("Ca_al_pftb_CObreak", "struct1"), ("Ca_al_pftb_CObreak", "struct2")),
#     # Ca_al_tfip_AlObreak
#     (("Ca_al_tfip_AlObreak", "struct1"), ("Ca_al_tfip_AlObreak", "struct2")),
#     (("Ca_al_tfip_AlObreak", "struct2"), ("Ca_al_tfip_AlObreak", "struct3")),
#     (("Ca_al_tfip_AlObreak", "struct3"), ("Ca_al_tfip_AlObreak", "struct4")),
#     # Ca_al_tfip_CObreak
#     (("Ca_al_tfip_CObreak", "struct1"), ("Ca_al_tfip_CObreak", "struct2")),
#     (("Ca_al_tfip_CObreak", "struct2"), ("Ca_al_tfip_CObreak", "struct3")),
#     # Ca_al_tftb_AlObreak
#     (("Ca_al_tftb_AlObreak", "struct1"), ("Ca_al_tftb_AlObreak", "struct2")),
#     (("Ca_al_tftb_AlObreak", "struct2"), ("Ca_al_tftb_AlObreak", "struct3")),
#     (("Ca_al_tftb_AlObreak", "struct3"), ("Ca_al_tftb_AlObreak", "struct4")),
#     # Ca_al_tftb_CObreak
#     (("Ca_al_tftb_CObreak", "struct1"), ("Ca_al_tftb_CObreak", "struct2")),
#     (("Ca_al_tftb_CObreak", "struct2"), ("Ca_al_tftb_CObreak", "struct3")),
#     # Ca_b_hfip_BObreak
#     (("Ca_b_hfip_BObreak", "struct1"), ("Ca_b_hfip_BObreak", "struct2")),
#     # Ca_b_hfip_CObreak
#     (("Ca_b_hfip_CObreak", "struct1"), ("Ca_b_hfip_CObreak", "struct2")),
#     (("Ca_b_hfip_CObreak", "struct2"), ("Ca_b_hfip_CObreak", "struct3")),
#     # Ca_b_hftb_BObreak
#     (("Ca_b_hftb_BObreak", "struct1"), ("Ca_b_hftb_BObreak", "struct2")),
#     # Ca_b_hftb_CFbreak
#     (("Ca_b_hftb_CFbreak", "struct1"), ("Ca_b_hftb_CFbreak", "struct2")),
#     # Ca_b_hftb_CObreak
#     (("Ca_b_hftb_CObreak", "struct1"), ("Ca_b_hftb_CObreak", "struct2")),
#     # Ca_b_pftb_BObreak
#     (("Ca_b_pftb_BObreak", "struct1"), ("Ca_b_pftb_BObreak", "struct2")),
#     (("Ca_b_pftb_BObreak", "struct2"), ("Ca_b_pftb_BObreak", "struct3")),
#     # Ca_b_pftb_CFbreak
#     (("Ca_b_pftb_CFbreak", "struct1"), ("Ca_b_pftb_CFbreak", "struct2")),
#     # Ca_b_pftb_CObreak
#     (("Ca_b_pftb_CObreak", "struct1"), ("Ca_b_pftb_CObreak", "struct2")),
#     # Ca_b_tfip_BObreak
#     (("Ca_b_tfip_BObreak", "struct1"), ("Ca_b_tfip_BObreak", "struct2")),
#     (("Ca_b_tfip_BObreak", "struct2"), ("Ca_b_tfip_BObreak", "struct3")),
#     # Ca_b_tfip_CObreak
#     (("Ca_b_tfip_CObreak", "struct1"), ("Ca_b_tfip_CObreak", "struct2")),
#     (("Ca_b_tfip_CObreak", "struct2"), ("Ca_b_tfip_CObreak", "struct3")),
#     # Ca_b_tftb_BObreak
#     (("Ca_b_tftb_BObreak", "struct1"), ("Ca_b_tftb_BObreak", "struct2")),
#     (("Ca_b_tftb_BObreak", "struct2"), ("Ca_b_tftb_BObreak", "struct3")),
#     (("Ca_b_tftb_BObreak", "struct3"), ("Ca_b_tftb_BObreak", "struct4")),
#     # Ca_b_tftb_CObreak
#     (("Ca_b_tftb_CObreak", "struct1"), ("Ca_b_tftb_CObreak", "struct2")),
#     (("Ca_b_tftb_CObreak", "struct2"), ("Ca_b_tftb_CObreak", "struct3")),
#     (("Ca_b_tftb_CObreak", "struct3"), ("Ca_b_tftb_CObreak", "struct4")),
#     # Mg_al_hfip_AlObreak
#     (("Mg_al_hfip_AlObreak", "struct1"), ("Mg_al_hfip_AlObreak", "struct2")),
#     (("Mg_al_hfip_AlObreak", "struct2"), ("Mg_al_hfip_AlObreak", "struct3")),
#     (("Mg_al_hfip_AlObreak", "struct3"), ("Mg_al_hfip_AlObreak", "struct4")),
#     # Mg_al_hfip_CObreak
#     (("Mg_al_hfip_CObreak", "struct1"), ("Mg_al_hfip_CObreak", "struct2")),
#     (("Mg_al_hfip_CObreak", "struct2"), ("Mg_al_hfip_CObreak", "struct3")),
#     # Mg_al_hftb_AlObreak
#     (("Mg_al_hftb_AlObreak", "struct1"), ("Mg_al_hftb_AlObreak", "struct2")),
#     (("Mg_al_hftb_AlObreak", "struct2"), ("Mg_al_hftb_AlObreak", "struct3")),
#     # Mg_al_hftb_CObreak
#     (("Mg_al_hftb_CObreak", "struct1"), ("Mg_al_hftb_CObreak", "struct2")),
#     (("Mg_al_hftb_CObreak", "struct2"), ("Mg_al_hftb_CObreak", "struct3")),
#     # Mg_al_pftb_AlObreak
#     (("Mg_al_pftb_AlObreak", "struct1"), ("Mg_al_pftb_AlObreak", "struct2")),
#     (("Mg_al_pftb_AlObreak", "struct2"), ("Mg_al_pftb_AlObreak", "struct3")),
#     # Mg_al_pftb_CObreak
#     (("Mg_al_pftb_AlObreak", "struct1"), ("Mg_al_pftb_CObreak", "struct2")),
#     (("Mg_al_pftb_CObreak", "struct2"), ("Mg_al_pftb_CObreak", "struct3")),
#     # Mg_al_tfip_AlObreak
#     (("Mg_al_tfip_AlObreak", "struct1"), ("Mg_al_tfip_AlObreak", "struct2")),
#     (("Mg_al_tfip_AlObreak", "struct2"), ("Mg_al_tfip_AlObreak", "struct3")),
#     (("Mg_al_tfip_AlObreak", "struct3"), ("Mg_al_tfip_AlObreak", "struct4")),
#     # Mg_al_tfip_CObreak
#     (("Mg_al_tfip_AlObreak", "struct1"), ("Mg_al_tfip_CObreak", "struct2")),
#     (("Mg_al_tfip_CObreak", "struct2"), ("Mg_al_tfip_CObreak", "struct3")),
#     # Mg_al_tftb_AlObreak
#     (("Mg_al_tftb_AlObreak", "struct1"), ("Mg_al_tftb_AlObreak", "struct2")),
#     (("Mg_al_tftb_AlObreak", "struct2"), ("Mg_al_tftb_AlObreak", "struct3")),
    
#     # Mg B HFIP - seems struct1 is missing from Mg_b_hfip_BObreak and Mg_b_hfip_CObreak
#     # Could use Mg_b_hfip_0.xyz from ox_red_potential_CIP_wb97xd:
#     # For now, ignoring

#     # Mg_b_hftb_BObreak
#     (("Mg_b_hftb_BObreak", "struct1"), ("Mg_b_hftb_BObreak", "struct2")),
#     (("Mg_b_hftb_BObreak", "struct2"), ("Mg_b_hftb_BObreak", "struct3")),
#     # Mg_b_hftb_CFbreak
#     (("Mg_b_hftb_CFbreak", "struct1"), ("Mg_b_hftb_CFbreak", "struct2")),
#     # Mg_b_hftb_CObreak
#     (("Mg_b_hftb_CObreak", "struct1"), ("Mg_b_hftb_CObreak", "struct2")),
#     # Mg_b_pftb_BObreak
#     (("Mg_b_pftb_BObreak", "struct1"), ("Mg_b_pftb_BObreak", "struct2")),
#     (("Mg_b_pftb_BObreak", "struct2"), ("Mg_b_pftb_BObreak", "struct3")),
#     # Mg_b_pftb_CFbreak
#     (("Mg_b_pftb_CFbreak", "struct1"), ("Mg_b_pftb_CFbreak", "struct2")),
#     # Mg_b_pftb_CObreak
#     (("Mg_b_pftb_CObreak", "struct1"), ("Mg_b_pftb_CObreak", "struct2")),
#     (("Mg_b_pftb_CObreak", "struct2"), ("Mg_b_pftb_CObreak", "struct3")),
#     (("Mg_b_pftb_CObreak", "struct3"), ("Mg_b_pftb_CObreak", "struct4")),
#     # Mg_b_tfip_BObreak
#     (("Mg_b_tfip_BObreak", "struct1"), ("Mg_b_tfip_BObreak", "struct2")),
#     (("Mg_b_tfip_BObreak", "struct2"), ("Mg_b_tfip_BObreak", "struct3")),   
#     # Mg_b_tfip_CObreak
#     (("Mg_b_tfip_BObreak", "struct1"), ("Mg_b_tfip_CObreak", "struct2")),
#     (("Mg_b_tfip_CObreak", "struct2"), ("Mg_b_tfip_CObreak", "struct3")),
#     # Mg_b_tftb_BObreak
#     (("Mg_b_tftb_BObreak", "struct1"), ("Mg_b_tftb_BObreak", "struct2")),
#     (("Mg_b_tftb_BObreak", "struct2"), ("Mg_b_tftb_BObreak", "struct3")),
#     (("Mg_b_tftb_BObreak", "struct3"), ("Mg_b_tftb_BObreak", "struct4")),
#     # Mg_b_tftb_CObreak
#     (("Mg_b_tftb_CObreak", "struct1"), ("Mg_b_tftb_CObreak", "struct2")),
#     (("Mg_b_tftb_CObreak", "struct2"), ("Mg_b_tftb_CObreak", "struct3")),
#     (("Mg_b_tftb_CObreak", "struct3"), ("Mg_b_tftb_CObreak", "struct4")),
# ]


# From 10.1021/jacs.1c05807
REACTIONS_LEDC_LEMC = [
    # Figure 4a
    (("figure4a", "struct1"), ("figure4a", "struct2")),
    # (("figure4a", "struct2"), ("figure4a", "struct3")),  # Not actually balanced
    (("figure4a", "struct3"), ("figure4a", "struct4")),
    (("figure4a", "struct4"), ("figure4a", "struct5")),
    (("figure4a", "struct3"), ("figure4a", "struct6")),
    (("figure4a", "struct7"), ("figure4a", "struct8")),
    # Figure 4b
    (("figure4b", "struct1"), ("figure4b", "struct2")),
    (("figure4b", "struct3"), ("figure4b", "struct4")),
    (("figure4b", "struct4"), ("figure4b", "struct5")),
    # Figure 6a
    (("figure6a", "struct1"), ("figure6a", "struct2")),
    (("figure6a", "struct2"), ("figure6a", "struct3")),
    (("figure6a", "struct3"), ("figure6a", "struct4")),
    (("figure6a", "struct5"), ("figure6a", "struct6")),
    # Figure 6b
    (("figure6b", "struct1"), ("figure6b", "struct2")),
    (("figure6b", "struct2"), ("figure6b", "struct3")),
    (("figure6b", "struct4"), ("figure6b", "struct5")),
    (("figure6b", "struct6"), ("figure6b", "struct7")),
    # Figure 8
    (("figure8", "struct1"), ("figure8", "struct2")),
    (("figure8", "struct2"), ("figure8", "struct3")),
    (("figure8", "struct4"), ("figure8", "struct5")),
    # Figure S2
    (("figureS2", "struct1"), ("figureS2", "struct2")),
    # Figure S3a
    (("figureS3a", "struct1"), ("figureS3a", "struct2")),
    # Figure S3b
    (("figureS3b", "struct1"), ("figureS3b", "struct2")),
    (("figureS3b", "struct3"), ("figureS3b", "struct4")),
    # Figure S4
    (("figureS4", "struct1"), ("figureS4", "struct2")),
    # Figure S5a
    (("figureS5a", "struct1"), ("figureS5a", "struct2")),
    (("figureS5a", "struct2"), ("figureS5a", "struct3")),
    (("figureS5a", "struct3"), ("figureS5a", "struct4")),
    # Figure S5b
    (("figureS5b", "struct1"), ("figureS5b", "struct2")),
    (("figureS5b", "struct3"), ("figureS5b", "struct4")),
    # Figure S5c
    (("figureS5c", "struct1"), ("figureS5c", "struct2")),
    (("figureS5c", "struct3"), ("figureS5c", "struct4")),
    (("figureS5c", "struct5"), ("figureS5c", "struct6")),
    (("figureS5c", "struct7"), ("figureS5c", "struct8")),
    (("figureS5c", "struct8"), ("figureS5c", "struct9")),
]

CHARGES_LEDC_LEMC = {
    "figure4a": 0,
    "figure4b": -1,
    "figure6a": 0,
    "figure6b": -1,
    "figure8": 0,
    "figureS2": 0,
    "figureS3a": 0,
    "figureS3b": 0,
    "figureS4": 0,
    "figureS5a": 0,
    "figureS5b": 0,
    "figureS5c": -1,
}


def process_ox(
    data_json_path: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING EC OXIDATION DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    mol_data = loadfn(data_json_path)

    logging.info(f"\tGENERATING MOLECULE GRAPHS")

    # First, construct all molecule graphs
    mgs = dict()
    for name, mol_datum in mol_data.items():
        mg = MoleculeGraph.with_local_env_strategy(mol_datum["molecule"], OpenBabelNN())
        mgs[name] = mg

    # Generate reaction data
    for reaction in REACTIONS_OX:
        rxn_id = "+".join(reaction["reactants"]) + "->" + "+".join(reaction["products"])

        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        # For this dataset, automatically assigning charges to sub-molecule graphs is nontrivial
        # For now, just don't break them down
        # This will make this subset of the data somewhat easier - no complexes need to be made
        rct_mgs = [mgs[x] for x in reaction["reactants"]]
        pro_mgs = [mgs[x] for x in reaction["products"]]

        # Confirm that reactants and products have same elements
        if rct_mgs[0].molecule.species == pro_mgs[0].molecule.species:
            # Reactants and products should always have the same atom mapping
            mapping = {(0, i): (0, i) for i in range(len(rct_mgs[0].molecule))}

            reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, mapping=mapping, label=f"ELYTE-TS:(OX){rxn_id}", clean=clean))
        else:
            reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(OX){rxn_id}", clean=clean))

    return reaction_data


def process_pf6(
    data_json_path: str | Path,
    clean: bool = True,
) -> List[Dict[str, Any]]:
    
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING PF6 DECOMPOSITION DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    mol_data = loadfn(data_json_path)

    logging.info(f"\tGENERATING MOLECULE GRAPHS")

    # First, construct all molecule graphs
    mgs = dict()
    for name, mol_datum in mol_data.items():
        mg = MoleculeGraph.with_local_env_strategy(mol_datum["molecule"], OpenBabelNN())
        mgs[name] = mg

    # Generate reaction data
    for reaction in REACTIONS_PF6:
        rxn_id = "+".join(reaction["reactants"]) + "->" + "+".join(reaction["products"])

        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        rct_mgs = list()
        pro_mgs = list()

        # Separate disconnected fragments
        for rct in reaction["reactants"]:
            for sub_mg in mgs[rct].get_disconnected_fragments():
                rct_mgs.append(sub_mg)

        for pro in reaction["products"]:
            for sub_mg in mgs[pro].get_disconnected_fragments():
                pro_mgs.append(sub_mg)

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(PF6){rxn_id}", clean=clean))

    return reaction_data


def process_mg(
    data_json_path: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING MG G2 DECOMPOSITION DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    mol_data = loadfn(data_json_path)

    logging.info(f"\tGENERATING MOLECULE GRAPHS")

    # First, construct all molecule graphs
    mgs = dict()
    for name, mol_datum in mol_data.items():
        mg = MoleculeGraph.with_local_env_strategy(mol_datum["molecule"], OpenBabelNN())
        mgs[name] = mg

    # Generate reaction data
    for reaction in REACTIONS_MG:
        rxn_id = "+".join(reaction["reactants"]) + "->" + "+".join(reaction["products"])

        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        rct_mgs = [mgs[x] for x in reaction["reactants"]]
        pro_mgs = [mgs[x] for x in reaction["products"]]

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(MG){rxn_id}", clean=clean))

    return reaction_data


def process_hiprgen(
    data_json_path: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING HIPRGEN DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    rxn_data = loadfn(data_json_path)

    logging.info(f"\tGENERATING MOLECULE GRAPHS")

    for rxn_id in REACTIONS_HIPRGEN:
        reaction = rxn_data[rxn_id]

        logging.info(f"\t\tProcessing reaction: {rxn_id}")
        
        # Construct molecule graphs
        # For this dataset, automatically assigning charges to sub-molecule graphs is nontrivial
        # For now, just don't break them down (if not already broken down)
        # This will make this subset of the data somewhat easier - no complexes need to be made
        rct_mgs = list()
        pro_mgs = list()
        for mol in reaction["reactant"]["molecule"]:
            rct_mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
            rct_mg = metal_edge_extender(rct_mg, **METAL_EDGE_EXTENDER_PARAMS)
            rct_mgs.append(rct_mg)

        # Special case - dimerization with two identical reactants
        if rxn_id == "carbene_dimerization":
            rct_mgs.append(rct_mgs[0])
        
        for mol in reaction["product"]["molecule"]:
            pro_mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
            pro_mg = metal_edge_extender(pro_mg, **METAL_EDGE_EXTENDER_PARAMS)
            pro_mgs.append(pro_mg)

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(HIPRGEN){rxn_id}", clean=clean))

    return reaction_data


def process_kmc(
    data_json_path: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING KMC DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    mol_data = loadfn(data_json_path)

    # Grab molecule graphs from this JSON, and reorganize by name
    mgs = dict()
    for mol in mol_data:
        mgs[mol["name"]] = mol["molecule_graph"]

    # Generate reaction data
    for reaction in REACTIONS_KMC:
        rxn_id = "+".join(reaction["reactants"]) + "->" + "+".join(reaction["products"])

        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        rct_mgs = [mgs[x] for x in reaction["reactants"]]
        pro_mgs = [mgs[x] for x in reaction["products"]]

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(KMC){rxn_id}", clean=clean))

    return reaction_data


def process_mesoscale(
    data_json_path: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:
    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING MESOSCALE DATASET FROM: {data_json_path}")

    if isinstance(data_json_path, str):
        data_json_path = Path(data_json_path)

    mol_data = loadfn(data_json_path)

    mgs = dict()
    for name, data in mol_data.items():
        mgs[name] = data["molecule_graph"]

    # Generate reaction data
    for reaction in REACTIONS_MESOSCALE:
        rxn_id = "+".join(reaction["reactants"]) + "->" + "+".join(reaction["products"])

        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        rct_mgs = [mgs[x] for x in reaction["reactants"]]
        pro_mgs = [mgs[x] for x in reaction["products"]]

        reaction_data.append(prepare_reaction_for_input(rct_mgs, pro_mgs, label=f"ELYTE-TS:(MESOSCALE){rxn_id}", clean=clean))

    return reaction_data


# def process_borate_aluminate(
#     base_dir: str | Path
# ) -> List[Dict[str, Any]]:

#     reaction_data = list()

#     logging.info(f"\tBEGINNING PROCESSING BORATE/ALUMINATE DATASET FROM: {base_dir}")

#     if isinstance(base_dir, str):
#         base_dir = Path(base_dir)

#     # This is inefficient, but only slightly
#     # Most structures are not used more than once

#     for reaction in REACTIONS_BORATE_ALUMINATE:
#         reactant = reaction[0]
#         product = reaction[1]

#         rxn_id = f"{reactant[0]}_{reactant[1]}->{product[0]}_{product[1]}"
#         logging.info(f"\t\tProcessing reaction: {rxn_id}")

#         # All complexes are neutral, so there's no need to modify charge/spin
#         rct_mol = Molecule.from_file(base_dir / reactant[0] / f"{reactant[1]}.xyz")
#         pro_mol = Molecule.from_file(base_dir / product[0] / f"{product[1]}.xyz")

#         # This is another case where there are disconnected fragments in some reactants/products
#         # As assigning charges to sub-molecule graphs is nontrivial, for now we leave complexes whole
#         # This will make this subset of the data somewhat easier - no complexes need to be made

#         rct_mg = MoleculeGraph.with_local_env_strategy(rct_mol, OpenBabelNN())
#         rct_mg = metal_edge_extender(rct_mg, **METAL_EDGE_EXTENDER_PARAMS)
#         pro_mg = MoleculeGraph.with_local_env_strategy(pro_mol, OpenBabelNN())
#         pro_mg = metal_edge_extender(pro_mg, **METAL_EDGE_EXTENDER_PARAMS)

#         # No mapping should be needed
#         if rct_mol.species == pro_mol.species:
#             mapping = {(0, i): (0, i) for i in range(len(rct_mol))}

#             reaction_data.append(
#                 prepare_reaction_for_input(
#                     [rct_mg],
#                     [pro_mg],
#                     mapping=mapping,
#                     label=f"ELYTE-TS:(BORATE/ALUMINATE){rxn_id}"
#                 )
#             )
#         else:
#             reaction_data.append(
#                 prepare_reaction_for_input(
#                     [rct_mg],
#                     [pro_mg],
#                     label=f"ELYTE-TS:(BORATE/ALUMINATE){rxn_id}"
#                 )
#             )

#     return reaction_data


def process_ledc_lemc(
    base_dir: str | Path,
    clean: bool = True
) -> List[Dict[str, Any]]:

    reaction_data = list()

    logging.info(f"\tBEGINNING PROCESSING LEDC/LEMC DATASET FROM: {base_dir}")

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    for reaction in REACTIONS_LEDC_LEMC:
        reactant = reaction[0]
        product = reaction[1]

        rxn_id = f"{reactant[0]}_{reactant[1]}->{product[0]}_{product[1]}"
        logging.info(f"\t\tProcessing reaction: {rxn_id}")

        charge = CHARGES_LEDC_LEMC[reactant[0]]

        rct_mol = Molecule.from_file(base_dir / reactant[0] / f"{reactant[1]}.xyz")
        rct_mol.set_charge_and_spin(charge)
        pro_mol = Molecule.from_file(base_dir / product[0] / f"{product[1]}.xyz")
        pro_mol.set_charge_and_spin(charge)

        # This is another case where there are disconnected fragments in some reactants/products
        # As assigning charges to sub-molecule graphs is nontrivial, for now we leave complexes whole
        # This will make this subset of the data somewhat easier - no complexes need to be made

        rct_mg = MoleculeGraph.with_local_env_strategy(rct_mol, OpenBabelNN())
        rct_mg = metal_edge_extender(rct_mg, **METAL_EDGE_EXTENDER_PARAMS)
        pro_mg = MoleculeGraph.with_local_env_strategy(pro_mol, OpenBabelNN())
        pro_mg = metal_edge_extender(pro_mg, **METAL_EDGE_EXTENDER_PARAMS)

        # No mapping should be needed
        if rct_mol.species == pro_mol.species:
            mapping = {(0, i): (0, i) for i in range(len(rct_mol))}

            reaction_data.append(
                prepare_reaction_for_input(
                    [rct_mg],
                    [pro_mg],
                    mapping=mapping,
                    label=f"ELYTE-TS:(LEDC/LEMC){rxn_id}",
                    clean=clean
                )
            )
        else:
            reaction_data.append(
                prepare_reaction_for_input(
                    [rct_mg],
                    [pro_mg],
                    label=f"ELYTE-TS:(LEDC/LEMC){rxn_id}",
                    clean=clean
                )
            )

    return reaction_data


def process_elyte_ts(
    ox_path: str | Path,
    pf6_path: str | Path,
    mg_path: str | Path,
    hiprgen_path: str | Path,
    kmc_path: str | Path,
    mesoscale_path: str | Path,
    # borate_aluminate_path: str | Path,
    ledc_lemc_path: str | Path,
    clean: bool = True
):
    
    reaction_data = list()
    reaction_data += process_ox(ox_path, clean=clean)
    reaction_data += process_pf6(pf6_path, clean=clean)
    reaction_data += process_mg(mg_path, clean=clean)
    reaction_data += process_hiprgen(hiprgen_path, clean=clean)
    reaction_data += process_kmc(kmc_path, clean=clean)
    reaction_data += process_mesoscale(mesoscale_path, clean=clean)
    # reaction_data += process_borate_aluminate(borate_aluminate_path)
    reaction_data += process_ledc_lemc(ledc_lemc_path, clean=clean)

    return reaction_data