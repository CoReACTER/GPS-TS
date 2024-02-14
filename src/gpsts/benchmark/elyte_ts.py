# Format: {name: {"molecule": mol, ...}, ...}
reactions_ox = [
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

# Format: {reaction: {"reactant": {"molecule": mol, ...}, "product": {"molecule": mol, ...}, "transition_state": {"molecule": mol, ...}}}
reactions_hiprgen = [
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

reactions_kmc = [
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

reactions_mesoscale = [
    # Part 1 - EC
    {"reactants": ["LiEC"], "products": ["LiEC_RO"], "barrier": 0.40},
    {"reactants": ["LiEC_RO_minus"], "products": ["C2H4", "LiCO3_minus"], "barrier": 0.13},
    {"reactants": ["LiCO3_minus", "LiEC_plus"], "products": ["LEDC"], "barrier": 0.32},
    {"reactants": ["LiEC_minus"], "products": ["LiEC_RO_shoulder"], "barrier": 0.09},
    {"reactants": ["LiEC_RO_shoulder"], "products": ["CO", "Li_(OCH2)2_minus"], "barrier": 0.09},
    {"reactants": ["Li_(OCH2)2_minus", "LiEC_plus"], "products": ["PEC_PEO_dimer"], "barrier": 0.27},
    {"reactants": ["Li_(OCH2)2_minus", "EC"], "products": ["PEC_dimer_closed"], "barrier": 0.53},
    {"reactants": ["PEC_dimer_closed"], "products": ["PEC_dimer_open_Li"], "barrier": 0.04},
    # Part 2 - DMC
    {"reactants": ["LiDMC"], "products": ["LMC", "CH3"], "barrier": 0.48},
    {"reactants": ["LiDMC"], "products": ["LiOCH3", "CH3OCO"], "barrier": 0.56},
    {"reactants": ["CH3", "EC"], "products": ["CH4", "EC-H"], "barrier": 0.89},
    {"reactants": ["CH3", "FEC"], "products": ["CH4", "FEC-H"], "barrier": 0.53},
    {"reactants": ["LiFEC-H"], "products": ["LiF", "VC"], "barrier": 0.00},
    # Part 3 - EMC
    {"reactants": ["LiEMC"], "products": ["LEC", "CH3"], "barrier": 0.66},
    {"reactants": ["LiEMC"], "products": ["LMC", "C2H5"], "barrier": 0.70},
    {"reactants": ["LiEMC"], "products": ["LiOCH3", "CH3CH2OCO"], "barrier": 0.44},
    {"reactants": ["LiEMC"], "products": ["LiOCH2CH3", "CH3OCO"], "barrier": 0.48},
    {"reactants": ["C2H5", "EC"], "products": ["C2H6", "EC-H"], "barrier": 0.58},
    {"reactants": ["C2H5", "FEC"], "products": ["C2H6", "FEC-H"], "barrier": 0.58},
    {"reactants": ["LiCH3OCO"], "products": ["CO", "LiOCH3"], "barrier": 0.09},
    {"reactants": ["LiCH3CH2OCO"], "products": ["CO", "LiOCH2CH3"], "barrier": 0.11},
    {"reactants": ["LiOCH3", "EMC"], "products": ["LiTetra112"], "barrier": 0.40},
    {"reactants": ["LiOCH2CH3", "EMC"], "products": ["LiTetra122"], "barrier": 0.39},
    {"reactants": ["LiTetra112"], "products": ["LiOCH2CH3", "DMC"], "barrier": 0.09},
    {"reactants": ["LiTetra122"], "products": ["LiOCH3", "DEC"], "barrier": 0.04},
    # Part 4 - DEC
    {"reactants": ["LiDEC"], "products": ["LEC", "C2H5"], "barrier": 0.70},
    {"reactants": ["LiDEC"], "products": ["LiOCH2CH3", "CH3CH2OCO"], "barrier": 0.42},
    # Part 5 - FEC
    {"reactants": ["LiFEC"], "products": ["LiFEC_RO"], "barrier": 0.09},
    {"reactants": ["LiFEC_RO"], "products": ["LiF", "FEC_RO-LiF"], "barrier": 0.09},
    {"reactants": ["FEC_RO-LiF"], "products": ["CO2", "OCHCH2"], "barrier": 0.52},
    {"reactants": ["FEC_RO-LiF_minus"], "products": ["CO2", "OCHCH2_minus"], "barrier": 0.77},
    {"reactants": ["OCHCH2_minus", "FEC"], "products": ["tetrahedral_FEC_minus"], "barrier": 0.13},
    # {"reactants": ["tetrahedral_FEC_minus"], "products": ["FEC_dimer_minus"], "barrier": 0.19},
    {"reactants": ["LiOCH3", "FEC"], "products": ["methoxydioxolanone", "LiF"], "barrier": 0.22},
    {"reactants": ["LiOCH2CH3", "FEC"], "products": ["ethoxydioxolanone", "LiF"], "barrier": 0.14},
    {"reactants": ["LiOCH3", "FEC"], "products": ["moec", "LiF"], "barrier": 0.20},
    {"reactants": ["LiOCH2CH3", "FEC"], "products": ["eoec", "LiF"], "barrier": 0.22},
    # Part 6 - VC
    {"reactants": ["LiVC"], "products": ["LiVC_RO"], "barrier": 0.05},
    {"reactants": ["LiVC_RO_minus"], "products": ["C2H2", "LiCO3_minus"], "barrier": 0.39},
    # Part 7 - PC
    {"reactants": ["LiPC"], "products": ["LiPC_RO"], "barrier": 0.40},
    {"reactants": ["LiPC_RO_minus"], "products": ["C3H6", "LiCO3_minus"], "barrier": 0.00},
    # Part 8 - BC
    {"reactants": ["LiBC"], "products": ["LiBC_RO"], "barrier": 0.40},
    {"reactants": ["LiBC_RO_minus"], "products": ["C4H8", "LiCO3_minus"], "barrier": 0.00},
]