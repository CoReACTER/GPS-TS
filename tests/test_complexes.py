import pytest

from ase.io import read

from gpsts.complexes import select_central_molecule, make_complexes


def test_select_central_molecule(molecules_2r1p):
    
    # Silly case: one molecule
    rxn2_atoms_rct = {0: [1, 2], 1: [0]}
    rxn2_atoms_pro = {0: [1, 2, 10]}
    assert select_central_molecule(molecules_2r1p["products"], rxn2_atoms_pro)[0] == 0

    # Two molecules, one with more reacting atoms
    assert select_central_molecule(molecules_2r1p["reactants"], rxn2_atoms_rct)[0] == 0

    # TODO: Need case where two molecules have equal numbers of reacting atoms
    # Simple example: Diels-Alder reaction?


def test_make_complexes_onerct_onepro(molecules_1r1p):

    rxn1_mapping = {(0, e): (0, i) for i, e in enumerate([8, 2, 5, 3, 4, 9, 0, 1, 7, 10, 6])}
    rxn1_atoms_rct = {0: [3, 7]}
    rxn1_atoms_pro = {0: [rxn1_mapping[(0, 3)], rxn1_mapping[(0, 7)]]}

    rxn1_bonds_breaking = [((0, 3), (0, 7))]
    rxn1_bonds_forming = list()

    rxn1_ent_comp, rxn1_exit_comp = make_complexes(
        molecules_1r1p["reactants"],
        molecules_1r1p["products"],
        rxn1_mapping,
        rxn1_atoms_rct,
        rxn1_atoms_pro,
        rxn1_bonds_breaking,
        rxn1_bonds_forming,
        rct_charges={0: -1},
        rct_spins={0: 1},
        pro_charges={0: -1},
        pro_spins={0: 1},
    )

    # This algorithm is stochastic, so there's no guarantee that we'll get the same complex
    # Architector tests by aligning the obtained complex with a reference
    # Since we have noncovalent, nondative bonds, even that may not work
    # TODO: think more deeply about how to really test the obtained structures
    assert rxn1_ent_comp.get_chemical_symbols() == rxn1_exit_comp.get_chemical_symbols()


def test_make_complexes_tworct_onepro(molecules_2r1p):

    rxn2_mapping = {
        (0, i): (0, i) for i in range(len(molecules_2r1p["reactants"][0]))
    }
    rxn2_mapping.update(
        {
            (1, 0): (0, 10),
            (1, 1): (0, 11)
        }
    )
    rxn2_atoms_rct = {0: [1, 2], 1: [0]}
    rxn2_atoms_pro = {0: [1, 2, 10]}

    rxn2_bonds_breaking = [((0, 1), (0, 1))]
    rxn2_bonds_forming = [((0, 2), (1, 0))]

    rxn2_ent_comp, rxn2_exit_comp = make_complexes(
        molecules_2r1p["reactants"],
        molecules_2r1p["products"],
        rxn2_mapping,
        rxn2_atoms_rct,
        rxn2_atoms_pro,
        rxn2_bonds_breaking,
        rxn2_bonds_forming,
        rct_charges={0: 0, 1: -1},
        rct_spins={0: 1, 1: 2},
        pro_charges={0: -1},
        pro_spins={0: 2},
    )

    assert rxn2_ent_comp.get_chemical_symbols() == rxn2_exit_comp.get_chemical_symbols()


def test_make_complexes_tworct_twopro(molecules_2r2p):

    rxn3_mapping = {
        (0, 0): (0, 5),
        (0, 1): (0, 2),
        (0, 2): (0, 1),
        (0, 3): (0, 3),
        (0, 4): (0, 4),
        (0, 5): (0, 0),
        (0, 6): (0, 7),
        (0, 7): (1, 2),
        (0, 8): (0, 6),
        (0, 9): (1, 3),
        (1, 0): (1, 0),
        (1, 1): (1, 1)
    }

    rxn3_atoms_rct = {0: [2, 5, 7, 9], 1: [0, 1]}
    rxn3_atoms_pro = {0: [0, 1], 1: [0, 1, 2, 3]}

    rxn3_bonds_breaking = [((0, 2), (0, 7)), ((0, 5), (0, 9))]
    rxn3_bonds_forming = [((0, 7), (1, 1)), ((0, 9), (1, 0))]

    rxn3_ent_comp, rxn3_exit_comp = make_complexes(
        molecules_2r2p["reactants"],
        molecules_2r2p["products"],
        rxn3_mapping,
        rxn3_atoms_rct,
        rxn3_atoms_pro,
        rxn3_bonds_breaking,
        rxn3_bonds_forming
    )

    assert rxn3_ent_comp.get_chemical_symbols() == rxn3_exit_comp.get_chemical_symbols()


def test_make_complexes_onerct_threepro(molecules_1r3p):

    rxn4_mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (1, 0),
        (0, 3): (0, 2),
        (0, 4): (0, 3),
        (0, 5): (0, 4),
        (0, 6): (0, 5),
        (0, 7): (2, 0),
        (0, 8): (2, 1),
        (0, 9): (2, 2),
        (0, 10): (0, 6),
        (0, 11): (1, 1)
    }
    rxn4_atoms_rct = {0: [0, 2, 5, 6, 8, 9, 10]}
    rxn4_atoms_pro = {0: [0, 4, 5, 6], 1: [0], 2: [1, 2]}

    rxn4_bonds_breaking = [((0, 0), (0, 2)), ((0, 6), (0, 8)), ((0, 5), (0, 10)), ((0, 9), (0, 10))]
    rxn4_bonds_forming = list()

    rxn4_ent_comp, rxn4_exit_comp = make_complexes(
        molecules_1r3p["reactants"],
        molecules_1r3p["products"],
        rxn4_mapping,
        rxn4_atoms_rct,
        rxn4_atoms_pro,
        rxn4_bonds_breaking,
        rxn4_bonds_forming
    )
    