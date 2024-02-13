import pytest

from ase.io import read

from gpsts.complexes import select_central_molecule, make_complexes


@pytest.fixture(scope="session")
def molecules_1r1p(test_dir):
    reactant = read(test_dir / "test_rxn_1" / "rxn1_reactant_-1_1.xyz")
    product = read(test_dir / "test_rxn_1" / "rxn1_product_-1_1.xyz")

    return {"reactants": [reactant], "products": [product]}


@pytest.fixture(scope="session")
def molecules_2r1p(test_dir):
    reactant_1 = read(test_dir / "test_rxn_2" / "rxn2_reactant1_0_1.xyz")
    reactant_2 = read(test_dir / "test_rxn_2" / "rxn2_reactant2_-1_2.xyz")
    product = read(test_dir / "test_rxn_2" / "rxn2_product_-1_2.xyz")
    
    return {"reactants": [reactant_1, reactant_2], "products": [product]}


@pytest.fixture(scope="session")
def molecules_2r2p(test_dir):
    reactant_1 = read(test_dir / "test_rxn_3" / "rxn3_reactant1_0_1.xyz")
    reactant_2 = read(test_dir / "test_rxn_3" / "rxn3_reactant2_0_1.xyz")
    product_1 = read(test_dir / "test_rxn_3" / "rxn3_product1_0_1.xyz")
    product_2 = read(test_dir / "test_rxn_3" / "rxn3_product2_0_1.xyz")
    
    return {"reactants": [reactant_1, reactant_2], "products": [product_1, product_2]}


@pytest.fixture(scope="session")
def molecules_1r3p(test_dir):
    reactant = read(test_dir / "test_rxn_4" /  "rxn4_reactant_0_1.xyz")
    product_1 = read(test_dir / "test_rxn_4" / "rxn4_product1_0_1.xyz")
    product_2 = read(test_dir / "test_rxn_4" / "rxn4_product2_0_1.xyz")
    product_3 = read(test_dir / "test_rxn_4" / "rxn4_product3_0_1.xyz")
    
    return {"reactants": [reactant], "products": [product_1, product_2, product_3]}


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

    assert rxn1_ent_comp.get_chemical_symbols() == rxn1_exit_comp.get_chemical_symbols()


def test_make_complexes_tworct_onepro(molecules_2r1p):
    pass


def test_make_complexes_tworct_twopro(molecules_2r2p):
    pass


def test_make_complexes_onerct_threepro(molecules_1r3p):
    pass
