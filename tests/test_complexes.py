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


def test_make_complexes_onerct_onepro():
    pass


def test_make_complexes_tworct_onepro():
    pass


def test_make_complexes_tworct_twopro():
    pass


def test_make_complexes_onerct_threepro():
    pass
