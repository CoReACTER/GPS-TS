from pathlib import Path

from ase.io import read

import pytest


@pytest.fixture(scope="session")
def test_dir():
    return Path(__file__).parent.parent.joinpath("test_files").resolve()


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


@pytest.fixture(scope="session")
def molecules_spec(test_dir):
    reactant_1 = read(test_dir / "test_rxn_5" / "rxn5_reactant1_0_1.xyz")
    reactant_2 = read(test_dir / "test_rxn_5" / "rxn5_reactant2_-1_1.xyz")
    reactant_3 = read(test_dir / "test_rxn_5" / "rxn5_reactant3_0_1.xyz")
    product_1 = read(test_dir / "test_rxn_5" / "rxn5_product1_-1_1.xyz")
    product_2 = read(test_dir / "test_rxn_5" / "rxn5_product2_0_1.xyz")

    return {"reactants": [reactant_1, reactant_2, reactant_3], "products": [product_1, product_2]}