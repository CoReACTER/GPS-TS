import pytest

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from gpsts.utils import (
    atoms_to_molecule_graph,
    construct_molecule_from_adjacency_matrix,
    prepare_reaction_for_input,
    read_adjacency_matrix
)


def test_read_adjacency_matrix(test_dir):
    file = test_dir / "mol_adj_matrix" / "adj_matrix"

    matrix = read_adjacency_matrix(file)
    assert matrix.shape == ((13, 13))
    assert matrix[0, 0] == 0
    assert matrix[0, 1] == 2
    assert matrix[1, 0] == 2


def test_construct_molecule_from_adjacency_matrix(test_dir):
    adj_matrix = read_adjacency_matrix(test_dir / "mol_adj_matrix" / "adj_matrix")

    bonds = set()
    for i in range(adj_matrix.shape[0]):
        for j in range(i):
            if adj_matrix[i, j] != 0:
                bonds.add((j, i))

    mol = Molecule.from_file(test_dir / "mol_adj_matrix" / "initial_structure.xyz")

    new_mol = construct_molecule_from_adjacency_matrix(mol, adj_matrix)
    mg = MoleculeGraph.with_local_env_strategy(new_mol, OpenBabelNN())
    mg_bonds = mg.graph.edges()

    assert len(bonds) == len(mg_bonds)
    
    for bond in bonds:
        assert (bond in mg_bonds or (bond[1], bond[0]) in mg_bonds)


# TODO: add test with mapping pre-assigned
# Also need to test `clean` option for dumping to JSON
# Also also: need to test the graph hashing
def test_prepare_reaction_for_input(molecules_1r1p, molecules_2r1p, molecules_2r2p):
    # Simple case: one reactant, one product
    # Test molecule graphs, and mapping
    rcts_1r1p = [atoms_to_molecule_graph(molecules_1r1p["reactants"][0], charge=-1)]
    pros_1r1p = [atoms_to_molecule_graph(molecules_1r1p["products"][0], charge=-1)]
    inputs_1r1p = prepare_reaction_for_input(
        rcts_1r1p,
        pros_1r1p
    )

    assert inputs_1r1p["reactants"][0] == rcts_1r1p[0]
    assert inputs_1r1p["products"][0] == pros_1r1p[0]

    for i, e in enumerate([5, 7, 1, 10, 2, 4, 3, 8, 9, 0, 6]):
        assert inputs_1r1p["mapping"][(0, i)] == (0, e)

    # More complicated case: two reactants, one product
    # Test charges, spins, mapping, reacting atoms
    rcts_2r1p = [
        atoms_to_molecule_graph(molecules_2r1p["reactants"][0]),
        atoms_to_molecule_graph(molecules_2r1p["reactants"][1], charge=-1, spin_multiplicity=2)
    ]
    pros_2r1p = [
        atoms_to_molecule_graph(molecules_2r1p["products"][0], charge=-1, spin_multiplicity=2)
    ]

    inputs_2r1p = prepare_reaction_for_input(
        rcts_2r1p,
        pros_2r1p
    )
    
    assert inputs_2r1p["rct_charges"][0] == 0
    assert inputs_2r1p["rct_charges"][1] == -1
    assert inputs_2r1p["rct_spins"][0] == 1
    assert inputs_2r1p["rct_spins"][1] == 2

    assert inputs_2r1p["pro_charges"][0] == -1
    assert inputs_2r1p["pro_spins"][0] == 2

    mapping = {
        (0, 0): (0, 0),
        (0, 1): (0, 1),
        (0, 2): (0, 2),
        (0, 3): (0, 3),
        (0, 4): (0, 4),
        (0, 5): (0, 5),
        (0, 6): (0, 7),
        (0, 7): (0, 6),
        (0, 8): (0, 9),
        (0, 9): (0, 8),
        (1, 0): (0, 11),
        (1, 1): (0, 10)
    }

    for k, v in inputs_2r1p["mapping"].items():
        assert v == mapping[k]

    assert sorted(inputs_2r1p["reacting_atoms_reactants"][0]) == [1, 2]
    assert inputs_2r1p["reacting_atoms_reactants"][1] == [1]
    assert sorted(inputs_2r1p["reacting_atoms_products"][0]) == [1, 2, 10]
    
    # Yet more complicated: two reactants, two products
    # Test mapping, bonds breaking, bonds forming
    rcts_2r2p = [
        atoms_to_molecule_graph(molecules_2r2p["reactants"][0]),
        atoms_to_molecule_graph(molecules_2r2p["reactants"][1])
    ]
    pros_2r2p = [
        atoms_to_molecule_graph(molecules_2r2p["products"][0]),
        atoms_to_molecule_graph(molecules_2r2p["products"][1])
    ]

    inputs_2r2p = prepare_reaction_for_input(
        rcts_2r2p,
        pros_2r2p
    )

    # Everything is behaving as expected, but this is a somewhat problematic example
    # The mapping is strange and isn't spatially reasonable for the reaction at hand
    # The main issues here are:
    # 1. The reaction involves a ring with no rotatable bonds
    # 2. There are no stereocenters (the hydrogens are actually equivalent)
    # 3. There are two reaction centers that should be close to each other

    mapping = {
        (0, 0): (0, 4),
        (0, 1): (0, 2),
        (0, 2): (0, 0),
        (0, 3): (0, 3),
        (0, 4): (0, 5),
        (0, 5): (0, 1),
        (0, 6): (1, 3),
        (0, 7): (0, 6),
        (0, 8): (0, 7),
        (0, 9): (1, 2),
        (1, 0): (1, 0),
        (1, 1): (1, 1)
    }

    for k, v in inputs_2r2p["mapping"].items():
        assert v == mapping[k]

    assert inputs_2r2p["bonds_breaking"] == [((0, 2), (0, 6)), ((0, 5), (0, 9))]
    assert inputs_2r2p["bonds_forming"] == [((1, 0), (0, 6)), ((1, 1), (0, 9))]


# TODO: test for data loading