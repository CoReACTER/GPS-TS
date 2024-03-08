import pytest

from ase.io import read

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

from gpsts.utils import (
    atoms_to_molecule_graph,
    construct_molecule_from_adjacency_matrix,
    load_benchmark_data,
    METAL_EDGE_EXTENDER_PARAMS,
    oxygen_edge_extender,
    prepare_reaction_for_input,
    read_adjacency_matrix
)


@pytest.fixture(scope="session")
def graph_mol(test_dir):
    
    atoms = read(test_dir / "graphs" / "furan1_pro_1.xyz", format="xyz")
    mol = Molecule.from_file(test_dir / "graphs" / "furan1_pro_1.xyz")
    mol.set_charge_and_spin(1)
    
    return atoms, mol


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


def test_prepare_reaction_for_input(molecules_1r1p, molecules_2r1p, molecules_2r2p):
    # Simple case: one reactant, one product
    # Test molecule graphs, and mapping
    rcts_1r1p = [atoms_to_molecule_graph(molecules_1r1p["reactants"][0], charge=-1, spin_multiplicity=1)]
    pros_1r1p = [atoms_to_molecule_graph(molecules_1r1p["products"][0], charge=-1, spin_multiplicity=1)]
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
        atoms_to_molecule_graph(molecules_2r1p["reactants"][0], charge=0, spin_multiplicity=1),
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
        atoms_to_molecule_graph(molecules_2r2p["reactants"][0], charge=0, spin_multiplicity=1),
        atoms_to_molecule_graph(molecules_2r2p["reactants"][1], charge=0, spin_multiplicity=1)
    ]
    pros_2r2p = [
        atoms_to_molecule_graph(molecules_2r2p["products"][0], charge=0, spin_multiplicity=1),
        atoms_to_molecule_graph(molecules_2r2p["products"][1], charge=0, spin_multiplicity=1)
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

    # Test pre-assigned atom mapping, label, and "clean" option
    # Also test hashing
    assigned_map = {
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

    inputs_2r2p_assigned = prepare_reaction_for_input(
        rcts_2r2p,
        pros_2r2p,
        mapping=assigned_map,
        label="test",
        clean=True
    )

    for k, v in inputs_2r2p_assigned["mapping"].items():
        assert v == assigned_map[k]

    assert inputs_2r2p_assigned["bonds_breaking"] == [((0, 2), (0, 7)), ((0, 5), (0, 9))]
    assert inputs_2r2p_assigned["bonds_forming"] == [((1, 0), (0, 9)), ((1, 1), (0, 7))]

    assert inputs_2r2p_assigned["label"] == "test"

    assert isinstance(inputs_2r2p_assigned["reactants"][0], dict)

    # Hashes should be the same regardless of if a map is assigned
    assert sorted(inputs_2r2p_assigned["reactant_graph_hashes"]) == sorted(inputs_2r2p["reactant_graph_hashes"])
    assert inputs_2r2p_assigned["reactant_graph_hashes"][0] == "176ba51f33c41703bc7ae8d746d124cd"


def test_atoms_to_molecule_graph(graph_mol):
    
    atoms, mol = graph_mol

    ref_mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    ref_mg = metal_edge_extender(ref_mg, **METAL_EDGE_EXTENDER_PARAMS)

    mg = atoms_to_molecule_graph(atoms, charge=1, spin_multiplicity=1)
    assert mg.molecule.charge == 1
    assert mg.molecule.spin_multiplicity == 1
    assert ref_mg.isomorphic_to(mg)
    assert ref_mg == mg


def test_oxygen_edge_extender(graph_mol):
    
    mol = graph_mol[1]

    orig_mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    assert len(orig_mg.graph.edges(2)) == 3

    # With too small cutoff, bond should not be detected
    mg_smallcutoff = oxygen_edge_extender(orig_mg, carbon_cutoff=1.4)
    assert len(mg_smallcutoff.graph.edges(2)) == 3

    # With larger cutoff, bond will be detected
    mg_largecutoff = oxygen_edge_extender(orig_mg, carbon_cutoff=1.7)
    assert len(mg_largecutoff.graph.edges(2)) == 4


def test_load_benchmark_data(test_dir):
    
    data = load_benchmark_data(test_dir / "elyte_ts_pf6.json")
    assert len(data) == 24