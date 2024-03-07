# stdlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Loading from e.g. JSON and zipped JSON files
from monty.serialization import loadfn

# n-dimensional arrays, etc.
import numpy as np

# Molecule representation, 3D structure generation, etc.
from openbabel import openbabel, pybel

# Molecule representation, mostly used for calculations and complex generation
from ase import Atoms
from ase.io import read

# Molecule representations, including graph representations
# Also tools for interconversion with ASE and OpenBabel
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, OpenBabelNN
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash

# Internal - atom mapping code
from gpsts.atom_mapping import get_reaction_atom_mapping


__author__ = "Evan Spotte-Smith, Samuel M. Blau"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


# "Constants" or standard parameters
METALS = [str(Element.from_Z(i)) for i in range(1, 87) if Element.from_Z(i).is_metal]

METAL_EDGE_EXTENDER_PARAMS = {
    "cutoff": 2.7,
    "metals": METALS,
    "coordinators": ("O", "N", "S", "C", "P", "Se", "Si", "Ge", "As", "Cl", "B", "I", "Br", "Te", "F", "Sb"),
}

MAX_BENCHMARK_REACTION_NUMATOMS = 30


def atoms_to_molecule_graph(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
) -> MoleculeGraph:
    """

    Utility function to convert an ASE `Atoms` object to a pymatgen `MoleculeGraph` object

    Args:
        atoms (Atoms): Molecule as an ASE `Atoms` object
        charge (int): Integral charge of the molecule
        spin_multiplicity (int): Integral spin multiplicity of the molecule

    Returns:
        mg (MoleculeGraph): MoleculeGraph based on the provided `atoms`

    """

    # Need to provide for ASEAtomsAdaptor
    atoms.charge = charge
    atoms.spin_multiplicity = spin_multiplicity

    mol = AseAtomsAdaptor.get_molecule(atoms)
    mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    mg = metal_edge_extender(mg, **METAL_EDGE_EXTENDER_PARAMS)
    # TODO: should we include oxygen_edge_extender here?
    
    return mg


def read_adjacency_matrix(
    file: str | Path,
) -> np.ndarray:
    """

    Read in an adjacency matrix from a file.
    
    This utility function is used to generate benchmark reactions from the `sella` paper [1] and subsequent
    (unpublished) work by the groups of Blau and Head-Gordon.

    Args:
        file (str | Path): Path to a file where the adjacency matrix is stored.
            Note that this function assumes a format with no header, where each line is a space-separated list of
            either 0 (no bond) or 1 (bond), e.g.

            0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0
            1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0
            0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0
            0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0
            0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0

    Returns:
        array (np.ndarray): The adjacency matrix as a numpy array

    """

    if isinstance(file, str):
        file = Path(file)

    with open(file) as file_obj:
        lines = file_obj.readlines()

        dimension = len(lines[0].strip().split())
        array = np.zeros((dimension, dimension))

        for ir, line in enumerate(lines):
            for ic, datum in enumerate(line.strip().split()):
                array[ir, ic] = int(datum)
    
    return array


def construct_molecule_from_adjacency_matrix(
    initial_structure: Molecule,
    matrix: np.ndarray,
    optimization_steps: int = 500
) -> Molecule:

    """

    Generate a 3D molecular structure from an adjacency matrix, primarily using OpenBabel

    Args:
        initial_structure (Molecule): base molecule to be used to generate a new structure
            with specified bonds
        matrix (np.ndarray): Adjacency matrix. Values are either 0 (no bond) or 1 (bond),
            and the shape is nxn, where `n` is the number of atoms in the molecule
        optimization_steps (int): Maximum number of steps for rough, force-field based optimization in OpenBabel.
            Default is 500

    Returns:
        output_mol (Molecule): pymatgen Molecule object

    """
    
    # Have to go through OpenBabel to make a reasonable structure based on bond orders
    # This code is taken from pymatgen - thanks Shyue Ping Ong and Qi Wang
    ob_mol = openbabel.OBMol()
    ob_mol.BeginModify()
    for site in initial_structure:
        coords = list(site.coords)
        atom_no = site.specie.Z
        ob_atom = openbabel.OBAtom()
        ob_atom.thisown = 0
        ob_atom.SetAtomicNum(atom_no)
        ob_atom.SetVector(*coords)
        ob_mol.AddAtom(ob_atom)
        del ob_atom

    bonds = set()
    # Extract unique bonds and bond orders from the adjacency matrix
    # There must be some nicer way to do this, but this should be plenty fast for our purposes
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i, j] != 0:
                bonds.add((j, i, matrix[i, j]))

    # Add in bonds
    # This will then be used to generate guess structures
    for bond in bonds:
        ob_mol.AddBond(
            int(bond[0]) + 1,  # OpenBabel uses 1-based indexing
            int(bond[1]) + 1,
            int(bond[2]),
            0,
            -1
        )
    
    pybelmol = pybel.Molecule(ob_mol)
    pybelmol.localopt(forcefield="uff", steps=optimization_steps)
    ob_mol = pybelmol.OBMol

    # Move from partially optimized OpenBabel molecule to pymatgen Molecule
    ad = BabelMolAdaptor(ob_mol)
    output_mol = ad.pymatgen_mol
    output_mol.set_charge_and_spin(initial_structure.charge, spin_multiplicity=initial_structure.spin_multiplicity) 
    
    return output_mol


def prepare_reaction_for_input(
    rct_mgs: List[MoleculeGraph],
    pro_mgs: List[MoleculeGraph],
    mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    label: Optional[str] = None,
    clean: bool = False
) -> Dict[str, Any]:

    """

    Prepare an input for `gpsts.complexes.make_complexes` based on a collection of reactant and product molecule
    graphs.

    Args:
        rct_mgs (List[MoleculeGraph]): Molecule graph representations for each reactant in this reaction
        pro_mgs (List[MoleculeGraph]): Molecule graph representations for each product in this reaction
        mapping (Optional[Dict[Tuple[int, int], Tuple[int, int]]]): Atom mapping between reactants and products.
            Default is None, meaning that an atom mapping will be generated
        label (Optional[str]): String label used to tag this reaction. Default is None.
        clean (bool): If True (default False), then convert MoleculeGraph objects to dictionaries to enable
            dumping

    Returns:
        reaction_data (Dict[str, Any]): Input dictionary, where keys are input parameters for `make_complexes` and
            values are relevant inputs for this reaction

    """

    # Charges and spins
    rct_charges = {i: e.molecule.charge for i, e in enumerate(rct_mgs)}
    rct_spins = {i: e.molecule.spin_multiplicity for i, e in enumerate(rct_mgs)}

    pro_charges = {i: e.molecule.charge for i, e in enumerate(pro_mgs)}
    pro_spins = {i: e.molecule.spin_multiplicity for i, e in enumerate(pro_mgs)}
    
    # If we don't have a mapping a priori, need to perform atom mapping
    if mapping is None:
        rct_map_number, prdt_map_number, _ = get_reaction_atom_mapping(
                rct_mgs, pro_mgs
        )

        # Reformat mapping
        #TODO: This is pretty inefficient (though much less costly than the MLIP)
        # Should probably just rewrite the atom mapping code to have a better output format
        mapping = dict()
        for ir, rct in enumerate(rct_map_number):
            for aatomind, aindex in rct.items():
                match = False

                for ip, pro in enumerate(prdt_map_number):
                    if match:
                        break

                    for batomind, bindex in pro.items():
                        if aindex == bindex:
                            mapping[(ir, aatomind)] = (ip, batomind)
                            match = True
                            break

                # No match found in any of the products
                # Should never happen, if atom mapping code isn't broken...
                if not match:
                    raise ValueError(f"Mapping failed! Atom {aatomind} of reactant {ir} could not be matched!")
    
    inverse_mapping = dict()
    for a, b in mapping.items():
        inverse_mapping[b] = a

    # Sanity check
    assert len(mapping) == sum([len(x.molecule) for x in rct_mgs])

    # Identifying broken and formed bonds
    bonds_rct = list()
    bonds_pro = list()

    bonds_breaking = list()
    bonds_forming = list()

    for ii, mg in enumerate(rct_mgs):
        for bond in mg.graph.edges():
            bonds_rct.append(
                (
                    (ii, bond[0]),
                    (ii, bond[1])
                )
            )

    for ii, mg in enumerate(pro_mgs):
        for bond in mg.graph.edges():
            bonds_pro.append(
                (
                    (ii, bond[0]),
                    (ii, bond[1])
                )
            )

    for bond in bonds_rct:
        map_bond_a = mapping[bond[0]]
        map_bond_b = mapping[bond[1]]
        if (
            (map_bond_a, map_bond_b) not in bonds_pro
            and (map_bond_b, map_bond_a) not in bonds_pro
        ):
            bonds_breaking.append(bond)

    for bond in bonds_pro:
        map_bond_a = inverse_mapping[bond[0]]
        map_bond_b = inverse_mapping[bond[1]]
        if (
            (map_bond_a, map_bond_b) not in bonds_rct
            and (map_bond_b, map_bond_a) not in bonds_rct
        ):
            bonds_forming.append(
                (map_bond_a, map_bond_b)
            )

    # Identify reacting atoms in reactants and products
    reacting_atoms_reactants = {i: list() for i in range(len(rct_mgs))}
    reacting_atoms_products = {i: list() for i in range(len(pro_mgs))}
    for bond in bonds_forming + bonds_breaking:
        ra_mol = bond[0][0]
        ra_atom = bond[0][1]
        rb_mol = bond[1][0]
        rb_atom = bond[1][1]

        pa = mapping[bond[0]]
        pb = mapping[bond[1]]


        if ra_atom not in reacting_atoms_reactants[ra_mol]:
            reacting_atoms_reactants[ra_mol].append(ra_atom)
        if rb_atom not in reacting_atoms_reactants[rb_mol]:
            reacting_atoms_reactants[rb_mol].append(rb_atom)

        if pa[1] not in reacting_atoms_products[pa[0]]:
            reacting_atoms_products[pa[0]].append(pa[1])
        if pb[1] not in reacting_atoms_products[pb[0]]:
            reacting_atoms_products[pb[0]].append(pb[1])

    reactant_graph_hashes = [
        weisfeiler_lehman_graph_hash(x.graph.to_undirected(), node_attr="specie")
        for x in rct_mgs
    ]

    product_graph_hashes = [
        weisfeiler_lehman_graph_hash(x.graph.to_undirected(), node_attr="specie")
        for x in pro_mgs
    ]

    if clean:
        rct_mgs = [x.as_dict() for x in rct_mgs]
        pro_mgs = [x.as_dict() for x in pro_mgs]

    reaction_data = {
        "label": label,
        "reactants": rct_mgs,
        "products": pro_mgs,
        "reactant_graph_hashes": reactant_graph_hashes,
        "product_graph_hashes": product_graph_hashes,
        "mapping": mapping,
        "reacting_atoms_reactants": reacting_atoms_reactants,
        "reacting_atoms_products": reacting_atoms_products,
        "bonds_breaking": bonds_breaking,
        "bonds_forming": bonds_forming,
        "rct_charges": rct_charges,
        "rct_spins": rct_spins,
        "pro_charges": pro_charges,
        "pro_spins": pro_spins
    }
    
    return reaction_data


def load_benchmark_data(data_path: str | Path) -> List[Dict[str, Any]]:
    """

    Load a collection of reactions from a dumped JSON or gzipped JSON file

    Args:
        data_path (str | Path): Path to dumped benchmark data file

    Returns:
        processed_data (List[Dict[str, Any]]): Parsed data

    """
    
    data = loadfn(data_path)

    processed_data = list()

    for datum in data:

        processed_datum = {
            "label": datum["label"],
            "reactants": datum["reactants"],
            "products": datum["products"],
            "reactant_graph_hashes": datum["reactant_graph_hashes"],
            "product_graph_hashes": datum["product_graph_hashes"],
        }

        mapping = {eval(k): tuple(v) for k, v in datum["mapping"].items()}
        processed_datum["mapping"] = mapping

        for key in ["reacting_atoms_reactants", "reacting_atoms_products"]:
            processed_datum[key] = {int(k): tuple(v) for k, v in datum[key].items()}

        for key in ["bonds_breaking", "bonds_forming"]:
            this_set = list()
            for collection in datum[key]:
                this_set.append((tuple(collection[0]), tuple(collection[1])))
            processed_datum[key] = this_set
        
        for key in ["rct_charges", "rct_spins", "pro_charges", "pro_spins"]:
            processed_datum[key] = {int(k): v for k, v in datum[key].items()}
        
        processed_data.append(processed_datum)
    
    return processed_data


def oxygen_edge_extender(
    mol_graph: MoleculeGraph,
    hydrogen_cutoff: float = 1.2,
    carbon_cutoff: float = 1.7
) -> MoleculeGraph:
    """
    Identify and add missed O-C or O-H bonds. This is particularly
    important when oxygen is forming three bonds, e.g. in H3O+ or XOH2+.
    See https://github.com/materialsproject/pymatgen/pull/2903 for details.

    TODO:
        - This should be in pymatgen
        - This should be generalized and should have flexible parameters, more like metal_edge_extender

    Args:
        mol_graph (MoleculeGraph): molecule graph to extend
        hydrogen_cutoff (float): 
        carbon_cutoff (float):

    Returns:
        MoleculeGraph: object with additional O-C or O-H bonds added (if any found)
    """
    num_new_edges = 0
    for idx in mol_graph.graph.nodes():
        if mol_graph.graph.nodes()[idx]["specie"] == "O":
            neighbors = [site[2] for site in mol_graph.get_connected_sites(idx)]
            for ii, site in enumerate(mol_graph.molecule):
                is_O_C_bond = str(site.specie) == "C" and site.distance(mol_graph.molecule[idx]) < carbon_cutoff
                is_O_H_bond = str(site.specie) == "H" and site.distance(mol_graph.molecule[idx]) < hydrogen_cutoff
                if ii != idx and ii not in neighbors and (is_O_C_bond or is_O_H_bond):
                    mol_graph.add_edge(idx, ii)
                    num_new_edges += 1
    return mol_graph