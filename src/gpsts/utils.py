from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from monty.serialization import loadfn

import numpy as np

from openbabel import openbabel, pybel

from ase import Atoms
from ase.io import read

from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, oxygen_edge_extender, OpenBabelNN
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash

from gpsts.atom_mapping import get_reaction_atom_mapping


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
    atoms.charge = charge
    atoms.spin_multiplicity = spin_multiplicity
    mol = AseAtomsAdaptor.get_molecule(atoms)
    mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
    mg = metal_edge_extender(mg, **METAL_EDGE_EXTENDER_PARAMS)
    return mg


def read_adjacency_matrix(
    file: str | Path,
) -> np.ndarray:
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
