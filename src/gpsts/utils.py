from typing import Any, Dict, List

from ase import Atoms
from ase.io import read
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import metal_edge_extender, oxygen_edge_extender, OpenBabelNN

from gpsts.atom_mapping import get_reaction_atom_mapping

METALS = [str(Element.from_Z(i)) for i in range(1, 87) if Element.from_Z(i).is_metal]

METAL_EDGE_EXTENDER_PARAMS = {
    "cutoff": 2.7,
    "metals": METALS,
    "coordinators": ("O", "N", "S", "C", "P", "Se", "Si", "Ge", "As", "Cl", "B", "I", "Br", "Te", "F", "Sb"),
}


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


def split_complex_to_components():
    pass


def prepare_reaction_for_input(
    rct_mgs: List[MoleculeGraph],
    pro_mgs: List[MoleculeGraph]
) -> Dict[str, Any]:

    # Charges and spins
    rct_charges = {i: e.molecule.charge for i, e in enumerate(rct_mgs)}
    rct_spins = {i: e.molecule.spin_multiplicity for i, e in enumerate(rct_mgs)}

    pro_charges = {i: e.molecule.charge for i, e in enumerate(pro_mgs)}
    pro_spins = {i: e.molecule.spin_multiplicity for i, e in enumerate(pro_mgs)}
    
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

    reaction_data = {
        "reactants": rct_mgs,
        "products": pro_mgs,
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