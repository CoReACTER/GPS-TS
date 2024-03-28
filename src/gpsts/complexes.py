# Copyright (c) CoReACTER.
# Distributed under the terms of the GPL version 3.

#stdlib
from typing import Any, List, Dict, Optional, Tuple, Union
import random

# Basic numeric/scientific python libraries
import numpy as np

# For reaction complex formation
from architector import view_structures, convert_io_molecule
import architector.io_arch_dock as io_arch_dock
from architector.io_molecule import Molecule as ArchMol

# For molecule representations
from ase import Atoms
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


def make_complex(
    core: ArchMol,
    other: ArchMol,
    reacting_atom_core: int,
    reacting_atom_other: int,
    params: Dict = dict()  # Can pass any normal Architector param here
):
    """
    Use Architector to create a complex, where the `other` molecule is
    noncovalently bound to the `core` molecule.

    Args:
        core (architector.io_molecule.Molecule, aka ArchMol): the core of the complex
        other (ArchMol): the molecule that will be (noncovalently) attached to the `core`
        reacting_atom_core (int): 0-based index of the atom in the `core` that will be connected to the `other` 
        reacting_atom_other (int): 0-based index of the atom in `other` that will be connected to the `core`
        params (Dict): Architector parameter dict; see Architector for details

    Returns:
        binding (Tuple[ArchMol, List[ArchMol]]): New complex combining `core` and `other`.

    References:
        `Architector for high-throughput cross-periodic table 3D complex building`,
        Nat. Commun. 2023, 14(1), p.2786, https://doi.org/10.1038/s41467-023-38169-2.
    """
    
    params['species_list'] = [other]
    params['species_location_method'] = 'targeted'

    # TODO: make it possible to add multiple attachment sites
    params['targeted_indices_close'] = [reacting_atom_core, reacting_atom_other]

    binding = io_arch_dock.add_non_covbound_species(core, parameters=params)
    
    return binding


def select_central_molecule(
    molecules: List[ArchMol], 
    reacting_atoms: Dict[int, List[int]]
) -> Tuple[int, ArchMol]:
    """
    From a collection of reacting molecules, identify which will be the `core` of the reacting complex.

    Currently, we somewhat naively choose the molecule with the most atoms involved in the reaction.
    If there is a tie, then the larger molecule is chosen to be the core.
    
    Note that this procedure may be subject to change in future versions.

    Args:
        molecules (List[architector.io_molecule.Molecule, aka ArchMol]): List of reacting molecules
        reacting_atoms (Dict[int, List[int]]): Map {molecule_index: atom_indices}, where `molecule_index`
            is the 0-based index of the molecule in `molecules` and `atom_indices` is a list of (0-based)
            indices of atoms in that molecule involved in the reaction (typically, these are atoms that have
            bonds changing in the reaction)

    Returns:
        central_index (int): Index of the atom to be used as the core of the complex
        central_mol (ArchMol): The chosen central molecule

    """

    # Pick one molecule to be the center
    # Arbitrarily, we choose the molecule with the most reacting atoms
    # If there's a tie, we choose the one that's largest
    central_index = None
    central_mol = None
    for ii, mol in enumerate(molecules):
        react_atoms = reacting_atoms.get(ii)
        if react_atoms is None:
            # Spectator - not directly participating in the reaction
            continue

        if central_index is None:
            central_index = ii
            central_mol = mol
        else:
            current_central_reacting_atoms = len(reacting_atoms[central_index])
            if len(react_atoms) > current_central_reacting_atoms:
                central_index = ii
                central_mol = mol
            elif len(react_atoms) == current_central_reacting_atoms:
                if len(mol.ase_atoms.get_chemical_symbols()) > len(central_mol.ase_atoms.get_chemical_symbols()):
                    central_index = ii
                    central_mol = mol

    return central_index, central_mol


def make_complexes(
    reactants: List[Union[Molecule, Atoms]],
    products: List[Union[Molecule, Atoms]],
    mapping: Dict[Tuple[int, int], Tuple[int, int]],
    reacting_atoms_reactants: Dict[int, List[int]],
    reacting_atoms_products: Dict[int, List[int]],
    bonds_breaking: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    bonds_forming: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    reactant_charges: Dict[int, int] = dict(),
    reactant_spins: Dict[int, int] = dict(),
    product_charges: Dict[int, int] = dict(),
    product_spins: Dict[int, int] = dict(),
    reactant_spectator_binding: Dict[int, Tuple[int, Tuple[int, int]]] = dict(),
    product_spectator_binding: Dict[int, Tuple[int, Tuple[int, int]]] = dict(),
    reactant_core: Optional[int] = None,
    product_core: Optional[int] = None,
    architector_params: Dict = dict()
) -> Tuple[Atoms, Atoms]:

    """

    Generate entrance and exit complexes for a reaction defined by a set of reactants and a set of products.

    Args:
        reactants (List[Union[Molecule, Atoms]]): Collection of reactant molecules, either as pymatgen Molecule objects
            or ASE Atoms objects
        products (List[Union[Molecule, Atoms]]): Collection of product molecules, either as pymatgen Molecule objects
            or ASE Atoms objects
        mapping (Dict[Tuple[int, int], Tuple[int, int]]): Atom mapping between reactants (keys) and products (values).
            Both keys and values are tuples (mol_ind, atom_ind), where `mol_ind` is the 0-based index of the molecule
            in `reactants` or `products` and `atom_ind` is the 0-based index of the atom in that molecule
        reacting_atoms_reactants (Dict[int, List[int]]): Key-value pair {mol_ind: atom_inds}, where `mol_ind` is the
            0-based index of the molecule in `reactants` and `atom_inds` is a list of atom indices in the molecule that
            are involved in the reaction. Typically, "involved in the reaction" means that the atom has some bonds that
            form or break during the reaction
        reacting_atoms_products (Dict[int, List[int]]): Key-value pair {mol_ind: atom_inds}, where `mol_ind` is the
            0-based index of the molecule in `products` and `atom_inds` is a list of atom indices in the molecule that
            are involved in the reaction. Typically, "involved in the reaction" means that the atom has some bonds that
            form or break during the reaction
        bonds_breaking (List[Tuple[Tuple[int, int], Tuple[int, int]]]): Bonds in `reactants` that break during the
            reaction. This collection is formatted as ((mol_1_ind, atom_1_ind), (mol_2_ind, atom_2_ind)), where
            `mol_x_ind` is the 0-based index of the xth molecule in `reactants` and `atom_x_ind` is the 0-based index
            of the atom in the xth molecule that is involved in this bond
        bonds_forming (List[Tuple[Tuple[int, int], Tuple[int, int]]]): Bonds in `reactants` that form during the
            reaction. This collection is formatted as ((mol_1_ind, atom_1_ind), (mol_2_ind, atom_2_ind)), where
            `mol_x_ind` is the 0-based index of the xth molecule in `reactants` and `atom_x_ind` is the 0-based index
            of the atom in the xth molecule that is involved in this bond
        reactant_charges (Dict[int, int]): Charges of the molecules in `reactants`. Keys are 0-based indexes of the
            molecules in `reactants`, and values are integral charges
        reactant_spins (Dict[int, int]): Spin multiplicities of the molecules in `reactants`. Keys are 0-based indexes
            of the molecules in `reactants`, and values are integral spin multiplicities
        product_charges (Dict[int, int]): Charges of the molecules in `products`. Keys are 0-based indexes of the
            molecules in `products`, and values are integral charges
        product_spins (Dict[int, int]): Spin multiplicities of the molecules in `products`. Keys are 0-based indexes of
            the molecules in `products`, and values are integral spin multiplicities
        reactant_spectator_binding (Dict[int, Tuple[int, Tuple[int, int]]]): Keys are indices of molecules in
            `reactants` that are spectators (do not participate in any covalent or covalent-like bond breaking or bond
            forming). Values take the format (spectator_atom_index, (binding_molecule_index, binding_atom_index)),
            where `spectator_atom_index` is the index of the atom in the particular spectator molecule that should
            be close to the atom `binding_atom_index` in the molecule in `reactants` with index
            `binding_molecule_index`. Note that the binding molecule (`binding_molecule_index`) can be another
            spectator or a molecule participating in the reaction, but it cannot be the same index as the key. Further
            note that, for now, spectators must "bind" to reacting molecules. We hope to relax this requirement soon.
        product_spectator_binding (Dict[int, Tuple[int, Tuple[int, int]]]): Keys are indices of molecules in
            `products` that are spectators (do not participate in any covalent or covalent-like bond breaking or bond
            forming). Values take the format (spectator_atom_index, (binding_molecule_index, binding_atom_index)),
            where `spectator_atom_index` is the index of the atom in the particular spectator molecule that should
            be close to the atom `binding_atom_index` in the molecule in `products` with index
            `binding_molecule_index`. Note that the binding molecule (`binding_molecule_index`) can be another
            spectator or a molecule participating in the reaction, but it cannot be the same index as the key. Further
            note that, for now, spectators must "bind" to reacting molecules. We hope to relax this requirement soon.
        reactant_core (Optional[int] = None): Index of the molecule in `reactants` that will serve as the "core" of
            the entrance complex. Default is None, which means that this function will decide which molecule to use
            as the core.
        product_core (Optional[int] = None): Index of the molecule in `products` that will serve as the "core" of
            the exit complex. Default is None, which means that this function will decide which molecule to use
            as the core.
        architector_params (Dict): Architector parameter dict; see Architector for details

    Returns:
        entrance_complex (Atoms): Entrance complex, as an ASE Atoms object
        exit_complex (Atoms): Exit complex, as an ASE Atoms object

    References:
        `Architector for high-throughput cross-periodic table 3D complex building`,
        Nat. Commun. 2023, 14(1), p.2786, https://doi.org/10.1038/s41467-023-38169-2.

    """

    # Convert reactants and products to ase Atoms objects
    rcts = list()
    for ir, r in enumerate(reactants):
        if isinstance(r, Molecule):
            ratoms = AseAtomsAdaptor.get_atoms(r)
            ratoms = convert_io_molecule(ratoms)
        else:
            ratoms = convert_io_molecule(r)

        # TODO: this isn't guaranteed to provide a reasonable spin/charge pairing
        # Default spin should really be the lowest spin possible with the molecule's charge
        ratoms.charge = int(reactant_charges.get(ir, 0))
        ratoms.uhf = int(reactant_spins.get(ir, 1) - 1)
        rcts.append(ratoms)

    pros = list()
    for ip, p in enumerate(products):
        if isinstance(p, Molecule):
            patoms = AseAtomsAdaptor.get_atoms(p)
            patoms = convert_io_molecule(patoms)
        else:
            patoms = convert_io_molecule(p)

        # TODO: this isn't guaranteed to provide a reasonable spin/charge pairing
        # Default spin should really be the lowest spin possible with the molecule's charge
        patoms.charge = int(product_charges.get(ip, 0))
        patoms.uhf = int(product_spins.get(ip, 1) - 1)
        pros.append(patoms)

    entrance_complexes = list()
    exit_complexes = list()

    intermediate_entrance = list()
    intermediate_exit = list()
    
    if len(rcts) == 1:
        # For now, don't bother re-posing the reactant if there's only one
        # TODO: Probably we can use CREST to get a conformer that's closest to the product(s)?
        entrance_complex = rcts[0]
        
        new_mapping = {i: mapping[(0, i)] for i in range(len(entrance_complex.ase_atoms))}
    else:
        if reactant_core is None:
            central_index, central_mol = select_central_molecule(rcts, reacting_atoms_reactants)
            if central_index is None:
                raise ValueError("No reacting molecules that can act as complex cores! Check "
                                 "`reacting_atoms_reactants` to ensure that at least one molecule is not behaving as "
                                 "a spectator.")
        else:
            central_index = reactant_core
            central_mol = rcts[central_index]

        ordering = [central_index]
        current_complex = central_mol
        internal_mapping_entrance = {(central_index, i): i for i in range(len(central_mol.ase_atoms))}
        
        # Add one "ligand" (additional reactant) at a time
        # Right now, we add the largest first, then the smallest
        # Don't know if that's reasonable
        for ii, rct in sorted(enumerate(rcts), key=lambda x: len(x[1].ase_atoms)):
            if ii == central_index:
                continue

            elif ii in reactant_spectator_binding:
                continue

            possible_bonds = [
                e
                for e in bonds_breaking + bonds_forming
                if (e[0][0] in ordering and e[1][0] == ii)
                or (e[1][0] in ordering and e[0][0] == ii)
            ]

            if len(possible_bonds) == 0:
                raise ValueError(f"Cannot add molecule {ii}; no place to bind on molecules {ordering}."
                                 "Check `reacting_atoms_reactants`, `bonds_breaking`, `bonds_forming`, and "
                                 "`reactant_spectator_binding`. If all variables are correct, try manually selecting "
                                 "a core.")

            # For now, randomly select bond to focus on
            # TODO: Is there a better way to do this?
            bond = random.choice(possible_bonds)
            if bond[0][0] in ordering:
                reacting_atom_core = internal_mapping_entrance[bond[0]]
                reacting_atom_other = bond[1][1]
            else:
                reacting_atom_core = internal_mapping_entrance[bond[1]]
                reacting_atom_other = bond[0][1]

            binding = make_complex(current_complex, rct, reacting_atom_core, reacting_atom_other, params=architector_params)

            # Bookkeeping
            ordering.append(ii)
            current_complex = binding[0]
            for jj, kk in enumerate(range(len(internal_mapping_entrance), len(current_complex.ase_atoms))):
                internal_mapping_entrance[(ii, jj)] = kk
        
        # Now add spectators
        for ii, loc_info in reactant_spectator_binding.items():
            # No choices needed - just bring the spectator close to the specified atom in a molecule that's already in
            # the complex
            # TODO: should make sure that the target is actually in the complex already (i.e. that `loc_info[1]` is
            # already in `internal_mapping_entrance`)
            binding = make_complex(current_complex, rcts[ii], internal_mapping_entrance[loc_info[1]], loc_info[0])

            # Bookkeeping
            ordering.append(ii)
            current_complex = binding[0]
            for jj, kk in enumerate(range(len(internal_mapping_entrance), len(current_complex.ase_atoms))):
                internal_mapping_entrance[(ii, jj)] = kk

        entrance_complex = current_complex
        new_mapping = dict()
        for key, value in mapping.items():
            new_mapping[internal_mapping_entrance[key]] = value 
        
    if len(pros) == 1:
        exit_complex = pros[0]

        ec_atoms = exit_complex.ase_atoms

        # Reorder atoms to match entrance complex
        current_exit_species = ec_atoms.get_chemical_symbols()
        current_exit_coords = ec_atoms.get_positions()

        species = [None] * len(current_exit_species)
        coords = np.zeros(current_exit_coords.shape)

        for key, value in new_mapping.items():
            species[key] = current_exit_species[value[1]]  # With only one product, value[0] will be 0 in all cases
            coords[key] = current_exit_coords[value[1]]

        new_exit_complex =  convert_io_molecule(Atoms(symbols=species, positions=coords))
        new_exit_complex.charge = exit_complex.charge
        new_exit_complex.uhf = exit_complex.uhf
        exit_complex = new_exit_complex
    
    else:
        if product_core is None:
            central_index, central_mol = select_central_molecule(pros, reacting_atoms_products)
            if central_index is None:
                raise ValueError("No reacting molecules that can act as complex cores! Check "
                                 "`reacting_atoms_products` to ensure that at least one molecule is not behaving as "
                                 "a spectator.")
        else:
            central_index = reactant_core
            central_mol = pros[central_index]

        ordering = [central_index]
        current_complex = central_mol
        internal_mapping_exit = {(central_index, i): i for i in range(len(central_mol.ase_atoms))}
        
        # Add one "ligand" (additional reactant) at a time, 
        for ii, pro in sorted(enumerate(pros), key=lambda x: len(x[1].ase_atoms)):
            if ii == central_index:
                continue

            elif ii in product_spectator_binding:
                continue

            possible_bonds = [
                (mapping[e[0]], mapping[e[1]])
                for e in bonds_breaking + bonds_forming
                if (mapping[e[0]][0] in ordering and mapping[e[1]][0] == ii)
                or (mapping[e[1]][0] in ordering and mapping[e[0]][0] == ii)
            ]

            if len(possible_bonds) == 0:
                raise ValueError(f"Cannot add molecule {ii}; no place to bind on molecules {ordering}."
                                 "Check `reacting_atoms_products`, `bonds_breaking`, `bonds_forming`, and "
                                 "`product_spectator_binding`. If all variables are correct, try manually selecting "
                                 "a core.")

            # For now, randomly select bond to focus on
            # TODO: Is there a better way to do this?
            bond = random.choice(possible_bonds)
            if bond[0][0] in ordering:
                reacting_atom_core = internal_mapping_exit[bond[0]]
                reacting_atom_other = bond[1][1]
            else:
                reacting_atom_core = internal_mapping_exit[bond[1]]
                reacting_atom_other = bond[0][1]

            binding = make_complex(current_complex, pro, reacting_atom_core, reacting_atom_other, params=architector_params)

            # Bookkeeping
            ordering.append(ii)
            current_complex = binding[0]
            for jj, kk in enumerate(range(len(internal_mapping_exit), len(current_complex.ase_atoms))):
                internal_mapping_exit[(ii, jj)] = kk

        # Now add spectators
        for ii, loc_info in product_spectator_binding.items():
            # No choices needed - just bring the spectator close to the specified atom in a molecule that's already in
            # the complex
            # TODO: should make sure that the target is actually in the complex already (i.e. that `loc_info[1]` is
            # already in `internal_mapping_entrance`)
            binding = make_complex(current_complex, pros[ii], internal_mapping_exit[loc_info[1]], loc_info[0])

            # Bookkeeping
            ordering.append(ii)
            current_complex = binding[0]
            for jj, kk in enumerate(range(len(internal_mapping_exit), len(current_complex.ase_atoms))):
                internal_mapping_exit[(ii, jj)] = kk

        exit_complex = current_complex
        current_exit_species = exit_complex.ase_atoms.get_chemical_symbols()
        current_exit_coords = exit_complex.ase_atoms.get_positions()
        
        species = [None] * len(current_exit_species)
        coords = np.zeros(current_exit_coords.shape)

        for key, value in new_mapping.items():
            species[key] = current_exit_species[internal_mapping_exit[value]]
            coords[key] = current_exit_coords[internal_mapping_exit[value]]

        new_exit_complex = Atoms(symbols=species, positions=coords)
        new_exit_complex.charge = exit_complex.charge
        new_exit_complex.uhf = exit_complex.uhf
        exit_complex = new_exit_complex

    if isinstance(entrance_complex, ArchMol):
        entrance_complex = entrance_complex.ase_atoms
    if isinstance(exit_complex, ArchMol):
        exit_complex = exit_complex.ase_atoms

    return entrance_complex, exit_complex