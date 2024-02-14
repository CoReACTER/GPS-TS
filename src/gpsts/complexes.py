# Copyright (c) CoReACTER.
# Distributed under the terms of the GPL version 3.

#stdlib
from typing import Any, List, Dict, Optional, Tuple, Union
import random

# For reaction complex formation
from architector import view_structures, convert_io_molecule
import architector.io_arch_dock as io_arch_dock
from architector.io_molecule import Molecule as ArchMol

# For molecule representations
from ase import Atoms
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

# Basic numeric/scientific python libraries
import numpy as np


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
    
    params['species_list'] = [other]
    params['species_location_method'] = 'targeted'

    # TODO: make it possible to add multiple attachment sites
    params['targeted_indices_close'] = [reacting_atom_core, reacting_atom_other]

    binding = io_arch_dock.add_non_covbound_species(core, parameters=params)
    
    return binding


def select_central_molecule(
    molecules: List[ArchMol], 
    reacting_atoms: Dict[int, List[int]]
):
    # Pick one molecule to be the center
    # Arbitrarily, we choose the molecule with the most reacting atoms
    # If there's a tie, we choose the one that's largest
    central_index = None
    central_mol = None
    for ii, mol in enumerate(molecules):
        react_atoms = reacting_atoms.get(ii)
        if react_atoms is None:
            raise ValueError("Need to provide list of reacting atoms for all molecules!"
                             "Format of reacting_atoms: {i: [ind_1, ind_2,..., ind_n]},"
                             "where 'i' is the 0-based index of the molecule in `molecules`"
                             "and 'ind_n' is the nth reacting atom, again using 0-based indices")

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
    rct_charges: Dict[int, int] = dict(),
    rct_spins: Dict[int, int] = dict(),
    pro_charges: Dict[int, int] = dict(),
    pro_spins: Dict[int, int] = dict(),
    reactant_core: Optional[int] = None,
    product_core: Optional[int] = None,
    architector_params: Dict = dict()
):

    # TODO: are we managing charge and spin effectively?

    # Convert reactants and products to ase Atoms objects
    rcts = list()
    for ir, r in enumerate(reactants):
        if isinstance(r, Molecule):
            ratoms = AseAtomsAdaptor.get_atoms(r)
            ratoms = convert_io_molecule(ratoms)
        else:
            ratoms = convert_io_molecule(r)

        ratoms.charge = int(rct_charges.get(ir, 0))
        ratoms.uhf = int(rct_spins.get(ir, 1) - 1)
        rcts.append(ratoms)

    pros = list()
    for ip, p in enumerate(products):
        if isinstance(p, Molecule):
            patoms = AseAtomsAdaptor.get_atoms(p)
            patoms = convert_io_molecule(patoms)
        else:
            patoms = convert_io_molecule(p)

        patoms.charge = int(pro_charges.get(ip, 0))
        patoms.uhf = int(pro_spins.get(ip, 1) - 1)
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

            possible_bonds = [
                e
                for e in bonds_breaking + bonds_forming
                if (e[0][0] in ordering and e[1][0] == ii)
                or (e[1][0] in ordering and e[0][0] == ii)
            ]

            if len(possible_bonds) == 0:
                raise ValueError(f"Cannot add molecule {ii}; no place to bind on molecules {ordering}."
                                 "Check `reacting_atoms_reactants`, `bonds_breaking`, and `bonds_forming`."
                                 "If all variables are correct, try manually selecting a core.")

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

            possible_bonds = [
                (mapping[e[0]], mapping[e[1]])
                for e in bonds_breaking + bonds_forming
                if (mapping[e[0]][0] in ordering and mapping[e[1]][0] == ii)
                or (mapping[e[1]][0] in ordering and mapping[e[0]][0] == ii)
            ]

            if len(possible_bonds) == 0:
                raise ValueError(f"Cannot add molecule {ii}; no place to bind on molecules {ordering}."
                                 "Check `reacting_atoms_products`, `bonds_breaking`, and `bonds_forming`."
                                 "If all variables are correct, try manually selecting a core.")

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