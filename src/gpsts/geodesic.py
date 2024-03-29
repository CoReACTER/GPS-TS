# Copyright (c) CoReACTER.
# Distributed under the terms of the GPL version 3.
# Some code taken/heavily influences by https://github.com/virtualzx-nad/geodesic-interpolate

# n-dimensional arrays, etc.
import numpy as np

# Molecule representation
from ase import Atoms

# For defining a reaction path geodesic using internal coordinates
from geodesic_interpolate import redistribute, Geodesic


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


def construct_geodesic_path(
    entrance_complex: Atoms,
    exit_complex: Atoms,
    nimages: int = 20,
    morse_scaling: float = 1.7,  # Default from geodesic_interpolate
    initial_tolerance: float = 1e-2,  # Default from geodesic_interpolate
    tolerance: float = 2e-3,
    friction: float = 1e-2,  # Default from geodesic_interpolate
    distance_cutoff: float = 3.0,  # Maximum distance for interaction to be included in internal coordinates
    maximum_iterations: int = 20,
    max_sweep_microiterations: int = 20
):
    """

    Construct a geodesic path linking an entrance complex to an exit complex. This method is based on the work of
    Zhu et al. (see References) and the `geodesic_interpolate` code
    (https://github.com/virtualzx-nad/geodesic-interpolate).
    
    Args:
        entrance_complex (Atoms): Entrance complex, as an ASE Atoms object
        exit_complex (Atoms): Exit complex, as an ASE Atoms object
        nimages (int): Number of points included in the geodesic path. This number must be at least 3 (default is 20).
            Note that the first image will be the `entrance_complex`, and the last image will be the `exit_complex`
        morse_scaling (float): Exponential parameter for the Morse potential
        initial_tolerance (float): Tolerance for initial generation of distributed interpolated structures (default
            is 0.02, taken from `geodesic_interpolate`) 
        tolerance (float): Convergence tolerance (default is 0.002, taken from `geodesic_interpolate`)
        friction (float): Friction term used to prevent large changes in geometry. (default is 0.01, taken from
            `geodesic_interpolate`)
        distance_cutoff (float): Distance cutoff for a bond between two atoms to be included in the coordinate system
            (default is 3A, taken from `geodesic_interpolate`)
        maximum_iterations (int): Maximum number of iterations for minimization (default is 20)
        max_sweep_microiterations (int): Maximum number of micro-iterations (default is 20, taken from `geodesic_interpolate`)

    Returns:
        path_atoms (List[Atoms]): Structures along the geodesic path, represented as ASE Atoms objects


    References:
    `Geodesic interpolation for reaction pathways`,
    J. Chem. Phys. 2019, 150(16), 164103, https://doi.org/10.1063/1.5090303.

    """

    if nimages < 3:
        raise ValueError("Number of images must be at least 3!")
    
    symbols = entrance_complex.get_chemical_symbols()
    entrance_geom = entrance_complex.get_positions()
    exit_geom = exit_complex.get_positions()

    geom = np.zeros((2, len(entrance_complex), 3))
    geom[0] = entrance_geom
    geom[1] = exit_geom

    # The following code adheres closely to __main__.py in geodesic_interpolate
    initial_path = redistribute(symbols, geom, nimages, tol=initial_tolerance)
    geodesic = Geodesic(symbols, initial_path, morse_scaling, threshold=distance_cutoff, friction=friction)

    # Sweep more efficient for larger systems
    use_sweep = len(symbols) > 35

    if use_sweep:
        path = geodesic.sweep(tol=tolerance, max_iter=maximum_iterations, micro_iter=max_sweep_microiterations)
    else:
        path = geodesic.smooth(tol=tolerance, max_iter=maximum_iterations)

    path_atoms = list()
    for image in path:
        atoms = Atoms(symbols, image)
        path_atoms.append(atoms)

    return path_atoms