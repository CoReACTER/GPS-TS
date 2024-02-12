# Molecule representation
from ase import Atoms

# For defining a reaction path geodesic using internal coordinates
from geodesic_interpolate import redistribute, Geodesic


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
        atoms.charge = entrance_complex.charge
        atoms.uhf = entrance_complex.uhf

        path_atoms.append(atoms)

    return path_atoms