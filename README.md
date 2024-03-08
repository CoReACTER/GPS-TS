# GPS-TS: Geodesic Path Search for Transition States

GPS-TS contains `gpsts`, a simple package to expore chemical reactivity. It consists of three main parts:

- Generating entrance and exit complexes, using [Architector](https://github.com/lanl/Architector) and [xTB](https://github.com/grimme-lab/xtb) to pose reacting molecules close to each other
- Generating plausible reaction pathways by interpolating along a geodesic in internal coordinate space (see [geodesic_interpolate](https://github.com/virtualzx-nad/geodesic-interpolate))
- Analyzing the energy profile along the geodesic path to identify regions for transition-state searches, using [QuAcc](https://github.com/quantum-Accelerators/quacc) as a workflow manager and the [Atomic Simulation Environment (ASE)](https://gitlab.com/ase/ase) to perform Q-Chem or ORCA calculations.

## Installation

GPS-TS requires Python version 3.9+. These installation instructions assume that you have access to pip. Note that we also recommend that you use [conda](https://docs.conda.io/en/latest/) or a related package manager to ensure a clean installation environment.

Before installing GPS-TS, you will need to install versions of ASE, QuAcc, and Architector outside of PyPI. You'll also need to install geodesic_interpolate.

To install ASE and QuAcc:

```sh
pip install --force-reinstall --no-deps https://gitlab.com/ase/ase/-/archive/master/ase-master.zip
pip install git+https://github.com/quantum-accelerators/quacc.git
```

To install Architector:

```sh
git clone https://github.com/lanl/Architector
cd Architector
git checkout Secondary_Solvation_Shell
pip install -e .
```

To install geodesic_interpolate:

```sh
git clone https://github.com/virtualzx-nad/geodesic-interpolate
cd geodesic-interpolate
pip install -e .
```

And finally, with these steps taken, you should be able to install GPS-TS:

```sh
git clone https://github.com/CoReACTER/GPS-TS
cd GPS-TS
pip install -e .
```

## Tests

GPS-TS uses [pytest](https://docs.pytest.org/en/8.0.x/) for unit testing. Testing is currently a work in progress, and specifically, functions related to our benchmark dataset (see `src/gpsts/benchmark`) and automated calculations (see `src/gpsts/quacc.py`) are not tested.

To run tests, first ensure that you have pytest installed:

```sh
pip install pytest
```

And then:

```sh
cd GPS-TS
pytest tests
```

## Documentation

TODO!

## Citation

TODO!

## License

GPS-TS is released under a GNU General Public License (GPL), version 3.0. This means that you, as a user, can copy, distribute, or modify GPS-TS, as long as your modified code is licensed under the GPL and is released publicly with installation instructions. For more information, read the LICENSE file provided in this repo!