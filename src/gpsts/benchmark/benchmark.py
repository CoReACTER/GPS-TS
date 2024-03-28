# stdlib
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

# Molecule representation
from ase import Atoms

# Calculation workflows
import jobflow as jf

# Internal imports
from gpsts.complexes import make_complexes
from gpsts.geodesic import construct_geodesic_path
from gpsts.quacc import path_static_flow_orca, path_static_flow_qchem
from gpsts.utils import load_benchmark_data


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


# EWCSS: These reactions cause my ThinkPad L14 Gen 3 on EndeavourOS Linux (kernel 6.6.20-1-lts) to crash
# Some problem with atom mapping - for now, not really worth me digging into
PROBLEM_LABELS = [
    "Sella: 100",
    "Sella: 181",
]


def generate_benchmark_complexes(
    path: str | Path
) -> Dict[str, Any]:

    """
    Generate entrance and exit complexes for a set of benchmark reactions

    Args:
        path (str | Path): Path to benchmark reactions to be processed

    Returns:
        entrance_exit_complexes (Dict[str, Any]): Key-value pairs, where they keys are reaction labels, and the values
            are dictionaries containing the reaction entrance and exit complexes, overall charge, and overall spin
            multiplicity
    """

    logging.info(f"GENERATING COMPLEXES FOR DATASET {path}")

    raw_data = load_benchmark_data(path)

    entrance_exit_complexes = dict()

    for reaction in raw_data:
        logging.info(f"\tGENERATING COMPLEX FOR REACTION {reaction['label']}")

        if reaction["label"] in PROBLEM_LABELS:
            logging.warning(f"\tSKIPPING REACTION {reaction['label']}: KNOWN PROBLEM")
            continue
        try:
            entrance_complex, exit_complex = make_complexes(
                [r.molecule for r in reaction["reactants"]],
                [p.molecule for p in reaction["products"]],
                reaction["mapping"],
                reaction["reacting_atoms_reactants"],
                reaction["reacting_atoms_products"],
                reaction["bonds_breaking"],
                reaction["bonds_forming"],
                reactant_charges=reaction["rct_charges"],
                reactant_spins=reaction["rct_spins"],
                product_charges=reaction["pro_charges"],
                product_spins=reaction["pro_spins"]
            )
            entrance_exit_complexes[reaction["label"]] = {
                "entrance_complex": entrance_complex,
                "exit_complex": exit_complex,
                "charge": sum(list(reaction["rct_charges"].values())),
                "spin": sum([x - 1 for x in list(reaction["rct_spins"].values())]) + 1, 
            }
        except ValueError:
            logging.warning(f"\tPROBLEM GENERATING COMPLEX FOR REACTION {reaction['label']}")
            continue

    return entrance_exit_complexes


def generate_geodesic_paths(
    entrance_exit_complexes: Dict[str, Any]
) -> Dict[str, Any]:

    """

    From entrance and exit complexes, generate geodesic paths for a collection of reactions

    Args:
        entrance_exit_complexes: Key-value pairs, where the keys are reaction labels and the values are
            dictionaries containing data about the reaction entrance and exit complexes; see
            `gpsts.benchmark.benchmark.generate_benchmark_complexes`

    Returns:
        geodesic_paths: Key-value pairs, where the keys are reaction labels and the values are dictionaries containing
            a list of points along a geodesic path from entrance complex to exit complex as well as the overall charge
            and spin multiplicity of the reaction

    """

    geodesic_paths = dict()

    for label, complexes in entrance_exit_complexes.items():
        logging.info(f"GENERATING PATH FOR REACTION {label}")
        geodesic_paths[label] = {
            "path": construct_geodesic_path(
                complexes["entrance_complex"],
                complexes["exit_complex"]
            ),
            "charge": complexes["charge"],
            "spin": complexes["spin"],
        }

    return geodesic_paths


def generate_path_flows_orca(
    geodesic_paths: Dict[str, Any],
    xc: str = "wb97m-v",
    basis: str = "ma-def2-svp",
    orcasimpleinput: List[str] | None = None,
    orcablocks: List[str] | None = None,
    nprocs: int | None = None,
    copy_files: str | Path | List[str | Path] | None = None,
    metadata: Dict[str, Any] | None = None
) -> List[jf.Flow]:
    
    """

    For each reaction in a collection, generate workflows to calculate the energy and forces of each point along a
    geodesic path linking reaction entrance and exit complexes.

    Args:
        geodesic_paths (Dict[str, Any]): Key-value pairs, where the keys are reaction labels and the values are
            dictionaries containing a list of points along a geodesic path from entrance complex to exit complex as
            well as the overall charge and spin multiplicity of the reaction; see
            `gpsts.benchmark.benchmark.generate_geodesic_paths`
        xc (str): Exchange-correlation functional (default is "wb97m-v", a range-separated hybrid meta-GGA functional
            from Mardirossian and Head-Gordon [1])
        basis (str): Basis set (default is "ma-def2-svp", the "minimally augmented" split-valence basis set based on
            the def2-SVP Karlsruhe basis set [2])
        orcasimpleinput (List[str] | None): List of ORCA "simple input" commands. For instance, include "SlowConv" to use
            parameters for more challenging SCF optimization cases, or "RIJCOSX" to use a linear scaling approximation.
            Default is None
        orcablocks (List[str] | None): List of "blocks" for an ORCA input file. For instance, '%scf convergence Tight end'
            could be included to specify that SCF optimization should use more stringent convergence criteria. Default
            is None
        nprocs (int): Number of processes to be used for this calculation. Default is None, meaning that one process
            will be used for each core on the machine where this calculation will be run
        copy_files (str | Path | List[str | Path] | None): Files to copy (and decompress) from source to the runtime
            directory. 
        metadata (Dict[str, Any]): Dictionary of metadata to be passed to JobFlow (default is None)

    Returns:
        flows (List[jf.Flow]): List of workflows, one for each reaction in the input collection `geodesic_paths`

    References:
        [1] `ωB97M-V: A combinatorially optimized, range-separated hybrid, meta-GGA density functional with VV10 nonlocal
        correlation`, J. Chem. Phys. 2016, 144(21), 214110, https://doi.org/10.1063/1.4952647.
        [2] `Minimally augmented Karlsruhe basis sets`,
        Theor. Chem. Acc. 2011, 128, 295-205, https://doi.org/10.1007/s00214-010-0846-z.

    """
    
    flows = list()

    for label, path in geodesic_paths.items():
        logging.info(f"GENERATING FLOW FOR REACTION {label}")
        this_meta = copy.deepcopy(metadata)

        if this_meta is None:
            this_meta = dict()

        this_meta["reaction_label"] = label

        flows.append(
            path_static_flow_orca(
                path["path"],
                charge=path["charge"],
                spin_multiplicity=path["spin"],
                xc=xc,
                basis=basis,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks,
                nprocs=nprocs,
                copy_files=copy_files,
                base_name=label,
                metadata=metadata,
            )
        )

    return flows


def generate_path_flows_qchem(
    geodesic_paths: Dict[str, Any],
    method: str = "wb97m-v",
    basis: str = "def2-svpd",
    copy_files: str | Path | List[str | Path] | None = None,
    metadata: Dict[str, Any] | None = None,
    **calc_kwargs
) -> List[jf.Flow]:

    """

    For each reaction in a collection, generate workflows to calculate the energy and forces of each point along a
    geodesic path linking reaction entrance and exit complexes.

    Args:
        geodesic_paths (Dict[str, Any]): Key-value pairs, where the keys are reaction labels and the values are
            dictionaries containing a list of points along a geodesic path from entrance complex to exit complex as
            well as the overall charge and spin multiplicity of the reaction; see
            `gpsts.benchmark.benchmark.generate_geodesic_paths`
        method (str): Exchange-correlation functional or other electronic structure method (default is "wb97m-v", a
            range-separated hybrid meta-GGA functional from Mardirossian and Head-Gordon [1])
        basis (str): Basis set (default is "def2-svpd", a split-valence basis set based on the def2-SVP Karlsruhe
            basis set that includes diffuse functions [2])
        copy_files (str | Path | List[str | Path] | None): Files to copy (and decompress) from source to the runtime
            directory. 
        base_name (str): Prefix to be used for all calculation names. Default is "Geodesic path statics"
        metadata (Dict[str, Any]): Dictionary of metadata to be passed to JobFlow (default is None)
        calc_kwargs (Dict): Dictionary of additional calculation parameters, to be passed to QuAcc's Q-Chem calculator;
            see QuAcc documentation
            (https://quantum-accelerators.github.io/quacc/reference/quacc/recipes/qchem/core.html)

    Returns:
        flow (jf.Flow): JobFlow Flow object containing static Q-Chem jobs for each point along
            `path_points`

    References:
        [1] `ωB97M-V: A combinatorially optimized, range-separated hybrid, meta-GGA density functional with VV10 nonlocal
        correlation`, J. Chem. Phys. 2016, 144(21), 214110, https://doi.org/10.1063/1.4952647.
        [2] `Property-optimized Gaussian basis sets for molecular response calculations`,
        J. Chem. Phys. 2010, 133(13), 134105, https://doi.org/10.1063/1.3484283.

    """
    
    flows = list()

    for label, path in geodesic_paths.items():
        logging.info(f"GENERATING FLOW FOR REACTION {label}")
        this_meta = copy.deepcopy(metadata)

        if this_meta is None:
            this_meta = dict()

        this_meta["reaction_label"] = label

        flows.append(
            path_static_flow_qchem(
                path["path"],
                charge=path["charge"],
                spin_multiplicity=path["spin"],
                method=method,
                basis=basis,
                copy_files=copy_files,
                base_name=label,
                metadata=metadata,
                **calc_kwargs
            )
        )

    return flows
