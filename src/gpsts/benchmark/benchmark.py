import copy
import logging
from typing import Any, Dict, List

from pathlib import Path

from ase import Atoms

import jobflow as jf

from gpsts.complexes import make_complexes
from gpsts.geodesic import construct_geodesic_path
from gpsts.quacc import path_static_flow_orca, path_static_flow_qchem
from gpsts.utils import load_benchmark_data


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


PROBLEM_LABELS = [
    "Sella: 100"
]


def generate_benchmark_complexes(
    path: str | Path
) -> Dict[str, Any]:

    logging.info(f"GENERATING COMPLEXES FOR DATASET {path}")

    raw_data = load_benchmark_data(path)

    entrance_exit_complexes = dict()

    for reaction in raw_data:
        logging.info(f"\tGENERATING COMPLEX FOR REACTION {reaction['label']}")

        if reaction["label"] in PROBLEM_LABELS:
            logging.warning(f"\tSKIPPING REACTION {reaction['label']}: KNOWN PROBLEM")
        try:
            entrance_complex, exit_complex = make_complexes(
                [r.molecule for r in reaction["reactants"]],
                [p.molecule for p in reaction["products"]],
                reaction["mapping"],
                reaction["reacting_atoms_reactants"],
                reaction["reacting_atoms_products"],
                reaction["bonds_breaking"],
                reaction["bonds_forming"],
                reaction["rct_charges"],
                reaction["rct_spins"],
                reaction["pro_charges"],
                reaction["pro_spins"]
            )
            entrance_exit_complexes[reaction["label"]] = {
                "entrance_complex": entrance_complex,
                "exit_complex": exit_complex,
                "charge": sum(reaction["rct_charges"]),
                "spin": sum([x - 1 for x in reaction["rct_spins"]]) + 1, 
            }
        except ValueError:
            logging.warning(f"\tPROBLEM GENERATING COMPLEX FOR REACTION {reaction['label']}")
            continue

    return entrance_exit_complexes


def generate_geodesic_paths(
    entrance_exit_complexes: Dict[str, Any]
) -> Dict[str, Any]:

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
    opt_params: Dict[str, Any] | None = None,
    nprocs: int | None = None,
    copy_files: str | Path | List[str | Path] | None = None,
    metadata: Dict[str, Any] | None = None
) -> List[jf.Flow]:
    
    flows = list()

    for label, path in geodesic_paths.items():
        logging.info(f"GENERATING FLOW FOR REACTION {label}")
        this_meta = copy.deepcopy(metadata)

        if this_meta is None:
            this_meta = dict()

        this_meta["reaction_label"] = label

        flows.append(
            path_static_flow(
                path["path"],
                charge=path["charge"],
                spin_multiplicity=path["spin"],
                xc=xc,
                basis=basis,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks,
                opt_params=opt_params,
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
