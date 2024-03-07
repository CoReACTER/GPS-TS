# Copyright (c) CoReACTER.
# Distributed under the terms of the GPL version 3.

#stdlib
import copy
from typing import Any, List, Dict, Optional, Tuple, Union
from pathlib import Path

# Ensuring that key libraries are installed
from monty.dev import requires

# Molecule representations
from ase import Atoms

# For ORCA and Q-Chem calculations
from quacc.recipes.orca.core import static_job as static_job_orca
from quacc.recipes.qchem.core import static_job as static_job_qchem
from sella import Sella

import jobflow as jf


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


@requires(Sella, "Sella must be installed. Refer to the quacc documentation.")
def path_static_flow_orca(
    path_points: List[Atoms],
    charge: int,
    spin_multiplicity: int,
    xc: str = "wb97m-v",
    basis: str = "ma-def2-svp",
    orcasimpleinput: List[str] | None = None,
    orcablocks: List[str] | None = None,
    nprocs: int | None = None,
    copy_files: str | Path | List[str | Path] | None = None,
    base_name: str = "Geodesic path statics",
    metadata: Dict[str, Any] | None = None
):

    """

    Generate a JobFlow Flow consisting of single-point calculations for each point along a path.
    These points are linked by metadata for ease of analysis.

    This function only works with the ORCA electronic structure code; for use with Q-Chem, use
    `path_static_flow_qchem`.

    TODO: should this function, or something like it, be in QuAcc rather than this repo?

    Args:
        path_points (List[Atoms]): Collection of structures (as ASE Atoms objects) that define a reaction path
        charge (int): Charge for all structures in `path_points`
        spin_multiplicity (int): Spin multiplicity for all structures in `path_points`; note that we are here
            implicitly assuming that there are no spin transitions along a reaction pathway, which may not always
            be a good assumption for reactions involving e.g. transition-metal coordination complexes
        xc (str): Exchange-correlation functional (default is "wb97m-v", a range-separated hybrid meta-GGA functional
            from Mardirossian and Head-Gordon [1])
        basis (str): Basis set (default is "ma-def2-svp", the "minimally augmented" split-valence basis set based on
            the def2-SVP Karlsruhe basis set [2])
        orcasimpleinput (List[str]): List of ORCA "simple input" commands. For instance, include "SlowConv" to use
            parameters for more challenging SCF optimization cases, or "RIJCOSX" to use a linear scaling approximation.
            Default is None
        orcablocks (List[str]): List of "blocks" for an ORCA input file. For instance, '%scf convergence Tight end'
            could be included to specify that SCF optimization should use more stringent convergence criteria. Default
            is None
        nprocs (int): Number of processes to be used for this calculation. Default is None, meaning that one process
            will be used for each core on the machine where this calculation will be run
        copy_files (str | Path | List[str | Path] | None): Files to copy (and decompress) from source to the runtime
            directory. 
        base_name (str): Prefix to be used for all calculation names. Default is "Geodesic path statics"
        metadata (Dict[str, Any]): Dictionary of metadata to be passed to JobFlow (default is None)

    Returns:
        flow (jf.Flow): JobFlow Flow object containing static ORCA jobs for each point along
            `path_points`

    References:
        [1] `ωB97M-V: A combinatorially optimized, range-separated hybrid, meta-GGA density functional with VV10 nonlocal
        correlation`, J. Chem. Phys. 2016, 144(21), 214110, https://doi.org/10.1063/1.4952647.
        [2] `Minimally augmented Karlsruhe basis sets`,
        Theor. Chem. Acc. 2011, 128, 295-205, https://doi.org/10.1007/s00214-010-0846-z.

    """

    jobs = list()
    for ii, point in enumerate(path_points):
        point_job = static_job_orca(
            point,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            xc=xc,
            basis=basis,
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
            nprocs=nprocs,
            copy_files=copy_files,
        )
        point_job.name = f"{base_name}: path_static {ii}"

        if metadata is None:
            point_job.update_metadata({
                "workflow": "geodesic_path_ts",
                "subflow": "geodesic_static",
                "point": ii
            })
        else:
            this_meta = copy.deepcopy(metadata)
            this_meta.update(
                {
                    "workflow": "geodesic_path_ts",
                    "subflow": "geodesic_static",
                    "point": ii
                }
            )
            point_job.update_metadata(this_meta)

        jobs.append(point_job)

    flow = jf.Flow(jobs)
    return flow


@requires(Sella, "Sella must be installed. Refer to the quacc documentation.")
def path_static_flow_orca(
    path_points: List[Atoms],
    charge: int,
    spin_multiplicity: int,
    method: str = "wb97m-v",
    basis: str = "def2-svpd",
    copy_files: str | Path | List[str | Path] | None = None,
    base_name: str = "Geodesic path statics",
    metadata: Dict[str, Any] | None = None,
    **calc_kwargs
):

    """

    Generate a JobFlow Flow consisting of single-point calculations for each point along a path.
    These points are linked by metadata for ease of analysis.

    This function only works with the Q-Chem electronic structure code; for use with ORCA, use
    `path_static_flow_qchem`.

    TODO: should this function, or something like it, be in QuAcc rather than this repo?

    Args:
        path_points (List[Atoms]): Collection of structures (as ASE Atoms objects) that define a reaction path
        charge (int): Charge for all structures in `path_points`
        spin_multiplicity (int): Spin multiplicity for all structures in `path_points`; note that we are here
            implicitly assuming that there are no spin transitions along a reaction pathway, which may not always
            be a good assumption for reactions involving e.g. transition-metal coordination complexes
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

    jobs = list()
    static_results = list()
    for ii, point in enumerate(path_points):
        point_job = static_job_qchem(
            point,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            method=method,
            basis=basis,
            copy_files=copy_files,
            **calc_kwargs
        )
        point_job.name = f"{base_name}: path_static {ii}"

        if metadata is None:
            point_job.update_metadata({
                "workflow": "geodesic_path_ts",
                "subflow": "geodesic_static",
                "point": ii
            })
        else:
            this_meta = copy.deepcopy(metadata)
            this_meta.update(
                {
                    "workflow": "geodesic_path_ts",
                    "subflow": "geodesic_static",
                    "point": ii
                }
            )
            point_job.update_metadata(this_meta)

        jobs.append(point_job)

    flow = jf.Flow(jobs)
    return flow