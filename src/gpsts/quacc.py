# Copyright (c) CoReACTER.
# Distributed under the terms of the GPL version 3.

#stdlib
from typing import Any, List, Dict, Optional, Tuple, Union
from pathlib import Path

from monty.dev import requires

# Molecule representations
from ase import Atoms

# For ORCA calculations with Sella optimizer
from quacc.recipes.orca.core import static_job, ase_relax_job
from sella import Sella

import jobflow as jf


__author__ = "Evan Spotte-Smith"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "February 2024"


@requires(Sella, "Sella must be installed. Refer to the quacc documentation.")
def path_static_flow(
    path_points: List[Atoms],
    charge: int = 0,
    spin_multiplicity: int = 1,
    xc: str = "wb97m-v",
    basis: str = "ma-def2-svp",
    orcasimpleinput: List[str] | None = None,
    orcablocks: List[str] | None = None,
    opt_params: Dict[str, Any] | None = None,
    nprocs: int | None = None,
    copy_files: str | Path | List[str | Path] | None = None,
    base_name: str = "Geodesic path statics",
    metadata: Dict[str, Any] | None = None
):

    jobs = list()
    static_results = list()
    for ii, point in enumerate(path_points):
        point_job = static_job(
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
        static_results.append(point_job.output)

    flow = jf.Flow(jobs)
    return flow