@requires(Sella, "Sella must be installed. Refer to the quacc documentation.")
def geodesic_path_flow(
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
    base_name: str = "Geodesic path + TS optimization",
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