import pytest

import numpy as np

from ase.io import read

from gpsts.geodesic import construct_geodesic_path


@pytest.fixture(scope="session")
def path_points(test_dir):
    mols = list()
    
    for i in range(20):
        mol = read(test_dir / "test_geo_path" / f"path_point{i}_0_1.xyz")
        mols.append(mol)

    return mols


def test_geodesic(path_points):
    path = construct_geodesic_path(
        path_points[0],
        path_points[-1]
    )

    assert len(path) == 20

    for ii, point in enumerate(path):
        point_pos = point.get_positions()
        ref_pos = path_points[ii].get_positions()

        # Path construction is non-deterministic
        np.testing.assert_array_almost_equal(point_pos, ref_pos, decimal=1)