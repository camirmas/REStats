import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal

from REStats.utils import transform, inv_transform

# test functions

def test_transforms():
    m = .5
    index = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    v_df = pd.DataFrame({"v": np.linspace(0, 20, len(index))}, index=index)

    v_tf, hr_stats = transform(v_df, m)

    v_tf["y"] = v_tf.v_scaled_std
    v_inv = inv_transform(v_tf, m, hr_stats)

    assert_array_almost_equal(v_df.v, v_inv.y)