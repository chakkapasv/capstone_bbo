import numpy as np
import pandas as pd

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

def suggest_next_point_hebo(X_past, y_past, dimension):
    """
    X_past: 2D numpy array (shape: n_samples, dimension)
    y_past: 1D numpy array (shape: n_samples,). Pass -y if MAXIMIZING.
    """
    # 1. Define Design Space (variables x0, x1, ... x_{dim-1})
    space_config = [
        {"name": f"x{i}", "type": "num", "lb": 0.0, "ub": 1.0}
        for i in range(dimension)
    ]
    space = DesignSpace().parse(space_config)

    # 2. Instantiate HEBO Optimizer
    opt = HEBO(space)

    # 3. Format historical data as Pandas DataFrames & NumPy arrays
    col_names = [f"x{i}" for i in range(dimension)]
    X_df = pd.DataFrame(X_past, columns=col_names)
    y_arr = np.array(y_past).reshape(-1, 1)

    # 4. Feed all past observations into HEBO
    opt.observe(X_df, y_arr)

    # 5. Suggest the next point
    rec_df = opt.suggest(n_suggestions=1)

    # 6. Extract raw float vector (matching skopt output format)
    next_point = rec_df.values[0].tolist()

    return next_point
