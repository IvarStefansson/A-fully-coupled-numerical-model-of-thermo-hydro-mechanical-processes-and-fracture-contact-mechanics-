"""
Example 2. Run same scenario as in Example 1, but with the two simplified models
M_0 and M_1.
"""

import numpy as np
import porepy as pp
import logging
import os
import utils

logging.basicConfig(level=logging.INFO)
from ex1_main import Example1Model

if __name__ == "__main__":
    params = {
        "nl_convergence_tol": 1e-8,
        "max_iterations": 20,
        "file_name": "thm",
        "dilation_angle": 0,
        "dilation_angle_constant_g": 0,
        "full_g_coupling": False,
    }
    gmsh_file_base = "ex1/mesh_"
    folder_base = "ex2/"
    refinement = 4
    msh_file = gmsh_file_base + str(refinement) + ".msh"
    model_names = ["no_dilation_", "weak_dilation_coupling_"]
    for model in model_names:
        pickle_file_name = folder_base + "gblist_" + model + str(refinement)
        folder_name = folder_base + model + str(refinement)
        params["folder_name"] = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # Create grid bucket from mesh file for reproducability
        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
        gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)

        # Set up model and run simulation
        if model is "weak_dilation_coupling_":
            params["dilation_angle_constant_g"] = np.radians(5)

        m = Example1Model(params, gb)
        pp.run_time_dependent_model(m, params)

        # Save data
        m.export_pvd()
        utils.write_fracture_data_txt(m)
        utils.write_pickle([gb], pickle_file_name)
