"""
Example setup and run script for the 3d stimulation and long-term cooling example.

Main differences from the example 1 setup are related to geometry, BCs, wells and
gravity.
"""

import logging
from typing import Tuple

import numpy as np
import porepy as pp

import thm_utils
from ex1_main import Example1Model, Granite, Water

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LS = 150


class Example3Model(Example1Model):
    """
    This class provides the parameter specification differing from examples 1 and 2.
    """

    def _set_fields(self, params, gb):
        super()._set_fields(params, gb)
        self.scalar_scale = 1e7
        self.temperature_scale = 1e0
        self.length_scale = params["length_scale"]

        self.T_0_Kelvin = 350
        self.background_temp_C = pp.KELKIN_to_CELSIUS(self.T_0_Kelvin)

        self.initial_aperture = 2e-3 / self.length_scale
        self.export_fields.append("well")
        self.production_well_key = "production_well"
        self.gravity_on = True
        self._iteration = 0

    def create_grid(self):
        """
        Method that creates the GridBucket of a 3D domain with three fractures.
        The first fracture is the one where injection takes place and
        production takes place in the third fracture. Fractures 1 and 2 intersect.
        """
        # Define the three fractures
        n_points = 16

        # Injection
        f_1 = pp.EllipticFracture(
            np.array([10, 3.5, -3]),
            11,
            18,
            0.5,
            0,
            0,
            num_points=n_points,
        )
        f_2 = pp.EllipticFracture(
            np.array([1, 5, 1]),
            15,
            10,
            np.pi * 0,
            np.pi / 4.0,
            np.pi / 2.0,
            num_points=n_points,
        )

        # Production
        f_3 = pp.EllipticFracture(
            np.array([-13, 0, 0]),
            20,
            10,
            0.5,
            np.pi / 3,
            np.pi / 1.6,
            num_points=n_points,
        )
        self.fractures = [f_1, f_2, f_3]

        # Define the domain
        size = 50
        self.box = {
            "xmin": -size,
            "xmax": size,
            "ymin": -size,
            "ymax": size,
            "zmin": -size,
            "zmax": size,
        }
        # Make a fracture network
        self.network = pp.FractureNetwork3d(self.fractures, domain=self.box)
        # Generate the mixed-dimensional mesh
        # write_fractures_to_csv(self)
        gb = self.network.mesh(self.mesh_args)

        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self._Nd = self.gb.dim_max()

        # Tag the wells
        self._tag_well_cells()
        self.n_frac = len(gb.grids_of_dimension(self._Nd - 1))
        self._update_all_apertures(to_iterate=False)
        self._update_all_apertures()

    def _faces_to_fix(self, g):
        """
        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.
        """
        all_bf, *_ = self._domain_boundary_sides(g)
        point = np.array(
            [
                [(self.box["xmin"] + self.box["xmax"]) / 2],
                [(self.box["ymin"] + self.box["ymax"]) / 2],
                [self.box["zmax"]],
            ]
        )
        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argsort(distances)
        faces = all_bf[indexes[: self._Nd]]
        return faces

    def _tag_well_cells(self):
        """
        Tag well cells with unitary values, positive for injection cells and negative
        for production cells.
        """
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            if g.dim < self._Nd:
                point = np.array(
                    [
                        [(self.box["xmin"] + self.box["xmax"]) / 2],
                        [self.box["ymax"]],
                        [0],
                    ]
                )
                distances = pp.distances.point_pointset(point, g.cell_centers)
                indexes = np.argsort(distances)
                if d["node_number"] == 1:
                    tags[indexes[-1]] = 1  # injection
                elif d["node_number"] == 3:
                    tags[indexes[-1]] = -1  # production
                    # write_well_cell_to_csv(g, indexes[-1], self)
            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

    def _source_flow_rates(self) -> Tuple[int, int]:
        """
        The rate is given in l/s = m^3/s e-3. Length scaling also needed to convert from
        the scaled length to m.
        The values returned depend on the simulation phase.
        """
        t = self.time
        tol = 1e-10
        injection, production = 0, 0
        if t > self.phase_limits[1] + tol and t < self.phase_limits[2] + tol:
            injection = 75
            production = 0
        elif t > self.phase_limits[2] + tol:
            injection, production = 20, -20
        w = pp.MILLI * (pp.METER / self.length_scale) ** self._Nd
        return injection * w, production * w

    def _bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values imitating an anisotropic background stress regime on all
        but three faces, which are fixed to ensure a unique solution.
        """
        faces = self._faces_to_fix(g)
        bc = pp.BoundaryConditionVectorial(g, faces, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def _bc_type_scalar(self, g) -> pp.BoundaryCondition:
        """
        We prescribe the pressure value at all external boundaries.
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values_mechanics(self, g) -> np.ndarray:
        """
        Lithostatic mechanical BC values.
        """
        bc_values = np.zeros((g.dim, g.num_faces))

        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        A = g.face_areas

        # Gravity acceleration
        gravity = (
            pp.GRAVITY_ACCELERATION
            * self.rock.DENSITY
            * self._depth(g.face_centers)
            / self.scalar_scale
        )

        we, sn, bt = 2 / 3, 3 / 2, 1
        we, sn, bt = 3 / 4, 3 / 2, 1
        bc_values[0, west] = (we * gravity[west]) * A[west]
        bc_values[0, east] = -(we * gravity[east]) * A[east]
        bc_values[1, south] = (sn * gravity[south]) * A[south]
        bc_values[1, north] = -(sn * gravity[north]) * A[north]
        bc_values[2, bottom] = (bt * gravity[bottom]) * A[bottom]
        bc_values[2, top] = -(bt * gravity[top]) * A[top]

        faces = self._faces_to_fix(g)
        bc_values[:, faces] = 0

        return bc_values.ravel("F")

    def _bc_values_scalar(self, g) -> np.ndarray:
        """
        Hydrostatic pressure BC values.
        """
        # Retrieve the boundaries where values are assigned
        all_bf, *_ = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        if self.gravity_on:
            depth = self._depth(g.face_centers[:, all_bf])
        else:
            depth = np.ones(all_bf.size) * pp.KILO * pp.METER

        bc_values[all_bf] = self.fluid.hydrostatic_pressure(depth) / self.scalar_scale

        return bc_values

    def _bc_values_temperature(self, g) -> np.ndarray:
        """
        Zero perturbation from initial temperature at all boundaries.
        """
        all_bf, *_ = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        bc_values[all_bf] = self.T_0_Kelvin
        return bc_values

    def _source_mechanics(self, g) -> np.ndarray:
        """
        Gravity term.
        """
        values = np.zeros((self._Nd, g.num_cells))
        values[2] = (
            pp.GRAVITY_ACCELERATION
            * self.rock.DENSITY
            * g.cell_volumes
            * self.length_scale
            / self.scalar_scale
        )
        return values.ravel("F")

    def _source_scalar(self, g) -> np.ndarray:
        """
        Source term for the scalar equation.
        For slightly compressible flow in the present formulation, this has
        units of m^3.

        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.
        """
        injection, production = self._source_flow_rates()
        wells = (
            injection
            * g.tags["well_cells"]
            * self.time_step
            * g.tags["well_cells"].clip(min=0)
        )
        wells += (
            production
            * g.tags["well_cells"]
            * self.time_step
            * g.tags["well_cells"].clip(max=0)
        )

        return wells

    def _source_temperature(self, g) -> np.ndarray:
        """
        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.
        """
        injection, production = self._source_flow_rates()

        # Injection well
        dT_in = -70
        weight = (
            self._fluid_density(g, dT=dT_in * self.temperature_scale)
            * self.fluid.specific_heat_capacity(self.background_temp_C)
            * self.time_step
            / self.T_0_Kelvin
        )
        rhs = (
            (self.T_0_Kelvin + dT_in)
            * weight
            * injection
            * g.tags["well_cells"].clip(min=0)
        )

        # Production well during phase III
        weight = (
            self._fluid_density(g)
            * self.fluid.specific_heat_capacity(self.background_temp_C)
            * self.time_step
            / self.T_0_Kelvin
        )

        lhs = (
            weight
            * production.clip(max=0)
            * g.tags["well_cells"].clip(max=0)
            / g.cell_volumes
        )
        # Set this directly into d to avoid additional return
        d = self.gb.node_props(g)
        pp.initialize_data(g, d, self.production_well_key, {"mass_weight": lhs})
        return rhs

    def _bc_type_temperature(self, g) -> pp.BoundaryCondition:
        """
        We prescribe the temperature value at all boundaries.
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = -1e2 * pp.YEAR
        # and time step
        self.time_step = -self.time / 2

        # We use
        self.end_time = 15 * pp.YEAR
        self.max_time_step = self.end_time / 20
        self.phase_limits = [self.time, 0, 10 * pp.HOUR, self.end_time]
        self.phase_time_steps = [self.time_step, 2 / 3 * pp.HOUR, self.end_time / 35, 1]
        self.time_step_factor = 1.25

    def _depth(self, coords) -> np.ndarray:
        """
        Unscaled depth. We center the domain at 1 km below the surface.
        """
        return 1.0 * pp.KILO * pp.METER - self.length_scale * coords[2]

    def _set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature effects except in fluid density.
        """
        self.rock = Granite()
        self.rock.BULK_MODULUS = pp.params.rock.bulk_from_lame(
            self.rock.LAMBDA, self.rock.MU
        )
        self.fluid = Water()

    def _assign_discretizations(self):
        """Assign MassMatrix discretiztion to account for production well in the
        energy balance.
        """
        super()._assign_discretizations()
        discr = pp.MassMatrix(self.production_well_key)

        for g, d in self.gb:
            if g.dim == self._Nd - 1:
                d[pp.DISCRETIZATION][self.temperature_variable].update(
                    {"production_well": discr}
                )

    def _save_data(self, errors, iteration_counter):
        """
        Save displacement jumps and number of iterations for visualisation purposes.
        These are written to file and plotted against time in Figure 9.
        """
        super()._save_data(errors, iteration_counter)
        if "well_p" not in self.__dict__:
            self.well_p = np.empty((0, 2))
            self.well_T = np.empty((0, 2))

        pressures = np.zeros((1, 2))
        temperatures = np.zeros((1, 2))

        for g, d in self.gb:

            T = d[pp.STATE][self.temperature_variable] * self.temperature_scale
            p = d[pp.STATE][self.scalar_variable] * self.scalar_scale
            ind = np.nonzero(g.tags["well_cells"])
            if d["node_number"] == 1:
                pressures[0, 0] = p[ind]
                temperatures[0, 0] = T[ind]
            elif d["node_number"] == 3:
                pressures[0, 1] = p[ind]
                temperatures[0, 1] = T[ind]

        self.well_p = np.concatenate((self.well_p, pressures))
        self.well_T = np.concatenate((self.well_T, temperatures))

    def compute_fluxes(self):
        gb = self.gb
        for g, d in gb:

            pa = d[pp.PARAMETERS][self.temperature_parameter_key]
            if self._iteration > 2:
                pa["darcy_flux_0"] = pa["darcy_flux_1"].copy()
            if self._iteration > 1:
                pa["darcy_flux_1"] = pa["darcy_flux"].copy()
        for e, d in gb.edges():
            pa = d[pp.PARAMETERS][self.temperature_parameter_key]
            if self._iteration > 2:
                pa["darcy_flux_0"] = pa["darcy_flux_1"].copy()
            if self._iteration > 1:
                pa["darcy_flux_1"] = pa["darcy_flux"].copy()

        super().compute_fluxes()
        a, b, c = 1, 1, 0
        node_update, edge_update = 0, 0
        if self._iteration > 4:
            for g, d in gb:

                pa = d[pp.PARAMETERS][self.temperature_parameter_key]

                v0 = pa["darcy_flux_0"]
                v1 = pa["darcy_flux_1"]
                v2 = pa["darcy_flux"]
                pa["darcy_flux"] = (a * v2 + b * v1 + c * v0) / (a + b + c)
                node_update += np.sum(np.power(v1 - v2, 2)) / np.sum(np.power(v2, 2))
            for e, d in gb.edges():
                pa = d[pp.PARAMETERS][self.temperature_parameter_key]

                v0 = pa["darcy_flux_0"]
                v1 = pa["darcy_flux_1"]
                v2 = pa["darcy_flux"]
                pa["darcy_flux"] = (a * v2 + b * v1 + c * v0) / (a + b + c)
                edge_update += np.sum(np.power(v1 - v2, 2)) / np.sum(np.power(v2, 2))
            logger.info(
                "Smoothed fluxes by {:.2e} and edge {:.2e} at time {:.2e}".format(
                    node_update, edge_update, self.time
                )
            )

        return


if __name__ == "__main__":
    # Define mesh sizes for grid generation
    ls = 15
    mesh_size = 3
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.5 * mesh_size,
        "mesh_size_bound": 3.6 * mesh_size,
    }
    params = {
        "folder_name": "ex3",
        "nl_convergence_tol": 2e-7,
        "max_iterations": 200,
        "file_name": "thm",
        "mesh_args": mesh_args,
        "dilation_angle": np.radians(5.0),
        "dilation_angle_constant_g": 0,
        "length_scale": ls,
        "max_memory": 7e7,
        "use_umfpack": True,
    }

    pickle_file_name = params["folder_name"] + "/gb"

    m = Example3Model(params, None)
    pp.run_time_dependent_model(m, params)
    m._export_pvd()
    thm_utils.write_pickle(m.gb, pickle_file_name)
    thm_utils.write_fracture_data_txt(m)
