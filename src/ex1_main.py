"""
This is a setup class for solving the THM equations with contact between the fractures. 
Fluid and temperature flow in the fractures is included in the model.

The simulation is set up for four phases:
    I the system is allowed to reach equilibrium under the influence of the mechanical BCs.
    II pressure gradient from left to right is added.
    III temperature reduced at left boundary.

Compare with model formulation in the paper.

The 2d geometry is borrowed from Berge et al (2019) and extended by an additional fracture, 
which forms an X intersection with the bottom left fracture of that network:
The domain is (0, 2) x (0, 1) and contains 8 fractures, two of which form an L intersection
and two of which form an X intersection.
Most of the properties are consistent with Berge et al., see also the table in XXX.

Setup at a glance
    One horizontal fracture
    Displacement condition on the north (y = y_max) boundary
    Dirichlet values for p and T at left and right boundaries
 
            \
             \ u = [0.005, -0.002]
              V
        _____________________
        |                   |
 p=1    | ----__   ------   | p=0
 T=-100 |               ----| T=0
    100 |   X     ------    |
        |____________\______
        \\\\\\\\\\\\\\
"""
import scipy.sparse.linalg as spla
import numpy as np
import porepy as pp
from porepy.models.thm_model import THM
import logging
import os
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_grid(mesh_args):
    """
    Method that creates and returns the GridBucket of a 2D domain with seven
    fractures.
    """
    frac_pts = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.7],
            [0.8, 0.65],
            [0.2, 0.3],
            [0.6, 0.2],
            [0.2, 0.15],
            [0.7, 0.4],
            [1.0, 0.4],
            [1.7, 0.85],
            [1.5, 0.65],
            [2.0, 0.55],
            [1, 0.3],
            [1.8, 0.4],
            [1.5, 0.05],
            [1.4, 0.25],
        ]
    ).T
    frac_edges = np.array(
        [[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
    ).T

    box = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}

    network = pp.FractureNetwork2d(frac_pts, frac_edges, domain=box)
    # Generate the mixed-dimensional mesh
    gb = network.mesh(mesh_args)
    return gb, network


class Example1Model(THM):
    """
    This class provides the parameter specification of the example, including grid/geometry,
    BCs, rock and fluid parameters and time parameters. Also provides some common modelling
    functions, such as the aperture computation from the displacement jumps, and data storage
    and export functions.
    """

    def __init__(self, params, gb):
        super().__init__(params)
        # Set additional case specific fields
        self.set_fields(params, gb)

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        gb = self.gb
        self.box = {
            "xmin": 0,
            "ymin": 0,
            "xmax": 2 / self.length_scale,
            "ymax": 1 / self.length_scale,
        }
        pp.contact_conditions.set_projections(gb)

        self.Nd = gb.dim_max()
        self.n_frac = len(gb.grids_of_dimension(self.Nd - 1))

    def porosity(self, g) -> float:
        if g.dim == self.Nd:
            return 0.01
        else:
            return 1.0

    def bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        Dirichlet values at top and bottom.
        """
        all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, south + north, "dir")
        # Assign Dirichlet conditions on internal boundaries
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def bc_values_mechanics(self, g) -> np.ndarray:
        """ Dirichlet displacement on the top, fixed on bottom and 0 Neumann
        on left and right.
        """
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros((g.dim, g.num_faces))
        # Assign values on the top (or "north") side
        x = 5e-4 / self.length_scale
        y = -2e-4 / self.length_scale
        bc_values[0, north] = x
        bc_values[1, north] = y
        return bc_values.ravel("F")

    def bc_type_scalar(self, g) -> pp.BoundaryCondition:
        """
        We prescribe the pressure value at the east and west boundary
        No flow across top or bottom.
        """
        # Define boundary regions
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, east + west, "dir")

    def bc_values_scalar(self, g) -> np.ndarray:
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        if self.time > self.phase_limits[1]:
            bc_values[west] = 4e7 / self.scalar_scale
        return bc_values

    def bc_type_temperature(self, g) -> pp.BoundaryCondition:
        """
        We prescribe the temperature value at the east and west boundary.
        No heat flow across top or bottom.
        """
        # Define boundary regions
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, east + west, "dir")

    def bc_values_temperature(self, g) -> np.ndarray:
        """ Cooling on the left from the onset of phase III.
        """
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        if self.time > self.phase_limits[2]:
            bc_values[west] = -15 / self.temperature_scale
        return bc_values

    def biot_alpha(self, g) -> np.ndarray:
        if g.dim == self.Nd:
            return 0.8
        else:
            return 1.0

    def biot_beta(self, g):
        """
        For TM, the coefficient is the product of the bulk modulus (=inverse of
        the compressibility) and the volumetric thermal expansion coefficient.
        """
        if g.dim == self.Nd:
            # Factor 3 for volumetric/linear, since the pp.Granite
            # thermal expansion expansion coefficient is the linear one at 20 degrees C.
            return self.rock.BULK_MODULUS * 3 * self.rock.THERMAL_EXPANSION
        else:
            # Solution debendent coefficient computed from previous iterate,
            # see Eq. (xx)
            _, T2, _ = self._variable_increment(g, self.temperature_variable)
            T0 = 0

            T0K = self.T_0_Kelvin
            clipped_DT = np.clip(
                (T2 - T0) * self.temperature_scale, a_min=-T0K / 2, a_max=T0K / 2
            )
            return clipped_DT / T0K * self.density(g)

    def density(self, g, dp=None, dT=None) -> np.ndarray:
        """ Density computed from current pressure and temperature solution
        taken from the previous iterate.
        """
        if dp is None:
            p_0 = self.scalar_scale * self.initial_scalar(g)
            _, p_k, p_n = self._variable_increment(
                g, self.scalar_variable, self.scalar_scale,
            )
            dp = p_k - p_0
            lim = 2e7  # boundary value taken as characteristic pressure
            dp = np.clip(dp, a_min=p_n - lim, a_max=p_n + lim)
        if dT is None:
            _, T_k, T_n = self._variable_increment(
                g, self.temperature_variable, self.temperature_scale
            )
            # Assume T_0 = 0:
            dT = T_k
            lim = self.T_0_Kelvin  # T_0_kelvin taken as characteristic temperature
            dT = np.clip(dT, a_min=T_n - lim, a_max=T_n + lim)

        rho_0 = 1e3 * (pp.KILOGRAM / pp.METER ** 3) * np.ones(g.num_cells)
        rho = rho_0 * np.exp(
            dp * self.fluid.COMPRESSIBILITY - dT * self.fluid.thermal_expansion(dT)
        )

        return rho

    def scalar_temperature_coupling_coefficient(self, g) -> float:
        """
        The temperature-pressure coupling coefficient is porosity times thermal
        expansion. The pressure and
        scalar scale must be accounted for wherever this coefficient is used.
        """
        b_f = self.fluid.thermal_expansion(self.background_temp_C)
        if g.dim < self.Nd:
            coeff = -b_f
        else:
            b_s = self.rock.THERMAL_EXPANSION
            phi = self.porosity(g)
            coeff = -b_f * phi - (1 - phi) * b_s
        return coeff

    def aperture(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]["aperture"]

    def specific_volumes(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the specific volume of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["specific_volume"]
        else:
            return self.gb.node_props(g)[pp.STATE]["specific_volume"]

    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        The aperture of a fracture is 
            initial aperture - || u_n || + tan(slip_angle) || jump(u_tau) ||
        """
        gb = self.gb
        for g, d in gb:

            apertures = np.ones(g.num_cells)
            if g.dim == (self.Nd - 1):
                # Initial aperture

                apertures *= self.initial_aperture
                # Reconstruct the displacement solution on the fracture
                g_h = gb.node_neighbors(g)[0]
                data_edge = gb.edge_props((g, g_h))
                if pp.STATE in data_edge:
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge, from_iterate=to_iterate
                    )
                    # Magnitudes of normal and tangential components
                    norm_u_n = np.absolute(u_mortar_local[-1])
                    norm_u_tau = np.linalg.norm(u_mortar_local[:-1], axis=0)
                    # Add contributions
                    apertures += (
                        norm_u_n + np.tan(self._dilation_angle_constant_g) * norm_u_tau
                    )
            if to_iterate:
                pp.set_iterate(
                    d,
                    {"aperture": apertures.copy(), "specific_volume": apertures.copy()},
                )
            else:
                state = {
                    "aperture": apertures.copy(),
                    "specific_volume": apertures.copy(),
                }
                pp.set_state(d, state)

        for g, d in gb:
            parent_apertures = []
            num_parent = []
            if g.dim < (self.Nd - 1):
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    g_h = e[0]

                    if g_h == g:
                        g_h = e[1]

                    if g_h.dim == (self.Nd - 1):
                        d_h = gb.node_props(g_h)
                        if to_iterate:
                            a_h = d_h[pp.STATE][pp.ITERATE]["aperture"]
                        else:
                            a_h = d_h[pp.STATE]["aperture"]
                        a_h_face = np.abs(g_h.cell_faces) * a_h
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Assumes g_h is master
                        a_l = (
                            mg.mortar_to_slave_avg()
                            * mg.master_to_mortar_avg()
                            * a_h_face
                        )
                        parent_apertures.append(a_l)
                        num_parent.append(np.sum(mg.mortar_to_slave_int().A, axis=1))
                    else:
                        raise ValueError("Intersection points not implemented in 3d")
                parent_apertures = np.array(parent_apertures)
                num_parents = np.sum(np.array(num_parent), axis=0)

                apertures = np.sum(parent_apertures, axis=0) / num_parents

                specific_volumes = np.power(
                    apertures, self.Nd - g.dim
                )  # Could also be np.product(parent_apertures, axis=0)
                if to_iterate:
                    pp.set_iterate(
                        d,
                        {
                            "aperture": apertures.copy(),
                            "specific_volume": specific_volumes.copy(),
                        },
                    )
                else:
                    state = {
                        "aperture": apertures.copy(),
                        "specific_volume": specific_volumes.copy(),
                    }
                    pp.set_state(d, state)

        return apertures

    def set_permeability_from_aperture(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        """
        # Viscosity has units of Pa s, and is consequently divided by the scalar scale.
        viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
        gb = self.gb
        key = self.scalar_parameter_key
        for g, d in gb:
            if g.dim < self.Nd:
                # Use cubic law in fractures. First compute the unscaled
                # permeability
                apertures = self.aperture(g, from_iterate=True)
                apertures_unscaled = apertures * self.length_scale
                k = np.power(apertures_unscaled, 2) / 12 / viscosity
                d[pp.PARAMETERS][key]["perm_nu"] = k
                # Multiply with the cross-sectional area, which equals the apertures
                # for 2d fractures in 3d
                specific_volumes = self.specific_volumes(g, True)

                k = k * specific_volumes

                # Divide by fluid viscosity and scale back
                kxx = k / self.length_scale ** 2
            else:
                # Use the rock permeability in the matrix
                kxx = (
                    self.rock.PERMEABILITY
                    / viscosity
                    * np.ones(g.num_cells)
                    / self.length_scale ** 2
                )
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from the neighboring fracture g_l
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_l, _ = gb.nodes_of_edge(e)
            data_l = gb.node_props(g_l)
            a = self.aperture(g_l, True)
            V = self.specific_volumes(g_l, True)
            # We assume isotropic permeability in the fracture, i.e. the normal
            # permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.slave_to_mortar_int() * np.divide(k_s, a * V / 2)
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})

    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature dependencies of the parameters.
        """
        self.rock = Granite()
        self.fluid = Water()

    def set_mechanics_parameters(self):
        """ Mechanical parameters.
        Note that we divide the momentum balance equation by self.scalar_scale. 
        A homogeneous initial temperature is assumed.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                bc = self.bc_type_mechanics(g)
                bc_values = self.bc_values_mechanics(g)
                sources = self.source_mechanics(g)

                # In the momentum balance, the coefficient hits the scalar, and should
                # not be scaled. Same goes for the energy balance, where we divide all
                # terms by T_0, hence the term originally beta K T d(div u) / dt becomes
                # beta K d(div u) / dt = coupling_coefficient d(div u) / dt.
                coupling_coefficient = self.biot_alpha(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "source": sources,
                        "fourth_order_tensor": C,
                        "biot_alpha": coupling_coefficient,
                        "time_step": self.time_step,
                    },
                )

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    {"biot_alpha": self.biot_beta(g), "bc_values": bc_values},
                )
            elif g.dim == self.Nd - 1:

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": self._friction_coefficient(g),
                        "contact_mechanics_numerical_parameter": 1e1,
                        "dilation_angle": self._dilation_angle,
                        "time": self.time,
                    },
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. Not used as of now.
            pp.initialize_data(
                mg,
                d,
                self.mechanics_parameter_key,
                {"mu": self.rock.MU, "lambda": self.rock.LAMBDA},
            )

    def set_scalar_parameters(self):

        for g, d in self.gb:

            a = self.aperture(g)
            specific_volumes = self.specific_volumes(g)

            # Define boundary conditions for flow
            bc = self.bc_type_scalar(g)
            # Set boundary condition values
            bc_values = self.bc_values_scalar(g)

            biot_coefficient = self.biot_alpha(g)
            compressibility = self.fluid.COMPRESSIBILITY

            mass_weight = compressibility * self.porosity(g)
            if g.dim == self.Nd:
                mass_weight += (
                    biot_coefficient - self.porosity(g)
                ) / self.rock.BULK_MODULUS
            elif g.dim == 1 and (self._dilation_angle_constant_g > 1e-5):
                biot_coefficient *= 0

            mass_weight *= self.scalar_scale * specific_volumes

            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "ambient_dimension": self.Nd,
                    "source": self.source_scalar(g)
                    + self.dVdt_source(g, d, self.scalar_parameter_key),
                },
            )

            t2s_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volumes
                * self.temperature_scale
            )
            pp.initialize_data(
                g,
                d,
                self.t2s_parameter_key,
                {"mass_weight": t2s_coupling, "time_step": self.time_step},
            )
        self.set_vector_source()

        self.set_permeability_from_aperture()

    def set_vector_source(self):
        if not getattr(self, "gravity_on", False):
            return
        for g, d in self.gb:

            dp, p2, p1 = self._variable_increment(g, self.scalar_variable)
            p0 = self.initial_scalar(g)
            Dp = (p1 - p0) * self.scalar_scale
            dT, T2, T1 = self._variable_increment(g, self.temperature_variable)
            DT = T1
            grho = (
                pp.GRAVITY_ACCELERATION
                * self.density(g, Dp, DT)
                / self.scalar_scale
                * self.length_scale
            )
            gravity = -grho
            gr = np.zeros((self.Nd, g.num_cells))
            gr[self.Nd - 1, :] = gravity
            d[pp.PARAMETERS][self.scalar_parameter_key]["vector_source"] = gr.ravel("F")
        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            params_l = self.gb.node_props(g1)[pp.PARAMETERS][self.scalar_parameter_key]
            mg = data_edge["mortar_grid"]
            a = mg.slave_to_mortar_avg() * self.aperture(g1)

            grho = (
                mg.slave_to_mortar_avg()
                * params_l["vector_source"][self.Nd - 1 :: self.Nd]
            )
            gravity = grho * a / 2
            gr = np.zeros((self.Nd, mg.num_cells))
            gr[self.Nd - 1, :] = gravity

            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"vector_source": gr.ravel("F")},
            )

    def set_temperature_parameters(self):
        """ temperature parameters.
        The entire equation is divided by the initial temperature in Kelvin.
        """
        div_T_scale = self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
        kappa_f = self.fluid.thermal_conductivity() * div_T_scale
        kappa_s = self.rock.thermal_conductivity() * div_T_scale

        heat_capacity_s = (
            self.rock.specific_heat_capacity(self.background_temp_C) * self.rock.DENSITY
        )
        for g, d in self.gb:
            heat_capacity_f = self.density(g) * self.fluid.specific_heat_capacity(
                self.background_temp_C
            )
            # ()
            # Aperture and cross sectional area
            specific_volumes = self.specific_volumes(g)
            porosity = self.porosity(g)
            # Define boundary conditions for flow
            bc = self.bc_type_temperature(g)
            # Set boundary condition values
            bc_values = self.bc_values_temperature(g)
            # and source values
            biot_coefficient = self.biot_beta(g)
            if g.dim == self.Nd - 1 and (self._dilation_angle_constant_g > 1e-5):
                biot_coefficient = 0

            effective_heat_capacity = (
                porosity * heat_capacity_f + (1 - porosity) * heat_capacity_s
            )
            mass_weight = (
                effective_heat_capacity
                * specific_volumes
                * self.temperature_scale
                / self.T_0_Kelvin
            )

            effective_conductivity = porosity * kappa_f + (1 - porosity) * kappa_s
            thermal_conductivity = pp.SecondOrderTensor(
                effective_conductivity * specific_volumes
            )

            advection_weight = (
                heat_capacity_f * self.temperature_scale / self.T_0_Kelvin
            )  # "should" have a length scale, but darcy_fluxes are scaled
            pp.initialize_data(
                g,
                d,
                self.temperature_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "second_order_tensor": thermal_conductivity,
                    "advection_weight": advection_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "source": self.source_temperature(g)
                    + self.dVdt_source(g, d, self.temperature_parameter_key),
                    "ambient_dimension": self.Nd,
                },
            )

            s2t_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volumes
                * self.scalar_scale
            )
            pp.initialize_data(
                g,
                d,
                self.s2t_parameter_key,
                {"mass_weight": s2t_coupling, "time_step": self.time_step},
            )

        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a1 = self.aperture(g1)
            a_mortar = mg.slave_to_mortar_avg() * a1
            kappa_n = 2 / a_mortar * kappa_f

            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.temperature_parameter_key,
                {"normal_diffusivity": kappa_n},
            )

    def dVdt_source(self, g, d, key) -> np.ndarray:
        """ Compute the dV/dt term for intersections
        from the previous iteration and time step.
        """
        val = np.zeros(g.num_cells)

        if g.dim == self.Nd:
            return val
        if g.dim < self.Nd - 1 or not np.isclose(self._dilation_angle_constant_g, 0):
            dV, *_ = self._variable_increment(g, "specific_volume")
            val = dV * g.cell_volumes
            if key == self.temperature_parameter_key:
                val *= -self.biot_beta(g)
        return val

    def assign_discretizations(self) -> None:
        """
        For long time steps, scaling the diffusive interface fluxes in the non-default
        way turns out to actually be beneficial for the condition number.
        """
        # Call parent class for disrcetizations.
        super().assign_discretizations()

        for e, d in self.gb.edges():
            d[pp.COUPLING_DISCRETIZATION][self.temperature_coupling_term][e][
                1
            ].kinv_scaling = True
            d[pp.COUPLING_DISCRETIZATION][self.scalar_coupling_term][e][
                1
            ].kinv_scaling = True

    def prepare_simulation(self):
        self.create_grid()
        self._set_time_parameters()
        self.set_rock_and_fluid()
        self.initial_condition()
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()

        self.discretize()
        self.initialize_linear_solver()

        self.export_step()

    def before_newton_iteration(self):
        """ Rediscretize. Should the parent be updated?
        """

        # iterations for the flux computation.
        self.compute_fluxes()
        self.update_all_apertures(to_iterate=True)

        self.set_parameters()
        self.assembler.discretize(
            term_filter=["!mpsa", "!stabilization", "!div_u", "!grad_p"]
        )

    def after_newton_iteration(self, solution):
        super().after_newton_iteration(solution)
        self.solution_vector = solution

    def after_newton_convergence(self, solution, errors, iteration_counter):
        g = self.gb.grids_of_dimension(self.Nd - 1)[0]

        super().after_newton_convergence(solution, errors, iteration_counter)
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures(to_iterate=True)

        self.export_step()
        self.adjust_time_step()
        self.save_data(errors, iteration_counter)

    def set_exporter(self):
        self.exporter = pp.Exporter(
            self.gb, self.file_name, folder_name=self.viz_folder_name + "_vtu"
        )
        self.export_times = []

    def export_step(self):
        """
        Export the current solution to vtu. The method sets the desired values in d[pp.STATE].
        For some fields, it provides zeros in the dimensions where the variable is not defined,
        or pads the vector values with zeros so that they have three components, as required
        by ParaView.
        We use suffix _exp on all exported variables, to separate from scaled versions also
        stored in d.
        """
        if "exporter" not in self.__dict__:
            self.set_exporter()
        for g, d in self.gb:
            if g.dim == self.Nd:
                pad_zeros = np.zeros((3 - g.dim, g.num_cells))
                u = d[pp.STATE][self.displacement_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = u_exp
                d[pp.STATE]["traction_exp"] = np.zeros(d[pp.STATE]["u_exp"].shape)
            elif g.dim == (self.Nd - 1):
                pad_zeros = np.zeros((2 - g.dim, g.num_cells))
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))

                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, from_iterate=False
                )
                mortar_u = data_edge[pp.STATE][self.mortar_displacement_variable]
                mg = data_edge["mortar_grid"]
                displacement_jump_global_coord = (
                    mg.mortar_to_slave_avg(nd=self.Nd)
                    * mg.sign_of_mortar_sides(nd=self.Nd)
                    * mortar_u
                )
                u_mortar_global = displacement_jump_global_coord.reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u_mortar_local * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = np.vstack(
                    (u_mortar_global * self.length_scale, pad_zeros)
                )
                traction = d[pp.STATE][self.contact_traction_variable].reshape(
                    (self.Nd, -1), order="F"
                )

                d[pp.STATE]["traction_exp"] = (
                    np.vstack((traction, pad_zeros)) * self.scalar_scale
                )
            else:
                d[pp.STATE]["traction_exp"] = np.zeros((3, g.num_cells))
                u_exp = np.zeros((3, g.num_cells))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = np.zeros((3, g.num_cells))

            d[pp.STATE]["aperture_exp"] = self.aperture(g) * self.length_scale
            if np.isclose(self.time, 0):
                d[pp.STATE]["aperture_0"] = self.aperture(g) * self.length_scale
                d[pp.STATE]["u_exp_0"] = u_exp

            d[pp.STATE]["p_exp"] = d[pp.STATE][self.scalar_variable] * self.scalar_scale
            d[pp.STATE]["T_exp"] = (
                d[pp.STATE][self.temperature_variable] * self.temperature_scale
            )
        self.exporter.write_vtk(self.export_fields, time_step=self.time)
        self.export_times.append(self.time)

    def export_pvd(self):
        """
        At the end of the simulation, after the final vtu file has been exported, the
        pvd file for the whole simulation is written by calling this method.
        """
        self.exporter.write_pvd(np.array(self.export_times))

    def save_data(self, errors, iteration_counter):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """
        n = self.n_frac
        if "u_jumps_tangential" not in self.__dict__:
            self.u_jumps_tangential = np.empty((0, n))
            self.u_jumps_normal = np.empty((0, n))
            self.force_tangential = np.empty((0, n))
            self.force_normal = np.empty((0, n))
            self.iterations = []

        self.iterations.append(iteration_counter + 1)

        tangential_u_jumps = np.zeros((1, n))
        normal_u_jumps = np.zeros((1, n))
        tangential_force = np.zeros((1, n))
        normal_force = np.zeros((1, n))
        for g, d in self.gb:
            for i in np.arange(1, 4):
                if np.isclose(self.phase_limits[i], self.time):
                    for var in ["p", "T", "u", "contact_traction"]:
                        val = d[pp.STATE].get("{}".format(var, i), np.empty(0))
                        d[pp.STATE]["{}_phase_{}".format(var, i)] = val
            if g.dim == self.Nd - 1:
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, from_iterate=False
                )
                tangential_jump = np.linalg.norm(
                    u_mortar_local[:-1] * self.length_scale, axis=0
                )
                normal_jump = u_mortar_local[-1] * self.length_scale
                vol = np.sum(g.cell_volumes)
                tangential_jump_norm = (
                    np.sqrt(np.sum(tangential_jump ** 2 * g.cell_volumes)) / vol
                )
                normal_jump_norm = (
                    np.sqrt(np.sum(normal_jump ** 2 * g.cell_volumes)) / vol
                )
                ind = g.frac_num
                tangential_u_jumps[0, ind] = tangential_jump_norm
                normal_u_jumps[0, ind] = normal_jump_norm

                contact_force = d[pp.STATE][self.contact_traction_variable]

                # Pick out the tangential and normal direction of the contact force.
                # The contact force of the first cell is in the first self.dim elements
                # of the vector, second cell has the next self.dim etc.
                # By design the tangential force is the first self.dim-1 components of
                # each cell, while the normal force is the last component.
                normal_indices = np.arange(self.Nd - 1, contact_force.size, self.Nd)
                tangential_indices = np.setdiff1d(
                    np.arange(contact_force.size), normal_indices
                )
                contact_force_normal = contact_force[normal_indices]
                contact_force_tangential = contact_force[tangential_indices].reshape(
                    (self.Nd - 1, g.num_cells), order="F"
                )

                tangential_force_norm = (
                    np.sqrt(np.sum(contact_force_tangential ** 2 * g.cell_volumes))
                    / vol
                )
                tangential_force[0, ind] = tangential_force_norm
                normal_force_norm = (
                    np.sqrt(np.sum(contact_force_normal ** 2 * g.cell_volumes)) / vol
                )
                normal_force[0, ind] = normal_force_norm

        self.u_jumps_tangential = np.concatenate(
            (self.u_jumps_tangential, tangential_u_jumps)
        )
        self.u_jumps_normal = np.concatenate((self.u_jumps_normal, normal_u_jumps))
        self.force_normal = np.concatenate((self.force_normal, normal_force))
        self.force_tangential = np.concatenate(
            (self.force_tangential, tangential_force)
        )

    def initial_condition(self) -> None:
        """
        Initial value for the Darcy fluxes. TODO: Add to THM.
        """
        for g, d in self.gb:
            # d[pp.STATE] = {"previous_iterate": {}}
            d[pp.PARAMETERS] = pp.Parameters()
            d[pp.PARAMETERS].update_dictionaries(
                [
                    self.mechanics_parameter_key,
                    self.mechanics_parameter_key_from_t,
                    self.scalar_parameter_key,
                    self.temperature_parameter_key,
                ]
            )
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures()
        super().initial_condition()

        for g, d in self.gb:
            d[pp.PARAMETERS][self.temperature_parameter_key].update(
                {"darcy_flux": np.zeros(g.num_faces)}
            )
            d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
            p0 = self.initial_scalar(g)
            state = {
                self.scalar_variable: p0,
                "u_exp_0": np.zeros(g.num_cells),
                "aperture_0": self.aperture(g) * self.length_scale,
            }
            iterate = {
                self.scalar_variable: p0,
                self.temperature_variable: np.zeros(g.num_cells),
            }  # For initial flux

            pp.set_state(d, state)
            pp.set_iterate(d, iterate)

    def initial_scalar(self, g) -> np.ndarray:
        if getattr(self, "gravity_on", False):
            depth = self._depth(g.cell_centers)
        else:
            depth = np.zeros(g.num_cells)

        return self.fluid.hydrostatic_pressure(depth) / self.scalar_scale

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = self.params.get("time", -1e5)
        # and time step
        self.time_step = self.params.get("time_step", -self.time)

        # We use
        self.end_time = 10 * pp.HOUR
        self.max_time_step = self.end_time / 5
        self.phase_limits = np.array([self.time, 0, 3e1, self.end_time])
        self.phase_time_steps = np.array([self.time_step, 2e0, 2 / 3 * pp.HOUR, 1.0])

    def _friction_coefficient(self, g) -> float:
        """
        Friction coefficient for the fractures.
        Used for comparison with traction ratio, which always is dimensionless, i.e. no scaling.
        """
        return 0.5

    def adjust_time_step(self):
        """
        Adjust the time step so that smaller time steps are used when the driving forces
        are changed. Also make sure to exactly reach the start and end time for
        each phase. 
        """
        # Default is to just increase the time step somewhat
        self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step

        # We also want to make sure that we reach the end of each simulation phase
        for dt, lim in zip(self.phase_time_steps, self.phase_limits):
            diff = self.time - lim
            if diff < 0 and -diff <= self.time_step:
                self.time_step = -diff

            if np.isclose(self.time, lim):
                self.time_step = dt
        # And that the time step doesn't grow too large after the equilibration phase
        if self.time > 0:
            self.time_step = min(self.time_step, self.max_time_step)

    def set_fields(self, params, gb=None):
        """
        Set various fields to be used in the model.
        """
        # We operate on the temperature difference T-T_0
        self.T_0 = 0

        self.T_0_Kelvin = 300
        self.background_temp_C = self.T_0_Kelvin - 273
        if gb is not None:
            self.gb = gb

        # Scaling coefficients
        self.scalar_scale = 1e9

        self.file_name = self.params["file_name"]
        self.folder_name = self.params["folder_name"]
        # Keywords
        self.mechanics_parameter_key_from_t = "mechanics_from_t"
        # Temperature
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable

        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        self.export_fields = [
            "u_exp",
            "p_exp",
            "T_exp",
            "traction_exp",
            "aperture_exp",
            "u_global",
            "cell_centers",
        ]
        # Initial aperture, a_0
        self.initial_aperture = 5e-4 / self.length_scale

        # Dilation angle
        self._dilation_angle = params.get("dilation_angle")
        self._dilation_angle_constant_g = params.get("dilation_angle_constant_g")
        self.full_g_coupling = params.get("full_g_coupling", True)

        self.mesh_args = params.get("mesh_args", None)

    def solve_direct(self, A, b):
        use_umfpack = self.params.get("use_umfpack", True)

        if use_umfpack:
            A.indices = A.indices.astype(np.int64)
            A.indptr = A.indptr.astype(np.int64)
        x = spla.spsolve(A, b, use_umfpack=use_umfpack)
        return x, use_umfpack

    def _variable_increment(self, g, variable, scale=1, x0=None):
        """ Extracts the variable solution of the current and previous time step and
        computes the increment.
        """
        d = self.gb.node_props(g)
        if x0 is None:
            x0 = d[pp.STATE][variable] * scale
        x1 = d[pp.STATE][pp.ITERATE][variable] * scale
        dx = x1 - x0
        return dx, x1, x0


class Water:
    """
    Fluid phase.
    """

    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref
        self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
        self.COMPRESSIBILITY = 1e-10 / pp.PASCAL
        self.BULK_MODULUS = 1 / self.COMPRESSIBILITY

    def thermal_expansion(self, delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return 4e-4

    def thermal_conductivity(self, theta=None):  # theta in CELSIUS
        """ Units: W / m K """
        if theta is None:
            theta = self.theta_ref
        return 0.6

    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        """ Units: J / kg K """
        return 4200

    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        return 0.001

    def hydrostatic_pressure(self, depth, theta=None):
        rho = 1e3 * (pp.KILOGRAM / pp.METER ** 3)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Granite(pp.Granite):
    """
    Solid phase.
    """

    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)
        self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)

        self.PERMEABILITY = 1e-15

    def thermal_conductivity(self, theta=None):
        return 3.0

    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        c_ref = 790.0
        return c_ref


if __name__ == "__main__":
    params = {
        "nl_convergence_tol": 1e-8,
        "max_iterations": 20,
        "file_name": "thm",
        "dilation_angle": np.radians(5),
        "dilation_angle_constant_g": 0,
    }

    root_folder = "ex1temp/"

    gmsh_file_base = root_folder + "mesh_"
    pickle_file_name = root_folder + "gblist"
    folder_base = root_folder + "convergence_study_"
    gb_list, models = [], []
    refinement_levels = np.arange(0, 5)
    refinement_levels = [0, 1, 2, 3]

    if refinement_levels[0] > 3:
        gb_list = utils.read_pickle(pickle_file_name)
        if gb_list is None:
            gb_list = []

    for i in refinement_levels:
        # Set file names for saving of results etc.
        folder_name = folder_base + str(i)
        params["folder_name"] = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create grid bucket from existing mesh files for reproducability
        msh_file = gmsh_file_base + str(i) + ".msh"
        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(file_name=msh_file)
        gb = pp.fracs.meshing.grid_list_to_grid_bucket(grid_list)

        m = Example1Model(params, gb)
        gb_list.append(gb)

        pp.run_time_dependent_model(m, params)

        # Export and save data
        m.export_pvd()
        utils.write_fracture_data_txt(m)
        utils.write_pickle(gb_list, pickle_file_name)
    # Compute grid mappings for error computation
    if len(gb_list) > 1:
        for gb in gb_list[:-1]:
            pp.grids.match_grids.gb_refinement(gb, gb_list[-1])
    utils.write_pickle(gb_list, pickle_file_name)
