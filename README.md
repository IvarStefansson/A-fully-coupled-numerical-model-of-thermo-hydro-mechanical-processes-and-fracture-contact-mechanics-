# A-fully-coupled-numerical-model-of-thermo-hydro-mechanical-processes-and-fracture-contact-mechanics-
Run scripts for the paper A fully coupled numerical model of thermo-hydro-mechanical processes and fracture contact mechanics in porous media by Ivar Stefansson, Inga Berre and Eirik Keilegavlen (all University of Bergen, Norway).

The simulations were performed using version 1.4.2 of PorePy corresponding to commit 110300c5900327e46a99dc77edada89b9b05081b, see https://github.com/pmgbergen/porepy/.

The movie found in the visualisation subfolder displays results for the second phase of Example 3 for each time step.
The results are:
	Top left: Fracture pressure perturbation and matrix cells satisfying ||p-p_H|| > 5e5 Pa.
	Bottom left: Fracture temperature perturbation and matrix cells satisfying T-T_0 < 0 K.
	Top right: Aperture.
	Bottom left: Tangential displacement jump increment for each time step.