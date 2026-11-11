"""
3D "cbox": the 2D C-shaped bracket (fixed left wall, prescribed vertical displacement of the
top-right corner, open third-medium mouth) extruded in z.

2D -> 3D
  * rectangle [0,1.1]x[0,0.5]  ->  box [0,1.1]x[0,0.5]x[0,DEPTH]   (DEPTH default = 1.1, the 2D width)
  * left wall (all components fixed)          -> left face x = 0
  * prescribed u_y at the vertex (1.0, 0.5)   -> same u_y on the edge (1.0, 0.5, z)
  * kinematics: full 3x3 F (functional_3d), neo-Hooke as in the 2D script by default
  * ZBC=plane_strain (+ NZ=1) reproduces the 2D solution exactly -> use it to validate.
"""
import os

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         locate_dofs_topological, assemble_scalar, form)
from ufl import Measure, TestFunction, derivative

from methods import make_material, huhu_lulu_regularization, huhu_regularization
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import (make_box_mesh, dofs_on_edges, z_support_bcs,
                      solve_with_load_stepping, sweep_main)

MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "neo_hooke")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "10"))
L_LAYER = 0.1                                   # extra medium layer on the right (2D: L_layer)
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else 1.0 + L_LAYER

# model parameters of the 2D script
MU, K, GAMMA = 5 / 14, 5 / 3, 1.0e-7
TARGET = -0.75
PRM = dict(Nsteps=100, Nsteps_min=40, Nsteps_max=800, crit=-0.8, inc=1.5, inc_late=1.1,
           dec=0.5, max_attempts=10,
           max_it=int(os.environ.get("NEWTON_MAXIT", "15")),
           atol=float(os.environ.get("NEWTON_ATOL", "1e-12")),
           rtol=float(os.environ.get("NEWTON_RTOL", "1e-9")),
           stol=float(os.environ.get("NEWTON_STOL", "1e-10")),
           linesearch="bt")


def run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit, verbose=False):
    comm = MPI.COMM_WORLD
    t_start = MPI.Wtime()
    if fe_degree < 2:
        raise ValueError("Hessian regularisation needs p >= 2")
    if Nx % 10 or Ny % 5:
        raise ValueError("cbox needs Nx % 10 == 0 and Ny % 5 == 0 so that the 0.1-wide walls "
                         "coincide with mesh lines")

    Nx_tot = Nx + int(round(Nx * L_LAYER))
    Lx, Ly = 1.0 + L_LAYER, 0.5
    mesh0 = make_box_mesh(comm, Lx, Ly, DEPTH, Nx_tot, Ny, NZ, CELL_TYPE)
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    num_local_cells = mesh0.topology.index_map(tdim).size_local
    h = comm.allreduce(float(np.min(mesh0.h(tdim, np.arange(num_local_cells)))), op=MPI.MIN)

    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    u = Function(Vu, name="displacement")

    # ---- material tagging: 1 = solid bracket, 2 = third medium (same rules as 2D) ----
    tol = 1e-8
    imap = mesh0.topology.index_map(tdim)
    # locate_entities also returns ghost cells -> size the array for owned + ghost cells
    marker = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
    marker[locate_entities(mesh0, tdim, lambda x: (x[0] >= 0.1 - tol) & (x[1] >= 0.1 - tol)
                           & (x[1] <= 0.4 + tol))] = 2
    body = np.concatenate([
        locate_entities(mesh0, tdim, lambda x: x[0] <= 0.1 + tol),
        locate_entities(mesh0, tdim, lambda x: (x[1] <= 0.1 + tol) & (x[0] <= 1.0 + tol)),
        locate_entities(mesh0, tdim, lambda x: (x[1] >= 0.4 - tol) & (x[0] <= 1.0 + tol))])
    for c in body:
        if marker[c] == 0:
            marker[c] = 1
    marker[marker == 0] = 2
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                            marker[:num_local_cells])

    metadata = {"quadrature_degree": 2 * fe_degree + 2}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)

    # ---- boundary conditions ----
    left_facets = locate_entities_boundary(mesh0, fdim, lambda x: np.isclose(x[0], 0.0))
    bcl = dirichletbc(np.zeros(3, dtype=default_scalar_type),
                      locate_dofs_topological(Vu, fdim, left_facets), Vu)
    disp = Constant(mesh0, default_scalar_type(0.0))
    corner = lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 0.5)
    bct = dirichletbc(disp, dofs_on_edges(mesh0, Vu.sub(1), corner), Vu.sub(1))
    bcs = [bcl, bct] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    # ---- energy ----
    mat = make_material(MU, K, MATERIAL_MODEL)
    kin = kinematics_3d(u)
    Pi_body = psi_3d(mat, kin) * dx(1)
    Pi_m = GAMMA * psi_2_3d(mat, kin) * dx(2)
    k = default_scalar_type(k_reg)
    R_reg = (huhu_lulu_regularization(u, v, dx, k, k, 2) if reg_method == "huhu_lulu"
             else huhu_regularization(u, v, dx, k, 2))
    L = derivative(Pi_body + Pi_m, u, v) + R_reg

    dof_count = Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    emit(f"--- cbox p={fe_degree}, {Nx_tot}x{Ny}x{NZ} {CELL_TYPE}, {num_cells} cells, {dof_count} dofs, "
         f"{reg_method} k={k_reg:.1e}, {MATERIAL_MODEL}, ZBC={ZBC}, depth={DEPTH:g} ---")
    time_setup = MPI.Wtime() - t_start

    writer = VTXWriter(comm, os.path.join(out_dir, "results_u.bp"), [u], engine="BP4")
    try:
        writer.write(0.0)
        stats = solve_with_load_stepping(
            L, u, bcs, disp, TARGET, PRM, emit=emit, verbose=verbose,
            on_step=lambda n, d: writer.write(float(n)) if n % WRITE_EVERY == 0 else None)
        writer.write(float(stats["load_steps"]) + 0.5)       # final state
    finally:
        writer.close()

    potential = comm.allreduce(assemble_scalar(form(Pi_body + Pi_m)), op=MPI.SUM)
    steps = stats["load_steps"]
    return {
        "fe_degree": fe_degree, "Nx": Nx, "Ny": Ny, "Nz": NZ, "mesh_h": h,
        "dof": int(dof_count), "num_cells": int(num_cells),
        "final_displacement": stats["current"], "n_steps": steps, "potential": potential,
        "time_total": comm.allreduce(MPI.Wtime() - t_start, op=MPI.MAX),
        "time_setup": comm.allreduce(time_setup, op=MPI.MAX),
        "time_snes": comm.allreduce(stats["time_snes"], op=MPI.MAX),
        "newton_iterations": stats["newton_iterations"],
        "avg_newton_per_step": stats["newton_iterations"] / steps if steps else 0.0,
        "failed_attempts": stats["failed_attempts"],
        "successful_attempts": stats["successful_attempts"],
        "disp_pct": stats["current"] / TARGET * 100.0,
        "u": u, "Vu": Vu, "mesh": mesh0, "material_tags": material_tags,
    }


if __name__ == "__main__":
    sweep_main("cbox3d", run_case, default_meshes="20x10,30x15,40x20")