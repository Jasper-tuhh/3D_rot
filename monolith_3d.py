"""
3D "monolith": the 2D rectangular frame with an internal third-medium cavity (box_mon),
extruded in z.

2D -> 3D
  * rectangle [0,2]x[0,0.5]  ->  box [0,2]x[0,0.5]x[0,DEPTH]     (DEPTH default = 2.0, the 2D width)
  * left corner (all components fixed)   -> edge (0, 0, z)
  * right corner, u_y = 0                -> edge (2, 0, z)
  * top midpoint, prescribed u_y         -> edge (1, 0.5, z)
  * full 3x3 kinematics (functional_3d); material model default 'hartmann_neff' as in 2D
  * ZBC=plane_strain (+ NZ=1) reproduces the 2D solution exactly -> use it to validate.
"""
import os

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         assemble_scalar, form)
from ufl import Measure, TestFunction, derivative

from methods import make_material, huhu_lulu_regularization, huhu_regularization
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import (make_box_mesh, dofs_on_edges, z_support_bcs,
                      solve_with_load_stepping, sweep_main)

MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "hartmann_neff")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "10"))
LX, LY = 2.0, 0.5
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else LX

# model parameters of the 2D script
MU, K, GAMMA = 10.0, 20.0, 1.0e-7
TARGET = -1.0
PRM = dict(Nsteps=100, Nsteps_min=10, Nsteps_max=800, crit=-0.8, inc=1.5, inc_late=1.1,
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
    if Nx % 20 or Ny % 5:
        raise ValueError("monolith needs Nx % 20 == 0 and Ny % 5 == 0 so that the 0.1-wide "
                         "walls coincide with mesh lines")

    mesh0 = make_box_mesh(comm, LX, LY, DEPTH, Nx, Ny, NZ, CELL_TYPE)
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    num_local_cells = mesh0.topology.index_map(tdim).size_local
    h = comm.allreduce(float(np.min(mesh0.h(tdim, np.arange(num_local_cells)))), op=MPI.MIN)

    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    u = Function(Vu, name="displacement")

    # ---- material tagging: 1 = frame, 3 = third medium (same rules as 2D) ----
    tol = 1e-8
    imap = mesh0.topology.index_map(tdim)
    # locate_entities also returns ghost cells -> size the array for owned + ghost cells
    marker = np.zeros(imap.size_local + imap.num_ghosts, dtype=np.int32)
    marker[locate_entities(mesh0, tdim, lambda x: (x[0] >= 0.1 - tol) & (x[0] <= LX - 0.1 + tol)
                           & (x[1] >= 0.1 - tol) & (x[1] <= 0.4 + tol))] = 3
    body = locate_entities(mesh0, tdim, lambda x: (x[0] <= 0.1 + tol) | (x[0] >= LX - 0.1 - tol)
                           | (x[1] <= 0.1 + tol) | (x[1] >= 0.4 - tol))
    for c in body:
        if marker[c] == 0:
            marker[c] = 1
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                            marker[:num_local_cells])

    metadata = {"quadrature_degree": 2 * fe_degree + 2}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)

    # ---- boundary conditions (2D vertices -> edges parallel to z) ----
    left_edge = lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    right_edge = lambda x: np.isclose(x[0], LX) & np.isclose(x[1], 0.0)
    top_edge = lambda x: np.isclose(x[0], 0.5 * LX) & np.isclose(x[1], LY)
    bcl = dirichletbc(np.zeros(3, dtype=default_scalar_type),
                      dofs_on_edges(mesh0, Vu, left_edge), Vu)
    zero = Constant(mesh0, default_scalar_type(0.0))
    bcr = dirichletbc(zero, dofs_on_edges(mesh0, Vu.sub(1), right_edge), Vu.sub(1))
    disp = Constant(mesh0, default_scalar_type(0.0))
    bct = dirichletbc(disp, dofs_on_edges(mesh0, Vu.sub(1), top_edge), Vu.sub(1))
    bcs = [bcl, bcr, bct] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    # ---- energy ----
    mat = make_material(MU, K, MATERIAL_MODEL)
    kin = kinematics_3d(u)
    Pi_body = psi_3d(mat, kin) * dx(1)
    Pi_m = GAMMA * psi_2_3d(mat, kin) * dx(3)
    k = default_scalar_type(k_reg)
    R_reg = (huhu_lulu_regularization(u, v, dx, k, k, 3) if reg_method == "huhu_lulu"
             else huhu_regularization(u, v, dx, k, 3))
    L = derivative(Pi_body + Pi_m, u, v) + R_reg

    dof_count = Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    emit(f"--- monolith p={fe_degree}, {Nx}x{Ny}x{NZ} {CELL_TYPE}, {num_cells} cells, {dof_count} dofs, "
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
    sweep_main("monolith3d", run_case, default_meshes="20x10,40x20,60x30")