"""
3D clamped box ("rotbox"): the monolith geometry, clamped at both ends, with the LEFT cross-section
TWISTED about the beam axis (x-parallel axis through the section centre (LY/2, DEPTH/2)), and
enclosed by one layer of third-medium elements.

  * right face (x = LX): fully clamped (u = 0)
  * left face  (x = 0) : rigid rotation by theta about the x-axis, u_x = 0
                         u_y = cos(th)*ry - sin(th)*rz - ry,  u_z = sin(th)*ry + cos(th)*rz - rz
                         with ry = y - LY/2, rz = z - DEPTH/2
  * the angle is applied exactly (not by linear interpolation of displacements), so 180 deg is fine
  * one element layer of third medium (tag 3) around the box in x, y and z; solid box stays in
    [0,LX]x[0,LY]x[0,DEPTH]
  * load parameter s runs from 0 to TARGET = -ROT_DEG (rad); theta = -s. Positive ROT_DEG = right-handed
    twist about +x.

Environment variables
  ROT_DEG   final twist angle in degrees              (default 180)
  LAYER     1/0  enclosing third-medium layer         (default 1)
  LAYER_Z   1/0  layer also in z                      (default 1)
  OUTER_BC  free | fixed  outer boundary of the layer (default free)
  ZBC must be 'free' (plane_strain is incompatible with a twist)
"""
import os

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         assemble_scalar, form, locate_dofs_geometrical)
from ufl import Measure, TestFunction, derivative

from methods import make_material, huhu_lulu_regularization, huhu_regularization
from functional_3d import kinematics_3d, psi_3d, psi_2_3d
from driver3d import (make_box_mesh, z_support_bcs,
                      solve_with_load_stepping, sweep_main)

MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "hartmann_neff")
CELL_TYPE = os.environ.get("CELL_TYPE", "hexahedron")
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "10"))
ROT_DEG = float(os.environ.get("ROT_DEG", "180"))
LAYER = os.environ.get("LAYER", "1") == "1"
LAYER_Z = LAYER and os.environ.get("LAYER_Z", "1") == "1"
OUTER_BC = os.environ.get("OUTER_BC", "free")
if ZBC == "plane_strain":
    raise ValueError("rotbox twists the cross-section: use ZBC=free")
LX, LY = 2.0, 0.5
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else LY

# model parameters of the 2D script
MU, K, GAMMA = 10.0, 20.0, 1.0e-7
TARGET = -np.deg2rad(ROT_DEG)          # load parameter s; rotation angle theta = -s
PRM = dict(Nsteps=100, Nsteps_min=10, Nsteps_max=800, crit=0.8 * TARGET, inc=1.5, inc_late=1.1,
           dec=0.5, max_attempts=10,
           max_it=int(os.environ.get("NEWTON_MAXIT", "15")),
           atol=float(os.environ.get("NEWTON_ATOL", "1e-12")),
           rtol=float(os.environ.get("NEWTON_RTOL", "1e-9")),
           stol=float(os.environ.get("NEWTON_STOL", "1e-10")),
           linesearch="bt")


class RotationLoad:
    """Drop-in replacement for the `disp` Constant of the load-stepping driver.

    Setting `.value = s` updates the prescribed displacement field `g` of the left face:
    rigid rotation by theta = -s about the x-parallel axis through `centre` = (y_c, z_c).
    """

    def __init__(self, g, face_dofs, coords, centre):
        self.g, self.dofs, self.centre = g, face_dofs, centre
        self.ry = coords[:, 1] - centre[0]
        self.rz = coords[:, 2] - centre[1]
        self._s = 0.0
        self.value = 0.0

    @property
    def value(self):
        return self._s

    @value.setter
    def value(self, s):
        self._s = float(np.asarray(s).ravel()[0])
        th = -self._s
        c, sn = np.cos(th), np.sin(th)
        gv = self.g.x.array.reshape(-1, 3)
        gv[self.dofs, 0] = 0.0
        gv[self.dofs, 1] = c * self.ry - sn * self.rz - self.ry
        gv[self.dofs, 2] = sn * self.ry + c * self.rz - self.rz
        self.g.x.scatter_forward()


def run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit, verbose=False):
    comm = MPI.COMM_WORLD
    t_start = MPI.Wtime()
    if fe_degree < 2:
        raise ValueError("Hessian regularisation needs p >= 2")
    if Nx % 20 or Ny % 5:
        raise ValueError("monolith needs Nx % 20 == 0 and Ny % 5 == 0 so that the 0.1-wide "
                         "walls coincide with mesh lines")

    # ---- mesh: box [0,LX]x[0,LY]x[0,DEPTH] plus one element layer of third medium ----
    # layer thickness = one element size per direction, so the mesh stays uniform & conforming
    nl = 1 if LAYER else 0
    nlz = 1 if LAYER_Z else 0
    hx, hy, hz = LX / Nx, LY / Ny, DEPTH / NZ
    ox, oy, oz = nl * hx, nl * hy, nlz * hz
    mesh0 = make_box_mesh(comm, LX + 2 * ox, LY + 2 * oy, DEPTH + 2 * oz,
                          Nx + 2 * nl, Ny + 2 * nl, NZ + 2 * nlz, CELL_TYPE)
    mesh0.geometry.x[:, :] -= np.array([ox, oy, oz])       # solid box back to [0,LX]x[0,LY]x[0,D]
    xmin, xmax, ymin, ymax = -ox, LX + ox, -oy, LY + oy
    zmin, zmax = -oz, DEPTH + oz

    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    num_local_cells = mesh0.topology.index_map(tdim).size_local
    h = comm.allreduce(float(np.min(mesh0.h(tdim, np.arange(num_local_cells)))), op=MPI.MIN)

    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    v = TestFunction(Vu)
    u = Function(Vu, name="displacement")

    # ---- material tagging: 1 = frame (solid), 3 = third medium (cavity + outer layer) ----
    tol = 1e-8
    imap = mesh0.topology.index_map(tdim)
    # locate_entities also returns ghost cells -> size the array for owned + ghost cells
    marker = np.full(imap.size_local + imap.num_ghosts, 3, dtype=np.int32)   # default: third medium

    in_box = lambda x: ((x[0] >= -tol) & (x[0] <= LX + tol) & (x[1] >= -tol) & (x[1] <= LY + tol)
                        & (x[2] >= -tol) & (x[2] <= DEPTH + tol))
    in_wall = lambda x: ((x[0] <= 0.1 + tol) | (x[0] >= LX - 0.1 - tol)
                         | (x[1] <= 0.1 + tol) | (x[1] >= 0.4 - tol))
    # frame = cells fully inside the box AND inside a 0.1-wide wall; layer cells touch x<0 etc. -> excluded
    frame = locate_entities(mesh0, tdim, lambda x: in_box(x) & in_wall(x))
    marker[frame] = 1
    material_tags = meshtags(mesh0, tdim, np.arange(num_local_cells, dtype=np.int32),
                             marker[:num_local_cells])

    metadata = {"quadrature_degree": 2 * fe_degree + 2}
    dx = Measure("dx", domain=mesh0, subdomain_data=material_tags, metadata=metadata)

    # ---- boundary conditions ----
    on_left = lambda x: (np.isclose(x[0], 0.0) & (x[1] >= -tol) & (x[1] <= LY + tol)
                         & (x[2] >= -tol) & (x[2] <= DEPTH + tol))
    on_right = lambda x: (np.isclose(x[0], LX) & (x[1] >= -tol) & (x[1] <= LY + tol)
                          & (x[2] >= -tol) & (x[2] <= DEPTH + tol))

    # right face: fully clamped
    right_dofs = locate_dofs_geometrical(Vu, on_right)
    bcr = dirichletbc(np.zeros(3, dtype=default_scalar_type), right_dofs, Vu)

    # left face: prescribed rigid twist about the x-parallel axis through the section centre
    left_dofs = locate_dofs_geometrical(Vu, on_left)
    g = Function(Vu, name="left_face_displacement")
    coords = Vu.tabulate_dof_coordinates()[left_dofs]
    disp = RotationLoad(g, left_dofs, coords, centre=(0.5 * LY, 0.5 * DEPTH))
    bcl = dirichletbc(g, left_dofs)

    bcs = [bcl, bcr] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    if LAYER and OUTER_BC == "fixed":
        outer = lambda x: (np.isclose(x[0], xmin) | np.isclose(x[0], xmax)
                           | np.isclose(x[1], ymin) | np.isclose(x[1], ymax)
                           | np.isclose(x[2], zmin) | np.isclose(x[2], zmax))
        bcs.append(dirichletbc(np.zeros(3, dtype=default_scalar_type),
                               locate_dofs_geometrical(Vu, outer), Vu))

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
    emit(f"--- rotbox p={fe_degree}, {Nx}x{Ny}x{NZ}(+layer) {CELL_TYPE}, {num_cells} cells, "
         f"{dof_count} dofs, {reg_method} k={k_reg:.1e}, {MATERIAL_MODEL}, ZBC={ZBC}, "
         f"depth={DEPTH:g}, rot={ROT_DEG:g} deg, layer={LAYER}(z={LAYER_Z}), outer={OUTER_BC} ---")
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
        "final_displacement": stats["current"],
        "final_angle_deg": float(np.rad2deg(-stats["current"])),
        "n_steps": steps, "potential": potential,
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
    sweep_main("rotbox3d", run_case, default_meshes="20x10,40x20,60x30")