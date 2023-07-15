#! /usr/bin/env python3

import tnl
import tnl_mhfem

from TNL.MHFEM.argtypes import argtype_existing_file
from TNL.MHFEM.mesh_utils import load_mesh, write_mesh_functions

def set_initial_condition(mesh, alpha, beta, gamma):
    numberOfCells = mesh.getEntitiesCount(mesh.getCell(0))
    vector = tnl.Array()
    vector.setSize(numberOfCells)
    vector.setValue(0)

    xDomainSize = 1
    yDomainSize = 1

    for k in range(numberOfCells):
        center = mesh.getEntityCenter(mesh.getCell(k))
        x = center[0] - xDomainSize / 2
        y = center[1] - yDomainSize / 2
        vector[k] = max( 0, ( x*x / alpha + y*y / beta + gamma ) * 0.2 )

    return vector

def set_boundary_condition(mesh, fname_out):
    faces = mesh.getEntitiesCount(mesh.getFace(0))

    bcs = tnl_mhfem.BoundaryConditionsStorage()
    bcs.dofSize = faces
    bcs.tags.setSize( bcs.dofSize )
    bcs.values.setSize( bcs.dofSize )
    bcs.dirichletValues.setSize( bcs.dofSize )

    bcs.tags.setValue( tnl_mhfem.BoundaryConditionsType.FixedValue )
    bcs.values.setValue( 0.0 )
    bcs.dirichletValues.setValue( 0.0 )

    bcs.save( fname_out )

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="Heat equation example initial condition generator")
    g = argparser.add_argument_group("input files")
    g.add_argument("--mesh", required=True, type=argtype_existing_file, help="path to file with TNL grid")
    g = argparser.add_argument_group("initial condition parameters")
    g.add_argument("--alpha", type=float, default=-0.05, help="Alpha value in initial condition")
    g.add_argument("--beta", type=float, default=-0.05, help="Beta value in initial condition")
    g.add_argument("--gamma", type=float, default=5, help="Gamma value in initial condition")
    g = argparser.add_argument_group("output files")
    g.add_argument("--output", required=True, help="where to save the mesh with the initial condition (VTK, VTU or PVTU format)")
    g.add_argument("--output-boundary", required=True, help="where to save the bundary conditions (TNL format)")

    args = argparser.parse_args()

    mesh = load_mesh(args.mesh)
    if hasattr(mesh, "getLocalMesh"):
        localMesh = mesh.getLocalMesh()
    else:
        localMesh = mesh

    ic = set_initial_condition(localMesh, args.alpha, args.beta, args.gamma)
    functions = {
        "InitialCondition[Z0]": ic,
    }
    write_mesh_functions(mesh, functions, args.output, cycle=0)

    if args.mesh.endswith(".pvtu"):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            import os.path
            dirname, basename = os.path.split(args.output_boundary)
            basename, ext = os.path.splitext(basename)
            basename += ".{}".format(comm.Get_rank())
            output_boundary = os.path.join(dirname, basename + ext)
        else:
            output_boundary = args.output_boundary
    else:
        output_boundary = args.output_boundary

    set_boundary_condition(localMesh, output_boundary)
