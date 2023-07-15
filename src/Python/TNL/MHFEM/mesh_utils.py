#! /usr/bin/env python3

import os.path
import io

import tnl

from .capture import capture_fd

class tnlIOError(Exception):
    pass

def get_mesh_reader(fname_mesh):
    for Reader in [tnl.VTKReader, tnl.VTUReader]:
        reader = Reader(fname_mesh)
        with capture_fd() as r:
            for Mesh in [tnl.Grid1D, tnl.Grid2D, tnl.Grid3D, tnl.MeshOfEdges, tnl.MeshOfTriangles, tnl.MeshOfTetrahedrons, tnl.MeshOfQuadrangles, tnl.MeshOfHexahedrons]:
                try:
                    mesh = Mesh()
                    reader.loadMesh(mesh)
                    return reader, mesh
                except (AssertionError, RuntimeError):
                    pass

    try:
        import tnl_mpi
        reader = tnl_mpi.PVTUReader(fname_mesh)
        with capture_fd() as r:
            for Mesh in [tnl_mpi.DistributedMeshOfEdges, tnl_mpi.DistributedMeshOfTriangles, tnl_mpi.DistributedMeshOfTetrahedrons, tnl_mpi.DistributedMeshOfQuadrangles, tnl_mpi.DistributedMeshOfHexahedrons]:
                try:
                    mesh = Mesh()
                    reader.loadMesh(mesh)
                    tnl_mpi.distributeFaces(mesh)
                    return reader, mesh
                except (AssertionError, RuntimeError):
                    pass
    except ImportError:
        pass

    raise tnlIOError("failed to load mesh from file {}".format(fname_mesh))

def load_mesh(fname_mesh):
    reader, mesh = get_mesh_reader(fname_mesh)
    return mesh

def get_mesh_writer(mesh, format):
    if format == ".vtk":
        module = "tnl"
        attr = "VTKWriter_"
    elif format == ".vtu":
        module = "tnl"
        attr = "VTUWriter_"
    elif format == ".pvtu":
        module = "tnl_mpi"
        attr = "PVTUWriter_"
    else:
        raise ValueError(f"unsupported format: {format} (must be either of '.vtk', '.vtu', '.pvtu')")

    mesh_type = mesh.__class__.__name__
    if mesh_type.startswith("Distributed"):
        if module == "tnl":
            raise ValueError("format {format} is not supported for the distributed mesh type {mesh_type}")
        # PVTUWriter is specialized by the local mesh type
        mesh_type = mesh.getLocalMesh().__class__.__name__

    attr += mesh_type
    return getattr(__import__(module), attr)

def write_mesh_functions(mesh, functions, fname, *, cycle=-1, time=-1):
    ext = os.path.splitext(fname)[1]
    if ext == ".pvtu":
        # create a .pvtu file (only rank 0 actually writes to the file)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            f = open(fname, "wb")
        else:
            f = io.BytesIO()
        writer_class = get_mesh_writer(mesh, ext)
        writer = writer_class(f)
        writer.writeCells(mesh)
        writer.writeMetadata(cycle=cycle, time=time)

        # write essential mesh functions for DistributedMesh
        writer.writePPointData(mesh.vtkPointGhostTypes(), "vtkGhostType")
        writer.writePPointData(mesh.getGlobalPointIndices(), "GlobalIndex")
        writer.writePCellData(mesh.vtkCellGhostTypes(), "vtkGhostType")
        writer.writePCellData(mesh.getGlobalCellIndices(), "GlobalIndex")
        # write specified mesh functions
        for name, vector in functions.items():
            writer.writePCellData(vector, name)

        # add <Piece> tags for each subdomain
        pvtu_fname = fname
        for i in range(comm.Get_size()):
            path = writer.addPiece(pvtu_fname, i)
            if i == comm.Get_rank():
                fname = path

        # extension for local file
        ext = ".vtu"
        localMesh = mesh.getLocalMesh()
    else:
        localMesh = mesh

    # create a .vtu file for local data
    f = open(fname, "wb")
    writer_class = get_mesh_writer(localMesh, ext)
    writer = writer_class(f)
    writer.writeMetadata(cycle=cycle, time=time)
    writer.writeCells(localMesh)
    if mesh is not localMesh:
        # write essential mesh functions for DistributedMesh
        writer.writePointData(mesh.vtkPointGhostTypes(), "vtkGhostType")
        writer.writePointData(mesh.getGlobalPointIndices(), "GlobalIndex")
        writer.writeCellData(mesh.vtkCellGhostTypes(), "vtkGhostType")
        writer.writeCellData(mesh.getGlobalCellIndices(), "GlobalIndex")
    # write specified mesh functions
    for name, vector in functions.items():
        writer.writeCellData(vector, name)
