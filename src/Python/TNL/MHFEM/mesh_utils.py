import os.path
import io

import pytnl.meshes


def get_mesh_writer(mesh, format):
    if format == ".vtk":
        writer = pytnl.meshes.VTKWriter
    elif format == ".vtu":
        writer = pytnl.meshes.VTUWriter
    elif format == ".pvtu":
        writer = pytnl.meshes.PVTUWriter
    else:
        raise ValueError(
            f"unsupported format: {format} (must be either of '.vtk', '.vtu', '.pvtu')"
        )

    mesh_class = mesh.__class__
    if mesh_class.__name__.startswith("Distributed"):
        if writer is not pytnl.meshes.PVTUWriter:
            raise ValueError(
                "format {format} is not supported for the distributed mesh type {mesh_type}"
            )
        # PVTUWriter is specialized by the local mesh type
        mesh_class = mesh.getLocalMesh().__class__

    return writer[mesh_class]


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
