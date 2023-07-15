#! /usr/bin/env python3

import argparse
import configparser
import subprocess
from pathlib import Path

# initialize directories
example_dir = Path(__file__).parent
project_dir = (example_dir / ".." / "..").resolve()
bin_dir = project_dir / "build" / example_dir.relative_to(project_dir)

def parse_config(config_path: Path):
    parser = configparser.ConfigParser()
    # ConfigParser does not support INI files without any section header,
    # so we prepend a fake one: https://stackoverflow.com/a/26859985
    with open(config_path, "r") as f:
        parser.read_string("[top]\n" + f.read())
    # return options in the fake section
    return parser, parser["top"]

def decompose_mesh(input_path: Path, output_path: Path, subdomains: int):
    args = [
        "tnl-decompose-mesh",
        "--input-file", input_path,
        "--output-file", output_path,
        "--subdomains", str(subdomains),
        "--ghost-levels", "1",
        "--metis-niter", "100",
        "--metis-ncuts", "10",
    ]
    subprocess.run(args, check=True, cwd=example_dir)

def init(mpi_ranks: int, mesh_path: Path, output_initial_path: Path, output_boundary_path: Path):
    args = []
    if mpi_ranks > 1:
        args += ["mpirun", "-np", str(mpi_ranks)]
    args += [
        example_dir / "init.py",
        "--mesh", mesh_path,
        "--output", output_initial_path,
        "--output-boundary", output_boundary_path,
    ]
    subprocess.run(args, check=True, cwd=example_dir)

def solve(device: str, mpi_ranks: int, config_path: Path, log_path: Path):
    solver_path = bin_dir / f"HeatEquation_{device}"

    args = []
    if mpi_ranks > 1:
        args += ["mpirun", "-np", str(mpi_ranks)]
    args += [
        solver_path,
        "--config", config_path,
        "--log-file", log_path,
        "--redirect-mpi-output", "true",
        "--redirect-mpi-output-dir", log_path.parent,
    ]

    # run the process and print its output as it is being executed
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          bufsize=1, cwd=example_dir, text=True) as p:
        for line in p.stdout:
            print(line, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Heat equation example")
    argparser.add_argument("--config", default="config.ini",
            help="path to the config file (relative to the path of this script)")
    argparser.add_argument("--exec", default="host", choices=["host", "cuda"],
            help="execution of the MHFEM solver")
    argparser.add_argument("--mpi-ranks", default=1, type=int,
            help="number of MPI ranks for distributed computation")

    # parse the command line arguments
    args = argparser.parse_args()

    # parse the config file
    config_path = example_dir / args.config
    config_parser, config = parse_config(config_path)

    # create the output directory
    output_dir = example_dir / config["output-directory"]
    output_dir.mkdir(exist_ok=True)

    # read paths from the config file
    mesh_path = example_dir / config["mesh"]
    output_initial_path = example_dir / config["initial-condition"]
    output_boundary_path = example_dir / config["boundary-conditions-file"]

    # prepare distributed computation
    if args.mpi_ranks > 1:
        # decompose the mesh
        distributed_mesh_path = output_dir / "mesh.pvtu"
        decompose_mesh(mesh_path, distributed_mesh_path, args.mpi_ranks)

        # update the paths in the config object
        mesh_path = distributed_mesh_path
        config["mesh"] = str(mesh_path.relative_to(example_dir))
        output_initial_path = output_initial_path.parent / (output_initial_path.stem + ".pvtu")
        config["initial-condition"] = str(output_initial_path.relative_to(example_dir))

        # save the updated config as a copy in the output directory
        config_path = output_dir / "config.ini"
        with open(config_path, "w") as f:
            config_parser.write(f)
        # remove the first line with the section header
        lines = open(config_path, "r").readlines()[1:]
        with open(config_path, "w") as f:
            for line in lines:
                f.write(line)

    # create the initial and boundary conditions
    init(args.mpi_ranks, mesh_path, output_initial_path, output_boundary_path)

    # execute the solver
    log_path = output_dir / "log.txt"
    solve(args.exec, args.mpi_ranks, config_path, log_path)
