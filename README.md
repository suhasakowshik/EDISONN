
# EDISONN: Estimation-based Direct Simulation on Nodal Networks (C++)

Locking-free, interpolation-free mechanics solver that operates directly on gradient estimation using nodal networks.  

---

## Features

- **Interpolation-free formulation:** gradient estimation from nodal connectivity, no element shape functions.
- **Locking-resistant:** robust under near-incompressibility and shear-dominated regimes.
- **2D/3D support:** linear elasticity; hooks for finite strain, viscoelastic add-ons (wiil be added later).
- **Unstructured meshes:** triangles/quads/tets/hexes; can operate on graph connectivity.
- **Parallelism:** OpenMP on CPU;
- **Modular:** pluggable gradient estimators, solvers (CG), and preconditioners.
- **I/O:** simple mesh readers and BC reader in .txt and text output.

---

## File Usage

The main executable `edisonn` accepts a single argument: a key-value input file.

Example:

```bash
./edisonn input.txt
```

Alternatively, you can use the helper script `run_edisonn.sh`, which will:

- Source Intel oneAPI (if available)
- Parse values from the input
- Name the log based on `simName`, `meshType`, and `version`
- Capture output in `full_run_log.txt`

To run:

```bash
chmod +x run_edisonn.sh
./run_edisonn.sh input.txt
```

Or from root if the input is in a subfolder (e.g. `input/cooks_membrane/input.txt`):

```bash
./scripts/run_edisonn.sh input/cooks_membrane/input.txt
```

Make sure the `edisonn` executable is in the same folder as the input and mesh files.

---

## Sample Input File (`input.txt`)

```txt
version = 1
meshType = tri
simName = cooks_membrane
problemType = plane_elasticity
planeType = plane_strain
loadType = distributed_load
smooth = true
smoothFitType = linear
load = 6.25
E = 70.0
nu = 0.3333
thickness = 1.0
smoothIter = 2
numThreads = 12
```

This file defines solver parameters, load and material info, and solver mode.

---

## Input Folder Structure

Each simulation folder must contain:

```
cooks_membrane/
├── edisonn                    # compiled executable
├── input.txt                  # simulation config file
├── cooks_membrane_tri-1_nodes.txt
├── cooks_membrane_tri-1_elems.txt
├── cooks_membrane_tri-1_dispBCs.txt
├── cooks_membrane_tri-1_forceBCs.txt
```

File names are determined by `simName`, `meshType`, and `version` in `input.txt`.

---

## Output

- `full_run_log.txt`: full stdout + stderr of the simulation run
- `cooks_membrane_tri-1_simulation_log.txt`: structured summary of results (if implemented)

---

## License

MIT License.
