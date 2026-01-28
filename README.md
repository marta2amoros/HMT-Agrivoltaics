# HMT-Agrivoltaics

Heat and mass transfer (HMT) utilities and notebooks for open-field bifacial semi-transparent agrivoltaic orchards modelling workflows. 

## What’s in this repo

This repository contains a small set of Python modules, a Jupyter notebook, and bundled input data.

### Repository structure

- `base.ipynb` — main notebook entry point for running/inspecting the workflow. :contentReference[oaicite:2]{index=2}  
- `functions_module.py` — helper functions used by the notebook/workflow. :contentReference[oaicite:3]{index=3}  
- `ET_func.py` — evapotranspiration-related utilities (as used in the workflow). :contentReference[oaicite:4]{index=4}  
- `py56FAO/` — local FAO-56 related code/resources used by the project. :contentReference[oaicite:5]{index=5}  
- `Optical.zip` — optical-related input files (zipped). :contentReference[oaicite:6]{index=6}  
- `Weather.zip` — weather input files (zipped). :contentReference[oaicite:7]{index=7}  

## Quick start
### 1) Clone the repository
```bash
git clone https://github.com/marta2amoros/HMT-Agrivoltaics.git
cd HMT-Agrivoltaics
```


### 2) Create a Python environment (recommended)

Using `venv`:

```bash
python -m venv .venv
```
### 3) Install dependencies

Install the packages used in `base.ipynb`:

```bash
pip install pvlib pandas numpy psychrolib matplotlib scipy tmm thermo pyfao56 import-ipynb nbimporter
```
### 4) Unzip input data
```bash
unzip -o Optical.zip -d Optical
unzip -o Weather.zip -d Weather
```

### 5) Run the notebook
```bash
jupyter notebook
```
Open `base.ipynb` and run cells top-to-bottom.


## Typical workflow

1. Unzip `Weather.zip` and `Optical.zip` into `Weather/` and `Optical/`.

2. Open and run `base.ipynb`.

3. The notebook calls functions from:

  - `ET_func.py` (evapotranspiration-related routines)

  - `functions_module.py` (shared utilities)

  - `py56FAO/` (FAO-56 related components)

## Data notes

`Weather.zip` should contain meteorological inputs and input data used by the workflow.

`Optical.zip` should contain optical inputs used by the workflow.

If you change the input datasets, keep the expected file names and folder structure (or update the paths used in `base.ipynb`).

