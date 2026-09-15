# Agent instructions

This repository is quite large and contains binaries.
Do not read files the user did not specifically ask you to read.

The repo uses a virtual python environment created using uv.
All dependencies are listed in pyproject.toml.
You can use these dependencies and run them with uv.

If you are editing a marimo file,
you can run it as a simple python file to catch all errors.

Never edit an existing file unless you are instructed otherwise.
Always create a new file.

## Directory of interest (DoI)

Our main task is in `./inv_uq/2026-03-12/` directory.
Open and edit files only in the directory of interest (DoI).

## Datasets

The DoI has two datasets:
`RERTR5_9_parameters.csv` and `RERTR12_9_parameters.csv`.
RERTR5 has low fission density data,
and RERTR12 has high fission density data.

### CAUTION

Some inputs have values and ranges near `1e-18`.
Never apply sklearn scalers directly to the raw inputs:
`MinMaxScaler` can treat such ranges as constant.
First make every input dimensionless by dividing it
by a recorded, nonzero parameter-specific reference scale,
then apply any additional scaling.
Verify that every nonconstant input remains nonconstant
and has an order-one transformed range.

## Parameter definitions

### Input

- dGrainHBS denotes the high burnup structure grain diameter.
- FaceCovMax means the fraction of grain face coverage needed for interlinkage.
- SwellLink determines the grain edge swelling starting interlinkage.
- rResolBulk is the thickness of the destructed outer shell
  in gas bubbles due to re-solution.
- DatomFissGBx is the grain boundary diffusion enhancement factor.
- StickProb is the probability that two atoms stick to form a bubble nucleus.
- fNucleate indicates the adjustment factor
  for the probability of bubble nucleation on the grain boundary.
- vResol is the probability that a gas bubble interacts with fission fragments.
- rResolGBB is the average re-solution distance
  an atom is ejected from the grain boundary gas bubbles.

### Output

- Fuel_Swelling: self explanatory.
- Porosity: highly correlated with Fuel_Swelling; should be ignored.
- Bubble_Size: average bubble size.
- C1--C6: six-bin discretization of
  the bubble-size concentration distribution.

## Experimental observations

For RERTR5, the experimentally observed values and measurement error/noise are:

| Param         | Obs.  | Std dev |
| -----         | ----- | ------- |
| Fuel_Swelling | 10.7  | 2.64    |
| C1            |  2.36 | 0.24    |
| C2            |  3.54 | 0.35    |
| C3            |  8.21 | 0.82    |
| C4            | 12.93 | 1.29    |
| C5            |  7.06 | 0.71    |
| C6            |  4.71 | 0.47    |

For RERTR12, the experimentally observed values and measurement error/noise are:

| Param         | Obs.  | Std dev |
| -----         | ----- | ------- |
| Fuel_Swelling | 32    | 2.64    |
| Bubble_Size   | 0.47  | 0.26    |

## Research goal

We will construct surrogate models using the two datasets.
We will mainly use Bayesian Neural Networks and Gaussian Processes
to build these surrogate models.

The goal is to perform calibration of input parameters
using Bayesian inference.
So, we need to find posterior distributions of input parameters
such that the forward propagation from those parameters
lead to output values close to the experimental observation.

RERTR5 and RERTR12 have different sets of output parameters available.
Some outputs might be highly correlated, and thus redundant.
The goal is to find posteriors of input parameters
using multiple (somewhat uncorrelated) outputs.
