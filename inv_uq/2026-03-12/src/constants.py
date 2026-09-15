"""Study-wide data definitions and fixed physical reference scales."""

from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parents[1]

DATASET_PATHS = {
    "RERTR5": STUDY_DIR / "RERTR5_9_parameters.csv",
    "RERTR12": STUDY_DIR / "RERTR12_9_parameters.csv",
}

INPUT_NAMES = (
    "dGrainHBS",
    "FaceCovMax",
    "SwellLink",
    "rResolBulk",
    "DatomFissGBx",
    "StickProb",
    "fNucleate",
    "vResol",
    "rResolGBB",
)

OUTPUT_NAMES = {
    "RERTR5": (
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "Porosity",
        "Fuel_Swelling",
    ),
    "RERTR12": ("Fuel_Swelling", "Porosity", "Bubble_Size"),
}

# Porosity is intentionally excluded because it is redundant with Fuel_Swelling.
MODELING_OUTPUT_NAMES = {
    "RERTR5": ("Fuel_Swelling", "C1", "C2", "C3", "C4", "C5", "C6"),
    "RERTR12": ("Fuel_Swelling", "Bubble_Size"),
}

EXPERIMENTAL_OBSERVATIONS = {
    "RERTR5": {
        "Fuel_Swelling": {"value": 10.7, "std_dev": 2.64},
        "C1": {"value": 2.36, "std_dev": 0.24},
        "C2": {"value": 3.54, "std_dev": 0.35},
        "C3": {"value": 8.21, "std_dev": 0.82},
        "C4": {"value": 12.93, "std_dev": 1.29},
        "C5": {"value": 7.06, "std_dev": 0.71},
        "C6": {"value": 4.71, "std_dev": 0.47},
    },
    "RERTR12": {
        "Fuel_Swelling": {"value": 32.0, "std_dev": 2.64},
        "Bubble_Size": {"value": 0.47, "std_dev": 0.26},
    },
}

# Fixed, nonzero, parameter-specific scales chosen from physical orders of
# magnitude. They are not fitted to either dataset and must be applied before
# any statistical scaler.
INPUT_REFERENCE_SCALES = {
    "dGrainHBS": 1.0e-4,
    "FaceCovMax": 1.0,
    "SwellLink": 1.0e-1,
    "rResolBulk": 1.0e-9,
    "DatomFissGBx": 1.0e4,
    "StickProb": 1.0e-7,
    "fNucleate": 1.0e-9,
    "vResol": 1.0e-18,
    "rResolGBB": 1.0e-7,
}

# A transformed range in this interval is considered order-one for the audit.
ORDER_ONE_RANGE = (1.0e-1, 1.0e1)

