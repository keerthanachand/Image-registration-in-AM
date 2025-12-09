#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG ==================
CSV_FILE = "PoreAnalysis.csv"
SAMPLE_NAME = "RB370"
VOXEL_SIZE = 0.00437945   # [mm]
# ============================================

# Read ONLY the same four columns as the original script
# Original: usecols = (1, 2, 8, 5)
usecols = [
    "Equivalent diameter [mm]",   # index 1
    "Voxel",                      # index 2
    "Pos. y [vox]",               # index 8  (was treated as z-pos previously!)
    "Compactness"                 # index 5  (was treated as sphericity)
]

# Load data using pandas (cleaner than np.loadtxt)
df = pd.read_csv(CSV_FILE, sep=";", decimal=",")

# Extract the four columns
eq_dia = df["Equivalent diameter [mm]"].to_numpy(float)
voxels = df["Voxel"].to_numpy(float)
y_pos  = df["Pos. y [vox]"].to_numpy(float)   # this WAS mistaken for z_pos
compactness = df["Compactness"].to_numpy(float)

# Convert physics like before
volume = voxels * (VOXEL_SIZE ** 3)  # voxel count → mm³
pos_mm = np.abs(y_pos * VOXEL_SIZE) # 
#pos = np.abs(df["Pos. y [vox]"].to_numpy())


# ========== PLOTS (same as old script) ==========

# 1. Volume-weighted histogram of (mistaken) z-position
plt.figure()
plt.hist(pos_mm, bins=50, density=True, weights=volume, facecolor="b", label=SAMPLE_NAME)
plt.xlabel("Position [mm]")
plt.ylabel("Volume weighted relative frequency")
plt.legend()
plt.tight_layout()
plt.savefig("VolumeWeightedDistribution_z_pos.jpg")
plt.close()

# 2. Number-weighted histogram of (mistaken) z-position
plt.figure()
plt.hist(pos_mm, bins=50, density=True, facecolor="b", label=SAMPLE_NAME)
plt.xlabel("Position [mm]")
plt.ylabel("Number weighted relative frequency")
plt.legend()
plt.tight_layout()
plt.savefig("NumberWeightedDistribution_z_pos.jpg")
plt.close()

# 3. Volume-weighted histogram of equivalent diameter
plt.figure()
plt.hist(eq_dia, bins=50, density=True, weights=volume, facecolor="b", label=SAMPLE_NAME)
plt.xlabel("Equivalent diameter [mm]")
plt.ylabel("Volume weighted relative frequency")
plt.legend()
plt.tight_layout()
plt.savefig("VolumeWeightedDistribution_eq_Dia.jpg")
plt.close()
