# H-MA-LNS for Topographic Asymmetric Capacitated Vehicle Routing Problem (ACVRP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Optimization: Pymoo](https://img.shields.io/badge/Optimization-Pymoo-orange.svg)](https://pymoo.org/)

This repository contains the source code, supplementary data, and comprehensive experimental results supporting our research on routing optimization under real-world topographic constraints. 

We introduce **H-MA-LNS**, a Hybrid Memetic Algorithm integrated with a Large Neighborhood Search (LNS) module, designed to solve the Asymmetric Capacitated Vehicle Routing Problem (ACVRP) by directly embedding OpenStreetMap (OSMnx) road networks and SRTM Digital Elevation Models (DEM).

##  Overview

Traditional VRP research often assumes flat, Euclidean surfaces. However, real-world logistics (especially for heavy-duty or electric vehicles) are heavily impacted by road gradients. This repository provides the tools to:
1. Extract real street networks and physical distances.
2. Overlay SRTM elevation data to calculate road slopes.
3. Compute an asymmetric, topography-aware impedance matrix.
4. Optimize the routing using a state-of-the-art hybrid metaheuristic (H-MA-LNS).

##  Repository Structure

The repository is systematically organized to ensure full reproducibility of the computational experiments conducted on the 24 Seoul-based instances originally proposed by Lee et al. (2021).
```text
├──  Elevation Data
│   ├── n37_e126_1arc_v3.tif        # SRTM Digital Elevation Model (Tile 1)
│   └── n37_e127_1arc_v3.tif        # SRTM Digital Elevation Model (Tile 2)
│
├──  Source Code
│   ├── src_solver_asymmetric.py    # Base H-MA-LNS solver for standard ACVRP
│   └── srcH_MA_LNS_Topographic.py  # Advanced solver integrating DEM & OSMnx slope penalties
│
├──  Standard Asymmetric Results (ACVRP)
│   ├── results_H-MA-LNS_SLAM250_DV1M5.xlsx
│   ├── ... (24 detailed instance files)
│
└──  Topographic Results (10 Runs Summary)
    ├── summary_10runs_Topo_H-MA-LNS_SLAM250_DV1M5.xlsx
    ├── ... (24 consolidated statistical summary files)