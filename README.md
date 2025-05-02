[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Wind Forecasting Package

Version 0.0.1

Developed by Marco Saretta, Thomas Røder and Lukas Karkossa in cooperatiom with DTU Wind and Jenni Rinker

## Overview

WindGPT is a lightweight, modular Python package designed to streamline the end-to-end workflow for wind-power time-series analysis and forecasting. At its basis is a `DataLoader` class that:

1. **Ingests** raw CSV data (power output plus meteorological variables)  
2. **Cleans** it by dropping unneeded columns and parsing timestamps  
3. **Scales** all features into a 0–1 range via MinMax scaling  
4. **Slices** the series into lagged input windows (`X`) and multi‐step targets (`Y`)  
5. (Optionally) **Plots** a battery of EDA graphics: histograms, KDEs, autocorrelations, seasonal decompositions, Welch periodograms, etc.  

Build up on that basis there are three forecasting models integrated: 
1. persistence model
2. linear regression model
3. support vector machine

On top of that foundation you can plug in your own forecasting models—ML or statistical—and easily train, validate, and visualize predictions. WindGPT requires only standard scientific-Python dependencies (NumPy, pandas, scikit-learn, statsmodels, Matplotlib, Seaborn) and can be installed in editable mode for rapid iteration.

## Quick-start guide

How can a user quickly get started with your code?  

In order to quickly run the code like it is, you must fo the following:

**Step 1:** Clone the Repository using git.
git clone <hhttps://github.com/DTUWindEducation/final-project-windgpt>
 - Make sure the working directory is the repository folder

**Step 2:** Create a new/empty environment?
    activate this environment

**Step 3:** Required dependencies
- Since we developed our own package, you simpliy need to install it and all dependencies are installed with it automatically. To install it correclty you need to be in the "main folder" final-project-windgpt and type the command pip install . 
   
- to check if the package is correctly installed you can use the command pip list and see if the package was installed.


**Step 4:** Inspect the Data:
- Make sure the necessary input files are in the data folder:
    - Location1.csv - Location4.csv
    - if you want to add data for your own location, please make sure it has the same "structure" as the original data. The columns Power and windspeed_100m are mandatory.

**Step 4:** Run the Simulation:
 Execute the main script:
 Go to the examples directory and run the script with
 - python main.py
This script processes the wind data for a specific location, and runs the chosen forecasting model, for x hours.

**Step 5:** Review Outputs:
- Review the generated plots showing the predicted power output for the specific location

## Architecture & Functionality

The overall archticture is displayed below. 

## Peer review

We embraced a true “four-eyes” development process on every feature:

1. **Pair programming & branch-level review**  
   - Each new feature or bug-fix was first co-authored or stepped through by two developers working together.
2. **Pull-request review by a third teammate**  
   - Once the pair felt confident, a formal GitHub PR was opened and assigned to a third team member for an independent pass:
     - Verified functionality by running tests and example scripts  
     - Checked code style, documentation completeness, and EDA outputs  
     - Left inline comments or suggestions, which were addressed before merging  
3. **Final sanity check**  
   - After merging, one more teammate double-checked the merged main branch: ran linting, re-ran the full test suite, and spot-checked the example plots.

This multi-stage review strategy ensured high code quality, consistent documentation, and robust test coverage across the entire package.
