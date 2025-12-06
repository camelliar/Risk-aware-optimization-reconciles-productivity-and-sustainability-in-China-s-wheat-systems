=========================================================================
PART A: SITE-LEVEL OPTIMIZATION (R + DSSAT)
=========================================================================

1. OVERVIEW
---------------------
This module performs multi-objective optimization for wheat management (Nitrogen and Irrigation) using the NSGA-II algorithm coupled with the DSSAT-CERES-Wheat model. The R script iteratively calls the DSSAT executable to generate Pareto-optimal solutions for a sample site-year.

2. DIRECTORY STRUCTURE
---------------------
Ensure the unzipped folder maintains the following structure for the demo to run:
/ (Root Directory)
 |__ run_optimization_demo.R      # The main R script
 |__ 53797/                       # Sample Site ID folder
      |__ 2013/                   # Sample Year folder
           |__ 37971301.WHX       # DSSAT Experiment File
           |__ DSSBatch.v47       # Batch file for DSSAT
           |__ *.SOL / *.CUL      # (Ensure Soil/Cultivar files are present if not standard)

3. PREREQUISITES
---------------------
A. Software:
   - R (tested on version 4.2.2 or higher)
   - DSSAT v4.7 (Must be installed on the local machine)

B. R Packages:
   Run the following command in R to install required dependencies:
   install.packages(c("mco", "DSSAT", "lubridate", "ggplot2", "foreach", "doParallel"))

4. CONFIGURATION (CRITICAL STEP)
---------------------
Before running the script, you MUST update the path to the DSSAT executable in 'run_optimization_demo.R':

1. Open 'run_optimization_demo.R'.
2. Locate line 24 (approx.):
   dssat_exe <- "C:/DSSAT47/DSCSM047.EXE"
3. Change this path to match the location of 'DSCSM047.EXE' on your computer. 
   - Windows Example: "C:/DSSAT47/DSCSM047.EXE"
   - Linux Example: "/usr/local/bin/dscsm047"

5. HOW TO RUN
---------------------
1. Open R or RStudio.
2. Set the working directory to the root of this unzipped folder:
   setwd("path/to/submission_code")
3. Run the script 'run_optimization_demo.R'.
   - The script will initialize a parallel cluster (default 2 cores).
   - It will perform a short optimization run (Generations=5, PopSize=12) for demonstration purposes.

6. EXPECTED OUTPUT
---------------------
Upon successful completion, a CSV file will be generated in the root directory:
Filename format: 53797_pareto_2013.csv

Columns include:
- Fer_Input: Optimized Nitrogen rate (kg/ha)
- Irr_Actual: Simulated Irrigation amount (mm)
- N2O: Nitrous oxide emissions (kg N/ha)
- N_Leach: Nitrate leaching (kg N/ha)
- Neg_Yield: Negative yield (for minimization objective)
- Fer_Optim: Parameter value for N
- Irr_Threshold: Parameter value for Irrigation

Run time: Approximately 1-3 minutes on a standard desktop computer for this demo configuration.


=========================================================================
PART B: SPATIAL PREDICTION (PYTHON + XGBOOST)
=========================================================================

1. OVERVIEW
---------------------
This module utilizes XGBoost machine learning models to scale up the optimization results from site-level to regional scale. It trains regressors on the optimized dataset and generates spatial predictions (raster maps) for optimal Nitrogen fertilizer and Irrigation requirements under various climate scenarios (2001-2020).

2. DIRECTORY STRUCTURE
---------------------
Ensure the Python scripts and data folders are organized as follows:

/ (Root Directory)
 |__ train_opt.py                 # Script 1: Model Training
 |__ grid_predict_fertilizer.py   # Script 2: Fertilizer Prediction
 |__ grid_predict_irrigation.py   # Script 3: Irrigation Prediction
 |__ train_data_fer.csv           # Training data for Fertilizer
 |__ train_data_irri.csv          # Training data for Irrigation

3. PREREQUISITES
---------------------
A. Software:
   - Python (Tested on version 3.8 or higher)

B. Python Libraries:
   Run the following command in your terminal/command prompt to install dependencies:
   pip install pandas numpy xgboost rasterio joblib scikit-learn

4. CONFIGURATION (CRITICAL STEP)
---------------------
1. Open 'grid_predict_fertilizer.py' and 'grid_predict_irrigation.py'.
2. Locate the "Global Configuration" section (approx. line 15):
   BASE_INPUT_DIR = 'D:/grid_pred/'
3. Change this path to the actual location of your raster data folders.
   - Example: BASE_INPUT_DIR = 'C:/Users/Name/Project/Data/'

5. HOW TO RUN
---------------------
Execute the scripts in the following order (Model training is required before prediction):

1. Model Training:
   Run 'train_opt.py'.
   - This reads the CSV files and trains the XGBoost models.
   - IMPORTANT: Verify that 'model_fer.pkl' and 'fer_train_features.save' are created in the 'model/' folder.

2. Fertilizer Prediction:
   Run 'grid_predict_fertilizer.py'.
   - This generates spatial maps for optimal nitrogen application across defined scenarios.

3. Irrigation Prediction:
   Run 'grid_predict_irrigation.py'.
   - This generates spatial maps for optimal irrigation requirements.

6. EXPECTED OUTPUT
---------------------
A. Model Artifacts (in 'model/' folder):
   - model_fer.pkl / model_irri.pkl (Trained Models)
   - fer_train_features.save / irri_train_features.save (Feature alignment lists)

B. Spatial Predictions (in Output Directory):
   GeoTIFF files will be generated for each year and scenario.
   - Path format: .../T-10%/fer2001.tif
   - Path format: .../T-10%/irri2001.tif