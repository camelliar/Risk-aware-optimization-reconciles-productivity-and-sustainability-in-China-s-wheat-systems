################################################################################
# Risk-aware Optimization Demo Code
# 
# Description:
# This script demonstrates the coupling of the NSGA-II algorithm (via 'mco' package)
# with the DSSAT-CERES-Wheat model. It optimizes Nitrogen (fertilizer) and 
# Irrigation management for a sample site-year to maximize yield and minimize 
# environmental losses.
#
# Requirements:
# 1. DSSAT v4.7 must be installed.
# 2. Input files (.WHX and DSSBatch.v47) must be placed in the specified directory structure.
#    Structure: ./[SiteID]/[Year]/
################################################################################

library(mco)
library(DSSAT)
library(lubridate)
library(ggplot2)
library(foreach)
library(doParallel)

# ================= CONFIGURATION =================
# Path to DSSAT Executable (PLEASE UPDATE THIS PATH FOR YOUR SYSTEM)
dssat_exe <- "C:/DSSAT47/DSCSM047.EXE" 

# Simulation settings
years <- c(2013)       # Sample Year
zds <- c(53797)        # Sample Site ID
basic_path <- getwd()  # Use current directory as base

# Optimization Bounds
N_min <- 10; N_max <- 200    # Nitrogen bounds (kg/ha)(demo)
W_min <- 10; W_max <- 400    # Irrigation bounds (mm)(demo) 

# Parallel settings
numCores <- 2 # Set to 2 for demo purposes (adjust based on hardware)

# ================= MAIN EXECUTION =================

# 1. Check if DSSAT executable exists
if (!file.exists(dssat_exe)) {
  stop(paste("DSSAT executable not found at:", dssat_exe, "\nPlease update the 'dssat_exe' path in the script."))
}

# 2. Setup Parallel Cluster
cl <- makeCluster(numCores)
registerDoParallel(cl)

print(paste("Starting optimization on", numCores, "cores..."))

# 3. Parallel Loop
results <- foreach(n = 1:length(zds), .packages = c("DSSAT", "mco"), .combine = 'rbind') %:%
  foreach(year = years, .packages = c("DSSAT", "mco"), .combine = 'rbind') %dopar% {
    
    zd <- zds[n]
    
    # Construct working directory path: ./SiteID/Year/
    # NOTE: Ensure this folder exists and contains the .WHX and DSSBatch.v47 files
    wd_path <- file.path(basic_path, as.character(zd), as.character(year))
    
    if (!dir.exists(wd_path)) {
      return(NULL) # Skip if folder doesn't exist
    }
    
    setwd(wd_path)
    
    # Identify XFile name (assumes standard DSSAT naming: 4-char ID + Year + 01.WHX)
    xfilename <- paste0(substr(zd, 2, 5), substr(year, 3, 4), '01.WHX')
    
    # Read input file
    if (!file.exists(xfilename)) { stop(paste("XFile not found:", xfilename)) }
    inputdata <- read_filex(xfilename)
    
    # --- Objective Function ---
    objective_function <- function(x) {
      fer <- x[1]  # Nitrogen Input
      irri <- x[2] # Irrigation Input
      
      # 1. Update Fertilizer (Distribution logic)
      num_days <- length(inputdata$`FERTILIZERS (INORGANIC)`$FAMN)
      if (num_days > 4 || num_days < 1) { num_days <- 2 } # Fallback safety
      
      ratio_list <- list(c(1), c(0.6, 0.4), c(0.6, 0.2, 0.2), c(0.4, 0.2, 0.2, 0.2))
      inputdata$`FERTILIZERS (INORGANIC)`$FAMN <- round(fer * ratio_list[[num_days]], 0)
      
      # 2. Update Irrigation (Automatic Irrigation Amount)
      inputdata$`SIMULATION CONTROLS`$IRAMT <- round(irri, 0)
      
      # 3. Write modified XFile
      write_filex(inputdata, xfilename, drop_duplicate_rows = TRUE, force_std_fmt = TRUE)
      
      # 4. Run DSSAT Model
      # Construct batch command: "DSCSM047.EXE B [BatchFileName]"
      batch_file <- paste0(substr(xfilename, 1, 8), ".v47") 
      # Or if you use a fixed name like DSSBatch.v47:
      batch_file_fixed <- "DSSBatch.v47"
      
      cmd <- paste(dssat_exe, "B", batch_file_fixed)
      exit_code <- system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
      
      if (exit_code != 0) { return(c(1e9, 1e9, 1e9, 1e9, 1e9)) } # Return penalty on failure
      
      # 5. Parse Outputs
      # Read N2O and Leaching from SoilNiBal.OUT
      if (file.exists('SoilNiBal.OUT')) {
        SoilNi <- readLines('SoilNiBal.OUT')
        n2o_data <- grep("N2O", SoilNi, value = TRUE)
        nl_data <- grep("N leached", SoilNi, value = TRUE)
        
        # Simple regex extraction
        n2o_val <- as.numeric(gsub(".*?([0-9]+\\.[0-9]+).*", "\\1", n2o_data[1]))
        nl_val  <- as.numeric(gsub(".*?([0-9]+\\.[0-9]+).*", "\\1", nl_data[1]))
      } else {
        n2o_val <- 0; nl_val <- 0
      }
      
      # Read Yield from Evaluate.OUT
      HWAM <- 0
      if (file.exists('Evaluate.OUT')) {
        smry <- read_output('Evaluate.OUT')
        if(nrow(smry) > 0) { HWAM <- smry$HWAMS }
      }
      
      # Read Actual Irrigation from Summary.OUT
      IRCM <- 0
      if (file.exists('Summary.OUT')) {
        smry1 <- read_output('Summary.OUT')
        if(nrow(smry1) > 0) { IRCM <- smry1$IRCM }
      }
      
      # 6. Return Objectives
      # Minimize: Fertilizer, Irrigation, N2O, Leaching, (-Yield)
      f1 <- fer
      f2 <- IRCM 
      f3 <- n2o_val
      f4 <- nl_val
      f5 <- -HWAM # Maximize Yield
      
      return(c(f1, f2, f3, f4, f5))
    }
    
    # --- Run NSGA-II ---
    # Reduced generations/popsize for demo purposes (speed up review)
    result <- nsga2(objective_function, 
                    idim = 2, 
                    odim = 5,
                    lower.bounds = c(N_min, W_min), 
                    upper.bounds = c(N_max, W_max),
                    popsize = 12,     # Small population for demo
                    generations = 5)  # Few generations for demo
    
    pareto <- as.data.frame(result$value)
    pareto_params <- as.data.frame(result$par)
    final_res <- cbind(pareto, pareto_params)
    names(final_res) <- c('Fer_Input', 'Irr_Actual', 'N2O', 'N_Leach', 'Neg_Yield', 'Fer_Optim', 'Irr_Threshold')
    
    # Save results
    out_file <- paste0(zd, '_pareto_', year, '.csv')
    write.csv(final_res, out_file, row.names = FALSE)
    
    return(final_res)
  }

stopCluster(cl)
print("Optimization demo completed.")