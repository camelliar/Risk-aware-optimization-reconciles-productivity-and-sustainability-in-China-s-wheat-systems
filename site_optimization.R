#-------------------------------------------------------------------------------
# 附录 A: 基于DSSAT与NSGA-II的农业管理多目标优化框架
#-------------------------------------------------------------------------------
library(DSSAT)      # DSSAT模型接口
library(mco)        # 多目标优化算法 (NSGA-II)
library(foreach)    # 并行循环框架
library(doParallel) # 并行计算后端

# --- 基础环境与路径 ---
path_dssat_executable <- "C:/DSSAT47/DSCSM047.EXE"
path_project_directory <- "path/to/your/simulation/parent/directory"
number_of_cpu_cores_to_use <- 12 # or dynamically detect: parallel::detectCores() - 2

# --- 待处理任务定义 ---
# 站点元数据应从外部文件加载，此处仅为结构示例
# sites_metadata_table <- read.csv("path/to/sites_metadata.csv")
sites_metadata_table <- data.frame(
  site_id = c("Site_A", "Site_B", "Site_C", "..."),
  site_code = c("S01", "S02", "S03", "..."),
  baseline_fertilizer_rate = c(220, 235, 210, "...") # 基准施肥量
)
years_vector <- 2013:2018

# --- 构建最终的配置对象 (CONFIG) ---
CONFIG <- list(
  paths = list(
    dssat_executable = path_dssat_executable,
    project_root = path_project_directory
  ),
  task_definition = list(
    sites = sites_metadata_table,
    years = years_vector
  ),
  nsga2_hyperparameters = list(
    population_size = nsga2_population_size,
    generations = nsga2_generations,
    decision_var_dims = num_decision_variables,
    objective_func_dims = num_objective_functions
  ),
  decision_variable_bounds = list(
    lower = c(fertilizer = fertilizer_lower_bound, irrigation = irrigation_lower_bound),
    upper_rules = list(
      irrigation_max = irrigation_upper_bound_absolute,
      fertilizer_offset = fertilizer_upper_bound_offset
    )
  ),
  agronomy_rules = list(
    fertilizer_ratios = list(
      `1` = ratio_for_1_fert_application,
      `2` = ratio_for_2_fert_applications,
      `3` = ratio_for_3_fert_applications,
      `4` = ratio_for_4_fert_applications
    )
  ),
  parallel_setup = list(
    num_cores = number_of_cpu_cores_to_use
  )
)


#-------------------------------------------------------------------------------
# 3. 核心功能函数: DSSAT-NSGAII 耦合优化器
#-------------------------------------------------------------------------------
run_single_optimization_task <- function(site_metadata, target_year, cfg) {
  
  # --- 步骤 1: 构建工作环境 ---
  working_directory <- file.path(cfg$paths$project_root, site_metadata$site_code, target_year)
  setwd(working_directory)
  
  # --- 步骤 2: 准备DSSAT输入和优化边界 ---
  xfile_name <- sprintf("%s%02d01.WHX", substr(site_metadata$site_id, 2, 5), target_year %% 100)
  base_dssat_input <- read_filex(xfile_name)
  
  current_upper_bounds <- c(
    fertilizer = site_metadata$baseline_fertilizer_rate + cfg$decision_variable_bounds$upper_rules$fertilizer_offset,
    irrigation = cfg$decision_variable_bounds$upper_rules$irrigation_max
  )
  
  # --- 步骤 3: 定义与DSSAT模型交互的目标函数 ---
  objective_function_dssat <- function(decision_vars) {
    tryCatch({
      current_fertilizer_total <- decision_vars[1]
      current_irrigation_amount <- decision_vars[2]
      
      dssat_input_modified <- base_dssat_input
      num_fert_applications <- length(dssat_input_modified$`FERTILIZERS (INORGANIC)`$FAMN)
      fert_ratios <- cfg$agronomy_rules$fertilizer_ratios[[as.character(num_fert_applications)]]
      if (is.null(fert_ratios)) {
        return(rep(1e9, cfg$nsga2_hyperparameters$objective_func_dims))
      }
      
      dssat_input_modified$`FERTILIZERS (INORGANIC)`$FAMN <- round(current_fertilizer_total * fert_ratios, 0)
      dssat_input_modified$`SIMULATION CONTROLS`$IRAMT <- round(current_irrigation_amount, 0)
      write_filex(dssat_input_modified, xfile_name, drop_duplicate_rows = TRUE, force_std_fmt = TRUE)
      
      batch_file_name <- paste0(substr(xfile_name, 1, 8), "DSSBatch.v47")
      system_command <- paste(cfg$paths$dssat_executable, "B", batch_file_name)
      system(system_command, ignore.stdout = TRUE, ignore.stderr = TRUE)
      
      soil_ni_output <- readLines('SoilNiBal.OUT')
      summary_output <- read_output('Evaluate.OUT')
      summary_all <- read_output('Summary.OUT')
      
      n2o_emission <- as.numeric(gsub(".*?([0-9]+\\.[0-9]+).*", "\\1", grep("N2O", soil_ni_output, value = TRUE)[1]))
      n_leached <- as.numeric(gsub(".*?([0-9]+\\.[0-9]+).*", "\\1", grep("N leached", soil_ni_output, value = TRUE)[1]))
      
      return(c(
        obj_fertilizer = current_fertilizer_total,
        obj_irrigation = summary_all$IRCM,
        obj_n2o        = n2o_emission,
        obj_n_leached  = n_leached,
        obj_yield_neg  = -summary_output$HWAMS 
      ))
    }, error = function(e) {
      return(rep(1e9, cfg$nsga2_hyperparameters$objective_func_dims))
    })
  }
  
  # --- 步骤 4: 执行NSGA-II优化算法 ---
  optimization_result <- nsga2(
    fn = objective_function_dssat, 
    idim = cfg$nsga2_hyperparameters$decision_var_dims,
    odim = cfg$nsga2_hyperparameters$objective_func_dims,
    lower.bounds = cfg$decision_variable_bounds$lower,
    upper.bounds = current_upper_bounds,
    popsize = cfg$nsga2_hyperparameters$population_size,
    generations = cfg$nsga2_hyperparameters$generations
  )
  
  # --- 步骤 5: 格式化、保存并返回结果 ---
  pareto_objectives <- as.data.frame(optimization_result$value)
  pareto_decisions <- as.data.frame(optimization_result$par)
  
  names(pareto_objectives) <- c('objective_fertilizer', 'objective_irrigation', 'objective_n2o', 'objective_nleached', 'objective_yield_neg')
  names(pareto_decisions) <- c('decision_fertilizer', 'decision_irrigation')
  
  final_pareto_set <- cbind(pareto_decisions, pareto_objectives)
  final_pareto_set$objective_yield_neg <- -final_pareto_set$objective_yield_neg
  names(final_pareto_set)[names(final_pareto_set) == "objective_yield_neg"] <- "objective_yield"
  
  output_filename <- sprintf("%s_%s_pareto_solutions_%d.csv", site_metadata$site_id, site_metadata$site_code, target_year)
  write.csv(final_pareto_set, output_filename, row.names = FALSE)
  
  return(final_pareto_set)
}

#-------------------------------------------------------------------------------
# 4. 主执行流程: 并行调度优化任务
#-------------------------------------------------------------------------------

# --- 初始化并行计算环境 ---
parallel_cluster <- makeCluster(CONFIG$parallel_setup$num_cores)
registerDoParallel(parallel_cluster)
on.exit(stopCluster(parallel_cluster), add = TRUE)

cat(sprintf("并行计算环境已启动，共使用 %d 个CPU核心。\n", getDoParWorkers()))

# --- 使用嵌套的foreach进行双重并行化 ---
optimization_log <- foreach(
  i = 1:nrow(CONFIG$task_definition$sites),
  .combine = 'c'
) %:%
  foreach(
    year = CONFIG$task_definition$years,
    .packages = c("DSSAT", "mco"),
    .export = c("CONFIG", "run_single_optimization_task"),
    .combine = 'c'
  ) %dopar% {
    
    current_site_info <- CONFIG$task_definition$sites[i, ]
    run_single_optimization_task(current_site_info, year, CONFIG)
    
    sprintf("任务成功: 站点[%s] - 年份[%d]", current_site_info$site_id, year)
  }

# --- 结束 ---
stopCluster(parallel_cluster)
cat("\n所有优化任务已完成，并行集群已安全关闭。\n")
print("任务执行日志:")
print(optimization_log)