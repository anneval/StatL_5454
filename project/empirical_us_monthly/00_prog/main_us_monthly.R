# ===========================================================================================================
# 0. INITIALIZATION
# ===========================================================================================================

# Clear all
rm(list = ls())
set.seed(1234)

# Set paths
#path <- '~/Dropbox/GCS_SIDEsummerschool_codes/GouletCoulombe/Couture_OOS/'
path <- 'C:/Private/lpirnbac/PhD Courses/Statistical Learning/StatL_5454/project/empirical_us_monthly/'
setwd(path)

paths <- list(pro = "00_prog",
              dat = "10_data",
              fig = "20_figures",
              tab = "30_tables",
              too = "40_tools",
              rst = "50_results")

# Install needed packages
myPKGs <- c('torch', 'pracma','glmnet','ranger',
            'ggplot2','reshape2', 'stringr','MacroRF',
            'caret','doParallel','gridExtra','gbm')
InstalledPKGs <- names(installed.packages()[,'Package'])
InstallThesePKGs <- myPKGs[!myPKGs %in% InstalledPKGs]
if (length(InstallThesePKGs) > 0) install.packages(InstallThesePKGs, repos = "http://cran.us.r-project.org")

# Load libraries
library(torch)      # Neural Networks
library(glmnet)     # Lasso, Ridge and Elastic-Net
library(ranger)     # Random Forest
library(MacroRF)    # Macro Random Forest
library(caret)      # Gradient boosting machine (GBM)
library(ggplot2)    # Graphs
library(reshape2)   # Data manipulation
library(gridExtra)  # To organize results graphs
library(pracma)     # Utilities (for PCA)
library(stringr)    # String manipulation
library(doParallel) # Parallel estimation
library(foreach)    # Parallel estimation

# If you are installing "torch" for the first time, you must use the following functions :
# install_torch()

# Load US data retreiver
source(paste(paths$too, 'MakeDataUS_function.R', sep='/'))

# Factor analysis (PCA) 
source(paste(paths$too, 'EM_sw.R', sep='/'))                                                       
source(paste(paths$too, 'factor.R', sep='/'))                                                      
source(paste(paths$too, 'ICp2.R', sep='/'))

# Neural Network function
source(paste(paths$too, 'MLP_function_v6b.R', sep='/'))

# Out-of-sample functions
source(paste(paths$pro, 'OOS_models.R', sep='/'))

# Number of CPU for the estimation (if you don't want to use the parallel setting : ncores <- NA)
ncores <- NA
torch_set_num_threads(1)

# ===========================================================================================================
# 1. OUT-OF-SAMPLE AND MODEL'S PARAMETERS
# ===========================================================================================================

## OOS Parameters ----------------------------------------------------------------
OOS_params <- list()

# Target names from FRED DB
OOS_params$targetName <- c("CPIAUCSL")  

# Change the transformation code of the target, "NA" to keep FRED's code
OOS_params$target_tcode <- c(5) 

# Forecasting horizons (in months)
OOS_params$horizon <- c(3, 12)

# Out-of-sample starting date
OOS_params$OOS_starting_date <- "3/1/2015" # change appropriately

# Number of FRED's factors
OOS_params$nFac <- 5 

# Number of target lags
OOS_params$lagY <- 2                          

# Number of regressors lags (factors included)
OOS_params$lagX <-                          

# Create MARX
OOS_params$lagMARX <- NA    

# Number of folds for CV
OOS_params$nfolds <- 5

# How many quarters between hyperparameters CV (in quarters)
OOS_params$reEstimate <- 20 # each 5 years 

# Which models to used ? Possible choice c("AR, BIC", "ARDI, BIC","LASSO","RIDGE","ELASTIC-NET","RF","GBM","NN,"AR-RF")
OOS_params$model_list <- c("AR, BIC", "ARDI, BIC","LASSO","RIDGE","RF","GBM","NN","AR-RF")

# Folder name in 50_results
OOS_params$save_path = "demo"

## Hyperparamters ----------------------------------------------------------------


# Elastic - Net hyperparameters (CV)
OOS_params$EN_hyps <- list(alpha_range = round(seq(0.01,0.99, length = 100),4))


# Boosting hyperparameters (CV)
OOS_params$Boosting_hyps <- list(man_grid = expand.grid(n.trees = c(seq(25, 700, by = 100)),
                                                        interaction.depth = c(3,5),
                                                        shrinkage = c(0.01),
                                                        n.minobsinnode = c(10)),
                                 fitControl = trainControl(method = "cv",
                                                           number = OOS_params$nfolds,
                                                           search = "grid"))

# Random Forest hyperparameters
OOS_params$RF_hyps <- list(num.trees = 500,
                           min.node.size = 3,
                           mtry = 1/3)

# Macro Random Forest hyperparamaters
OOS_params$MacroRF_hyps <- list(x_pos = c(2,3),
                                B = 20,
                                mtry_frac = 0.15,
                                minsize = 15,
                                block_size = 8)

# Neural network hyperparameters
OOS_params$nn_hyps <- list(n_features=NA,
                           nodes=rep(100,5),      # same number of nodes in every layers
                           patience=10,           # Return the best model
                           epochs=100,
                           lr=0.001,
                           tol=0.01,
                           show_train=3,          # 1=show each bootstrap loss, 2=progress bar, 3+=show nothing
                           num_average=5,
                           dropout_rate=0.2,
                           sampling_rate = 0.75,
                           batch_size = 32,
                           num_batches = NA)

# ===========================================================================================================
# 2. PARALLEL ESTIMATION
# ===========================================================================================================

# Create all possible of targets and horizons
combn <- list(var = c(1:length(OOS_params$targetName)),
              hor = OOS_params$horizon) 
all_options <- expand.grid(combn)
all_options <- all_options[order(all_options$var,all_options$hor, decreasing = F),]
rownames(all_options) <- c()

start <- Sys.time()

# Parallel estimation
if(!is.na(ncores)) {
  
  # Start parallel clustering
  cl <- makeCluster(ncores)
  registerDoParallel(cl) # Shows the number of Parallel Workers to be used
  
  foreach(i=c(1:nrow(all_options))) %dopar% Forecast_all(it_pos = i,
                                                         all_options = all_options,
                                                         paths = paths,
                                                         OOS_params = OOS_params,
                                                         seed = 124)
  
  stopImplicitCluster()

# Single core estimation    
} else{
  
  for (i in 1:nrow(all_options)) {
    
    Forecast_all(it_pos = i,
                 all_options = all_options,
                 paths = paths,
                 OOS_params = OOS_params,
                 seed = 124) 
  }
  
}
end <- Sys.time()
end-start

# ===========================================================================================================
# 3. RESULTS
# ===========================================================================================================

results <- process_results(paths,OOS_params = OOS_params, benchmark = "AR, BIC") # To use plain MSE put benchmark = NA

# Show MSE ratio
round(results$mse_table,3)
round(results$mse_table_2019,3)

# MSE ratio barplots and predictions plots (the plots are saved in 20_Figures)
mse_barplot_h1 <- list()
#mse_barplot_h4 <- list()
pred_plot_h1 <- list()
#pred_plot_h4 <- list()

for(var in 1:dim(results$mse_table)[3]) {
  
  # MSE
  mse_barplot_h1[[var]] <- quick_barplot(results, hor = 1, var = var)
  #mse_barplot_h4[[var]] <- quick_barplot(results, hor = 4, var = var)
  
  # Predictions
  pred_plot_h1[[var]] <- quick_plot(results, hor = 1, var = var)
 # pred_plot_h4[[var]] <- quick_plot(results, hor = 4, var = var)
  
  # Put the 2 graphs together
  p <- arrangeGrob(pred_plot_h1[[var]],mse_barplot_h1[[var]],
                   nrow = 2, ncol = 1)
  ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",1,".png")
  ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
  #p <- arrangeGrob(pred_plot_h4[[var]],mse_barplot_h4[[var]],
                   #nrow = 2, ncol = 1)
  #ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",4,".png")
  #ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
}

# Quick view, you need to choose the position of the target you want to see
# targets order : (1) "CPIAUCSL", (2) "UNRATE", (3) "HOUST", (4) "PAYEMS", (5) "GDPC1"
mse_barplot_h1[[1]]
pred_plot_h1[[1]]
mse_barplot_h1[[3]]
pred_plot_h1[[3]]
