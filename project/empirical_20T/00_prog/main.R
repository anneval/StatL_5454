# ===========================================================================================================
# 0. INITIALIZATION
# ===========================================================================================================

# Clear all
rm(list = ls())
set.seed(1234)

# Set paths
path <- 'C:/Users/avalder/OneDrive - WU Wien/Documents/Study/SoSe_24/Statistical Learning/assignments/StatL_5454/project/empirical/'
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
source(paste(paths$too, 'MakeDataUK_function.R', sep='/'))

# Factor analysis (PCA) 
source(paste(paths$too, 'EM_sw.R', sep='/'))                                                       
source(paste(paths$too, 'factor.R', sep='/'))                                                      
source(paste(paths$too, 'ICp2.R', sep='/'))

# Neural Network function
source(paste(paths$too, 'MLP_function_v6b.R', sep='/'))

# Out-of-sample functions
source(paste(paths$pro, 'OOS_models.R', sep='/'))

# Number of CPU for the estimation (if you don't want to use the parallel setting : ncores <- NA)
ncores <- 4
torch_set_num_threads(1)

# ===========================================================================================================
# 1. OUT-OF-SAMPLE AND MODEL'S PARAMETERS
# ===========================================================================================================

## OOS Parameters ----------------------------------------------------------------
OOS_params <- list()

# Target names from FRED DB
# OOS_params$targetName <- c("CPIAUCSL","UNRATE",  
#                            "HOUST", "PAYEMS",
#                            "GDPC1")  
OOS_params$targetName <- c("CPI_ALL")

# Change the transformation code of the target, "NA" to keep FRED's code
OOS_params$target_tcode <- c(NA) # not really needed here since we use the balanced UK data set 

# Forecasting horizons (in quarter), month here 
OOS_params$horizon <- c(3,12)
#OOS_params$horizon <- c(1,3,12)


# Out-of-sample starting date
OOS_params$OOS_starting_date <- "2015-01-01"

# Number of FRED's factors
OOS_params$nFac <- 8

# Number of target lags
OOS_params$lagY <- 6                          

# Number of regressors lags (factors included)
OOS_params$lagX <- 6                         

# Create MARX
OOS_params$lagMARX <- NA    

# Number of folds for CV
OOS_params$nfolds <- 5

# How many quarters between hyperparameters CV (in quarters)
OOS_params$reEstimate <- 60 # each 5 years 

# Which models to used ? Possible choice c("AR, BIC", "ARDI, BIC","LASSO","RIDGE","ELASTIC-NET","RF","GBM","NN,"AR-RF")
#OOS_params$model_list <- c("AR, BIC", "ARDI, BIC","LASSO","RIDGE","RF","GBM","NN","AR-RF")

OOS_params$model_list <- c("AR, BIC","RF","AR-RF","RF-MAF","FA-ARRF") #RF_MAF, RF, FA-ARRF

# Folder name in 50_results
OOS_params$save_path = "demo"

# number of MAF per variable
OOS_params$nMAF <- 2

# lags of "non-factor" exogenous variables (i.e. all the fred variables that are not "target")
OOS_params$lagOtherX <- 1

## Hyperparamters ----------------------------------------------------------------


# Elastic - Net hyperparameters (CV)
#OOS_params$EN_hyps <- list(alpha_range = round(seq(0.01,0.99, length = 100),4))


# Boosting hyperparameters (CV)
# OOS_params$Boosting_hyps <- list(man_grid = expand.grid(n.trees = c(seq(25, 700, by = 100)),
#                                                         interaction.depth = c(3,5),
#                                                         shrinkage = c(0.01),
#                                                         n.minobsinnode = c(10)),
#                                  fitControl = trainControl(method = "cv",
#                                                            number = OOS_params$nfolds,
#                                                            search = "grid"))

# Random Forest a) hyperparameters
OOS_params$RF_hyps <- list(num.trees = 500,
                           min.node.size = 3,
                           mtry = 1/3)



# Random Forest b) hyperparameters
OOS_params$RF_MAF_hyps <- list(num.trees = 500,
                           min.node.size = 3,
                           mtry = 1/3)


# NOTE for AR-RF and FA-AR-RF: user provides variable names and OOS_models(.) determines the corresponding
#       (column) position in the data

# Macro Random Forest (AR) hyperparamaters
temp_x <- paste0("L_", 0:(OOS_params$lagY-1), "y") # names of lagged y-values

OOS_params$MacroRF_hyps <- list(x_vars = temp_x,
                                #x_pos = c(2,3,4,5,6,7),  
                                B = 100, 
                                mtry_frac = 0.15,
                                minsize = 15,
                                block_size = 24) # block size is 24 in monthly i.e. 2 years


# Macro Random Forest (FA-AR) hyperparamaters
temp_x <- c(temp_x, "L0_F_UK1", "L0_F_UK2") # names of lagged y-values and first two factors

OOS_params$FA_MacroRF_hyps <- list(x_vars = temp_x,
                                   #x_pos = c(2,3,4,5,6,7,26,27), 
                                   B = 100, # more trees?
                                   mtry_frac = 0.15,
                                   minsize = 15,
                                   block_size = 24) # block size is 

# OLD PART
# # Macro Random Forest hyperparamaters
# OOS_params$MacroRF_hyps <- list(x_pos = c(2,3,4,5,6,7), #### für den ARRF & month lags i.e. 2 Quarter before now monthly 
#                                 B = 50,
#                                 mtry_frac = 0.15,
#                                 minsize = 15,
#                                 block_size = 24) # block size is 24 in monthly i.e. 2 years
# 
# # Macro Random Forest hyperparamaters
# OOS_params$FA_MacroRF_hyps <- list(x_pos = c(2,3,4,5,6,7,26,27), #### für den ARRF & month lags i.e. 2 Quarter before now monthly 
#                                 B = 50,
#                                 mtry_frac = 0.15,
#                                 minsize = 15,
#                                 block_size = 24) # block size is 24 in monthly i.e. 2 years

# Neural network hyperparameters
# OOS_params$nn_hyps <- list(n_features=NA,
#                            nodes=rep(100,5),      # same number of nodes in every layers
#                            patience=10,           # Return the best model
#                            epochs=100,
#                            lr=0.001,
#                            tol=0.01,
#                            show_train=3,          # 1=show each bootstrap loss, 2=progress bar, 3+=show nothing
#                            num_average=5,
#                            dropout_rate=0.2,
#                            sampling_rate = 0.75,
#                            batch_size = 32,
#                            num_batches = NA)

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

betas_mrffa_H3 <- results$mrf_fa_store[['3']][['betas']]
betas_mrf_H3 <- results$mrf_store[['3']][['betas']]
betas_mrffa_H12 <- results$mrf_fa_store[['12']][['betas']]
betas_mrf_H12 <- results$mrf_store[['12']][['betas']]


# bleibt geht für alle 
persistence <- rowSums(betas_mrffa[,2:7])
persistence_p <- ts.plot(as.ts(persistence))
F1_p <-  ts.plot(as.ts(betas_mrffa[,8]))
F2_p <-  ts.plot(as.ts(betas_mrffa[,9]))
intercept_p <- ts.plot(as.ts(betas_mrffa[,1]))


# Show MSE ratio
round(results$mse_table,3)
round(results$mse_table_2019,3)

# MSE ratio barplots and predictions plots (the plots are saved in 20_Figures)
mse_barplot_h3 <- list()
mse_barplot_h12 <- list()
pred_plot_h3 <- list()
pred_plot_h12 <- list()

for(var in 1:dim(results$mse_table)[3]) {
  
  # MSE
  mse_barplot_h3[[var]] <- quick_barplot(results, hor = 3, var = var)
  mse_barplot_h12[[var]] <- quick_barplot(results, hor = 12, var = var)
  
  # Predictions
  pred_plot_h3[[var]] <- quick_plot(results, hor = 3, var = var)
  pred_plot_h12[[var]] <- quick_plot(results, hor = 12, var = var)
  
  # Put the 2 graphs together
  p <- arrangeGrob(pred_plot_h3[[var]],mse_barplot_h3[[var]],
                   nrow = 2, ncol = 1)
  ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",1,".png")
  ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
  p <- arrangeGrob(pred_plot_h12[[var]],mse_barplot_h12[[var]],
                   nrow = 2, ncol = 1)
  ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",4,".png")
  ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
}

# Quick view, you need to choose the position of the target you want to see
# targets order : (1) "CPIAUCSL", (2) "UNRATE", (3) "HOUST", (4) "PAYEMS", (5) "GDPC1"
mse_barplot_h3[[1]]
pred_plot_h3[[1]]
mse_barplot_h12[[1]]
pred_plot_h12[[1]]

