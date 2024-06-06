# ===========================================================================================================
# 0. INITIALIZATION
# ===========================================================================================================

# Clear all
rm(list = ls())
set.seed(1234)

# Set paths
#path <- '~/Dropbox/GCS_SIDEsummerschool_codes/GouletCoulombe/Couture_OOS/'
#path <- 'C:/Private/lpirnbac/PhD Courses/Statistical Learning/StatL_5454/project/empirical_us_monthly/'
path <- 'C:/Users/mhochhol/OneDrive - WU Wien/Desktop/empirical_us_monthly/empirical_us_monthly/'

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
OOS_params$target_tcode <- c(5) # make stationary

# Forecasting horizons (in months)
OOS_params$horizon <- c(3, 12)

# Out-of-sample starting date
OOS_params$OOS_starting_date <- "1/1/2015" # start of OOS period

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

# How many quarters between hyperparameters CV (in months)
OOS_params$reEstimate <- 60 # each 5 years 

# Which models to used ? 
OOS_params$model_list <- c("AR, BIC","RF","AR-RF","RF-MAF","FA-ARRF") 
# TO DO: adjust OOS_models


# Folder name in 50_results
OOS_params$save_path = "demo"

# starting date of training sample
OOS_params$training_starting_date <- "1998-01-01"

# number of MAF per variable
OOS_params$nMAF <- 2

# lags of "non-factor" exogenous variables (i.e. all the fred variables that are not "target")
OOS_params$lagOtherX <- 1



## Hyperparamters ----------------------------------------------------------------


# # Elastic - Net hyperparameters (CV)
# OOS_params$EN_hyps <- list(alpha_range = round(seq(0.01,0.99, length = 100),4))
# 
# 
# # Boosting hyperparameters (CV)
# OOS_params$Boosting_hyps <- list(man_grid = expand.grid(n.trees = c(seq(25, 700, by = 100)),
#                                                         interaction.depth = c(3,5),
#                                                         shrinkage = c(0.01),
#                                                         n.minobsinnode = c(10)),
#                                  fitControl = trainControl(method = "cv",
#                                                            number = OOS_params$nfolds,
#                                                            search = "grid"))

# Random Forest hyperparameters
OOS_params$RF_hyps <- list(num.trees = 500,
                           min.node.size = 3,
                           mtry = 1/3)



# Random Forest (MAF) hyperparameters
OOS_params$RF_MAF_hyps <- list(num.trees = 500,
                               min.node.size = 3,
                               mtry = 1/3)


# Macro Random Forest (AR) hyperparamaters
# NOTE: user provides variable names and OOS_models(.) determines the corresponding
#       (column) position in the data
temp_x <- paste0("L_", 0:(OOS_params$lagY-1), "y") # names of lagged y-values

OOS_params$MacroRF_hyps <- list(x_vars = temp_x,
                                #x_pos = c(2,3,4,5,6,7),  
                                B = 50, # more trees?
                                mtry_frac = 0.15,
                                minsize = 15,
                                block_size = 24) # block size is 24 in monthly i.e. 2 years


# Macro Random Forest (FA-AR) hyperparamaters
temp_x <- c(temp_x, "L0_F_US1", "L0_F_US2") # names of lagged y-values and first two factors
# QUESTION: should we use L1_F_.. instead?
OOS_params$FA_MacroRF_hyps <- list(x_vars = temp_x,
                                   #x_pos = c(2,3,4,5,6,7,26,27), 
                                   B = 50, # more trees?
                                   mtry_frac = 0.15,
                                   minsize = 15,
                                   block_size = 24) # block size is 


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

# RUN ONCE
# start <- Sys.time()
# 
# # Parallel estimation
# if(!is.na(ncores)) {
#   
#   # Start parallel clustering
#   cl <- makeCluster(ncores)
#   registerDoParallel(cl) # Shows the number of Parallel Workers to be used
#   
#   foreach(i=c(1:nrow(all_options))) %dopar% Forecast_all(it_pos = i,
#                                                          all_options = all_options,
#                                                          paths = paths,
#                                                          OOS_params = OOS_params,
#                                                          seed = 124)
#   
#   stopImplicitCluster()
# 
# # Single core estimation    
# } else{
#   
#   for (i in 1:nrow(all_options)) {
#     
#     Forecast_all(it_pos = i,
#                  all_options = all_options,
#                  paths = paths,
#                  OOS_params = OOS_params,
#                  seed = 124) 
#   }
#   
# }
# end <- Sys.time()
# end-start

# ===========================================================================================================
# 3. RESULTS
# ===========================================================================================================

results <- process_results(paths,OOS_params = OOS_params, benchmark = "AR, BIC") # To use plain MSE put benchmark = NA

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


#### --- betas --- ###

# AR-RF
mrfs <- results$mrf_store
mean_betas_mrf <- lapply(mrfs, function(x) x$betas)
mean_betas_mrf <- lapply(mean_betas_mrf, function(x) y <- cbind(x, rowSums(x[,2:7]))) # add persistence = sum over auto-regressive coefficients
mean_output_mrf <- lapply(mean_betas_mrf, function(x){
  y <- x[,c(1,8)] # select intercept and persistence
  colnames(y) <- c("intercept", "persistence")
  return(y)
})

# FA-AR-RF
fa_mrfs <- results$fa_mrf_store
mean_betas_fa_mrf <- lapply(fa_mrfs, function(x) x$betas)
mean_betas_fa_mrf <- lapply(mean_betas_fa_mrf, function(x) y <- cbind(x, rowSums(x[,2:7]))) # add persistence = sum over auto-regressive coefficients
mean_output_fa_mrf <- lapply(mean_betas_fa_mrf, function(x){
  y <- x[,c(1,10,8,9)] # select intercept and persistence
  colnames(y) <- c("intercept", "persistence","real_factor", "forward_factor")
  return(y)
})


#########




betas_mrffa_H3 <- results$fa_mrf_store[['3']][['betas']]
betas_mrf_H3 <- results$mrf_store[['3']][['betas']]
betas_mrffa_H12 <- results$fa_mrf_store[['12']][['betas']]
betas_mrf_H12 <- results$mrf_store[['12']][['betas']]

# Quantiles
quant_mrffa_H3 <- results$fa_mrf_store[['3']][['betas.draws.raw']]
quant_mrf_H3 <- results$mrf_store[['3']][['betas.draws.raw']]
quant_mrffa_H3_per <- apply(quant_mrffa_H3[,2:7,], 3, rowSums)
quant_mrf_H3_per <- apply(quant_mrf_H3[,2:7,], 3, rowSums)

quantiles_mrffa_H3 <- apply(quant_mrffa_H3, c(1,2), function(x) quantile(x, probs = c(0.1, 0.16, 0.84, 0.9)))
quantiles_mrf_H3 <- apply(quant_mrf_H3, c(1,2), function(x) quantile(x, probs = c(0.1,  0.16, 0.84, 0.9)))
quantiles_mrffa_H3_per <- apply(quant_mrffa_H3_per, 1, function(x) quantile(x, probs = c(0.1, 0.16, 0.84, 0.9)))
quantiles_mrf_H3_per <- apply(quant_mrf_H3_per, 1, function(x) quantile(x, probs = c(0.1,  0.16, 0.84, 0.9)))

quant_mrffa_H12 <- results$fa_mrf_store[['12']][['betas.draws.raw']]
quant_mrf_H12 <- results$mrf_store[['12']][['betas.draws.raw']]
quant_mrffa_H12_per <- apply(quant_mrffa_H12[,2:7,], 3, rowSums)
quant_mrf_H12_per <- apply(quant_mrf_H12[,2:7,], 3, rowSums)

quantiles_mrffa_H12 <- apply(quant_mrffa_H12, c(1,2), function(x) quantile(x, probs = c(0.1, 0.16, 0.84, 0.9)))
quantiles_mrf_H12 <- apply(quant_mrf_H12, c(1,2), function(x) quantile(x, probs = c(0.1,  0.16, 0.84, 0.9)))
quantiles_mrffa_H12_per <- apply(quant_mrffa_H12_per, 1, function(x) quantile(x, probs = c(0.1, 0.16, 0.84, 0.9)))
quantiles_mrf_H12_per <- apply(quant_mrf_H12_per, 1, function(x) quantile(x, probs = c(0.1,  0.16, 0.84, 0.9)))




png("20_figures/MRFFA_H3.png", height=650, width=1000)
par(mfrow = c(2, 2))

# MRFFA- H3

var <- betas_mrffa_H3[,1]
var_q <- quantiles_mrffa_H3[,,1]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n')
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Intercept-H3")
abline(v = 205, lty = 2, col = "black")


var <- rowSums(betas_mrffa_H3[,2:7])
var_q <- quantiles_mrffa_H3_per[,]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Persistence-H3")
abline(v = 205, lty = 2, col = "black")


var <- betas_mrffa_H3[,8]
var_q <- quantiles_mrffa_H3[,,8]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Real Activity Factor-H3")
abline(v = 205, lty = 2, col = "black")


var <- betas_mrffa_H3[,9]
var_q <- quantiles_mrffa_H3[,,9]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Forward-Looking Factor-H3")
abline(v = 205, lty = 2, col = "black")

dev.off()

# MRFFA- H12

png("20_figures/MRFFA_H12.png", height=650, width=1000)
par(mfrow = c(2, 2))

var <- betas_mrffa_H12[,1]
var_q <- quantiles_mrffa_H12[,,1]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Intercept-H12")
abline(v = 205, lty = 2, col = "black")

var <- rowSums(betas_mrffa_H12[,2:7])
var_q <- quantiles_mrffa_H12_per[,]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Persistence-H12")
abline(v = 205, lty = 2, col = "black")

var <- betas_mrffa_H12[,8]
var_q <- quantiles_mrffa_H12[,,8]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Real Activity Factor-H12")
abline(v = 205, lty = 2, col = "black")

var <- betas_mrffa_H12[,9]
var_q <- quantiles_mrffa_H12[,,9]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRFFA-Forward-Looking Factor-H12")
abline(v = 205, lty = 2, col = "black")

dev.off()

# MRF- H3

png("20_figures/MRF_H3.png", height=325, width=1000)
par(mfrow = c(1, 2))


var <- betas_mrf_H3[,1]
var_q <- quantiles_mrf_H3[,,1]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRF-Intercept-H3")
abline(v = 205, lty = 2, col = "black")


var <- rowSums(betas_mrf_H3[,2:7])
var_q <- quantiles_mrf_H3_per[,]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2019, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRF-Persistence-H3")
abline(v = 205, lty = 2, col = "black")

dev.off()

# MRF- H12

png("20_figures/MRF_H12.png", height=325, width=1000)
par(mfrow = c(1, 2))


var <- betas_mrf_H12[,1]
var_q <- quantiles_mrf_H12[,,1]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRF-Intercept-H12")
abline(v = 205, lty = 2, col = "black")

var <- rowSums(betas_mrf_H12[,2:7])
var_q <- quantiles_mrf_H12_per[,]
y_range <- range(c(var_q[1,], var_q[4,]))
plot(1:length(var), as.ts(var), type="n", xlab="", ylab="", ylim=y_range, xaxt='n' )
x <- 1:length(var)
polygon(c(x, rev(x)), c(var_q[1,], rev(var_q[4,])), col = "darkgray")
polygon(c(x, rev(x)), c(var_q[2,], rev(var_q[3,])), col = "lightgray")
lines(x, as.ts(var), lwd = 2, col = "black")
years <- seq(1998, 2017, by = 2)
axis(1, at = seq(1, length(x), by = 24), labels = years)
title("MRF-Persistence-H12")
abline(v = 205, lty = 2, col = "black")

dev.off()
par(mfrow = c(1, 1))






