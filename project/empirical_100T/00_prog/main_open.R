# ===========================================================================================================
# 0. INITIALIZATION
# ===========================================================================================================

# Clear all
rm(list = ls())
set.seed(1234)

# Set paths
path <- 'C:/Users/avalder/OneDrive - WU Wien/Documents/Study/SoSe_23/summerschool/PCG/Codes/MRF_and_ML_OOS_exercise/'
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
ncores <- 1
torch_set_num_threads(1)

# ===========================================================================================================
# 1. OUT-OF-SAMPLE AND MODEL'S PARAMETERS
# ===========================================================================================================

## OOS Parameters ----------------------------------------------------------------
OOS_params <- list()

# Target names from FRED DB
OOS_params$targetName <- c("GDPC1","CPIAUCSL")[1]

# Change the transformation code of the target, "NA" to keep FRED's code
OOS_params$target_tcode <- c(5,5) 

# Forecasting horizons (in quarter)
OOS_params$horizon <- c(1,4)

# Out-of-sample starting date
OOS_params$OOS_starting_date <- "3/1/2015"

# Number of FRED's factors
OOS_params$nFac <- 5 

# Number of target lags
OOS_params$lagY <- 2                          

# Number of regressors lags (factors included)
OOS_params$lagX <- 1                         

# Create MARX
OOS_params$lagMARX <- NA    

# Number of folds for CV
OOS_params$nfolds <- 5

# How many quarters between hyperparameters CV (in quarters)
OOS_params$reEstimate <- 20 # each 5 years 

# Which models to used ? Possible choice c("AR, BIC", "ARDI, BIC","LASSO","RIDGE","ELASTIC-NET","RF","GBM","NN,"AR-RF")
OOS_params$model_list <- c("AR, BIC","AR-RF") #"ARDI, BIC","LASSO","RIDGE","RF","GBM","NN"

# Folder name in 50_results
OOS_params$save_path = "demo_v2"

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
# 2.ESTIMATION
# ===========================================================================================================

# Create all possible of targets and horizons
combn <- list(var = c(1:length(OOS_params$targetName)),
              hor = OOS_params$horizon) 
all_options <- expand.grid(combn)
all_options <- all_options[order(all_options$var,all_options$hor, decreasing = F),]
rownames(all_options) <- c()

# Choice of variable and horizon
var <- 1
hor <- 4

# Variable and Horizon to forecast #############################################################
# ==============================================================================================

# Get the data
USdata <- MakeDataUS(path = paste0(path,paths$dat,"/"), targetName = OOS_params$targetName[var], h = hor,
                     nFac = OOS_params$nFac, lag_y = OOS_params$lagY, lag_f = OOS_params$lagX, lag_marx = OOS_params$lagMARX,
                     versionName = "current",
                     download = T, EM_algorithm=T, EM_last_date=NA,
                     frequency = 2, target_new_tcode=OOS_params$target_tcode[var])
data <- USdata[[1]]$lagged_data

# Estimation parameters ########################################################################
# ==============================================================================================

# Parameters
H_max <- c(max(all_options$hor))
H <- unique(all_options$hor)
nfolds <- OOS_params$nfolds
reEstimate <- OOS_params$reEstimate
OOS_date <- OOS_params$OOS_starting_date
model_list <- OOS_params$model_list
lagY <- OOS_params$lagY
lagX <- OOS_params$lagX
nfac <- OOS_params$nFac

# Out of sample start
OOS <- nrow(data) - which(rownames(data) == OOS_date) + 1

## Hyperparamters ----------------------------------------------------------------

# Elastic - Net hyperparameters (CV)
EN_hyps <<- OOS_params$EN_hyps

# Random Forest hyperparameters
RF_hyps <<- OOS_params$RF_hyps

# GBM hyperparameters (CV)
Boosting_hyps <<- OOS_params$Boosting_hyps

# Neural network hyperparameters 
nn_hyps <<- OOS_params$nn_hyps

# Macro Random Forest hyperparameters
MacroRF_hyps <<- OOS_params$MacroRF_hyps

## Storage -----------------------------------------------------------------------

prediction_oos <- array(data = NA, dim = c(OOS,length(unique(all_options$var)),H_max,length(model_list)))
rownames(prediction_oos) <- rownames(data)[c((length(rownames(data))-OOS+1):nrow(data))]
dimnames(prediction_oos)[[2]] <- OOS_params$targetName
dimnames(prediction_oos)[[3]] <- paste0("H",1:H_max)
dimnames(prediction_oos)[[4]] <- model_list

err_oos <-  array(data = NA, dim = c(OOS,length(unique(all_options$var)),H_max,length(model_list)))
rownames(err_oos) <- rownames(data)[c((length(rownames(data))-OOS+1):nrow(data))]
dimnames(err_oos)[[2]] <-  OOS_params$targetName
dimnames(err_oos)[[3]] <- paste0("H",1:H_max)
dimnames(err_oos)[[4]] <- model_list

# Estimation ######################################################################################
# =================================================================================================

for (v in var) {
  for (h in hor) {
    pos <- 1
    for (oos in OOS:1) {
      
      start_estim <- Sys.time()
      
      ## Data management ------------------------------------------------------------
      t <- (nrow(data)-oos+1)
      
      train_data <- data[1:t,]
      train_pos <- 1:(nrow(train_data)-h)
      oos_pos <- nrow(train_data)
      
      ## Models ---------------------------------------------------------------------
      
      ### 1) AR, BIC ------
      
      if("AR, BIC" %in% model_list) {
        
        if(((pos-1) %% reEstimate) == 0) {
          bic <- Inf
          for (P in 1:(lagY+1)) {
            AR <- lm(y~., data = as.data.frame(train_data[train_pos,1:(P+1)]))
            
            if(BIC(AR)< bic) {
              P_min_ar <- P
              bic <- BIC(AR)
            }
          }
        }
        
        AR <- lm(y~., data = as.data.frame(train_data[train_pos,1:(P_min_ar+1)]))
        pred <- predict.lm(AR, newdata = as.data.frame(train_data[,1:(P_min_ar+1)]))[oos_pos]
        
        m <- which(model_list == "AR, BIC")
        err_oos[pos,v,h,m] <- train_data[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      ### 2) ARDI, BIC -----
      
      if("ARDI, BIC" %in% model_list) {
        
        if(((pos-1) %% reEstimate) == 0) {
          bic <- Inf
          for(K in 1:nfac) {
            for(P in 0:(lagY-1)) {
              for (M in 0:(lagX-1)) {
                
                facName <- unlist(lapply(1:K, function(x) paste0("L",0:M,"_F_US",x)))
                posfac <- which(colnames(train_data) %in% facName)
                
                lmdata <- train_data[,c(1:(P+1),posfac)]
                rownames(lmdata) <- c()
                
                ARDI <- lm(y~., data = as.data.frame(lmdata[train_pos,]))
                
                if(BIC(ARDI)< bic) {
                  P_min <- P
                  M_min <- M
                  K_min <- K
                  bic <- BIC(ARDI)
                }
              }
            }
          }
        }
        
        facName <- unlist(lapply(1:K_min, function(x) paste0("L",0:M_min,"_F_US",x)))
        posfac <- which(colnames(train_data) %in% facName)
        lmdata <- data[,c(1:(P_min+1),posfac)]
        rownames(lmdata) <- c()
        
        ARDI <- lm(y~., data = as.data.frame(lmdata[train_pos,]))
        pred <- predict.lm(ARDI,newdata = as.data.frame(lmdata))[oos_pos]
        
        m <- which(model_list == "ARDI, BIC")
        err_oos[pos,v,h,m] <- train_data[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      
      
      ### 3) LASSO ------
      
      if("LASSO" %in% model_list) {
        
        newtrain <- train_data
        
        if(((pos-1) %% reEstimate) == 0) {
          lambda_lasso=cv.glmnet(y = newtrain[train_pos,1], x = newtrain[train_pos,-1], alpha = 1, nfolds = nfolds)$lambda.min
        }
        lasso=glmnet(y=newtrain[train_pos,1], x = newtrain[train_pos,-1],
                     lambda = lambda_lasso, alpha = 1)
        pred=predict(lasso, newx = newtrain[,-1])[oos_pos]
        
        m <- which(model_list == "LASSO")          
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      ### 4) RIDGE -------
      
      if("RIDGE" %in% model_list) {
        
        newtrain <- train_data
        
        if(((pos-1) %% reEstimate) == 0) {
          lambda_ridge=cv.glmnet(y = newtrain[train_pos,1], x = newtrain[train_pos,-1],
                                 alpha = 0, nfolds = nfolds)$lambda.min
        }
        lasso=glmnet(y=newtrain[train_pos,1], x = newtrain[train_pos,-1], lambda = lambda_ridge, alpha = 0)
        pred=predict(lasso, newx = newtrain[,-1])[oos_pos]
        
        m <- which(model_list == "RIDGE")
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      ### 5) Elastic-Net -------
      
      if("ELASTIC-NET" %in% model_list) {
        
        newtrain <- train_data
        
        alpha_range <- EN_hyps$alpha_range
        if(((pos-1) %% reEstimate) == 0) {
          perf_mat <- matrix(NA, 3, length(alpha_range))
          for (i in 1:length(alpha_range)){
            fit_cv <- cv.glmnet(y = newtrain[train_pos,1], x = newtrain[train_pos,-1], alpha=alpha_range[i], nfolds=nfolds)
            min_cv <- min(fit_cv$cvm)
            lambda_min <- fit_cv$lambda[which.min(fit_cv$cvm)]
            perf_mat[1,i] <- alpha_range[i]
            perf_mat[2,i] <- lambda_min
            perf_mat[3,i] <- min_cv
          }
          best_alpha_en <- perf_mat[1,][which.min(perf_mat[3,])]
          lambda_en <- perf_mat[2,][which.min(perf_mat[3,])]
        }
        
        lasso=glmnet(y=newtrain[train_pos,1], x = newtrain[train_pos,-1], lambda = lambda_en, alpha = best_alpha_en)
        pred=predict(lasso, newx = newtrain[,-1])[oos_pos]
        
        m <- which(model_list == "ELASTIC-NET")
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      ### 6) Random Forest -------
      
      if("RF" %in% model_list) {
        
        newtrain <- train_data
        
        RF=ranger(y~., data = as.data.frame(newtrain[train_pos,]), mtry = (ncol(newtrain[,-1])*RF_hyps$mtry),
                  num.trees = RF_hyps$num.trees, min.node.size = RF_hyps$min.node.size)
        pred=predict(RF, data = as.data.frame(newtrain[,]))$predictions[oos_pos]
        
        m <- which(model_list == "RF")
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      ### 7) Boosting -------
      
      if("GBM" %in% model_list) {
        
        newtrain <- train_data
        oosX = newtrain[oos_pos,-1]
        dim(oosX) <- c(1,length(oosX))
        
        if(((pos-1) %% reEstimate) == 0) {
          tuned_gbm <- caret::train(y ~., data = as.matrix(newtrain[train_pos,]),
                                    method = "gbm",
                                    metric = "RMSE",
                                    trControl = Boosting_hyps$fitControl,
                                    tuneGrid = Boosting_hyps$man_grid,
                                    verbose=FALSE
          )
          mod <- tuned_gbm$finalModel
        } else{
          good_hyps <- Boosting_hyps
          good_hyps$fitControl$method <- "none"
          GBM <- caret::train(y ~., data = as.matrix(newtrain[train_pos,]),
                              method = "gbm",
                              metric = "RMSE",
                              trControl = good_hyps$fitControl,
                              tuneGrid = tuned_gbm$bestTune,
                              verbose=FALSE
          )
          mod <- GBM$finalModel
        }
        
        pred <- predict(mod, oosX, n.trees = mod$n.trees)
        
        m <- which(model_list == "GBM")
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      
      
      ### 8) NN -------
      
      if("NN" %in% model_list) {
        
        newtrain <- train_data
        rownames(newtrain) <- c()
        nn_hyps$n_features <- ncol(newtrain[,-1])
        
        if(((pos-1) %% reEstimate) == 0) {
          
          mlp <- MLP(X = newtrain[-oos_pos,-1], Y = newtrain[-oos_pos,1],
                     Xtest = newtrain[oos_pos,-1], Ytest = newtrain[oos_pos,1],
                     nn_hyps = nn_hyps,
                     standardize=T,seed=1234)
          
        }
        
        pred <- mean(predict_nn(mlp, newtrain[oos_pos,-1], nn_hyps))
        
        m <- which(model_list == "NN")
        err_oos[pos,v,h,m] <- newtrain[oos_pos,1] - pred
        prediction_oos[pos,v,h,m] <- pred
      }
      
      
      ### 9) Macro AR-RF -------
      
      if("AR-RF" %in% model_list) {
        
        newtrain <- train_data
        newtrain <- as.matrix(newtrain[c(train_pos,oos_pos),])
        rownames(newtrain) <- c()
        
        if(((pos-1) %% reEstimate) == 0) {
          
          mrf <- MRF(data=newtrain[train_pos,],
                     y.pos=1,
                     S.pos=2:ncol(newtrain),
                     x.pos=MacroRF_hyps$x_pos,
                     oos.pos=c(),
                     minsize=MacroRF_hyps$minsize,
                     mtry.frac=MacroRF_hyps$mtry_frac,
                     prior.var=1/c(0.01,0.25,rep(1,length(MacroRF_hyps$x_pos)-1)),
                     prior.mean=coef(lm(as.matrix(newtrain[train_pos,1])~as.matrix(newtrain[train_pos,MacroRF_hyps$x_pos]))),
                     subsampling.rate=0.80,
                     keep.forest=TRUE,
                     block.size=MacroRF_hyps$block_size,
                     trend.pos=ncol(newtrain),
                     trend.push=4,
                     fast.rw=TRUE,
                     ridge.lambda=0.5,
                     B=MacroRF_hyps$B,
                     printb=FALSE)
        }
        
        rf_test <- as.data.frame(newtrain[,-1])
        pred <- pred.given.mrf(mrf, newdata = rf_test)
        
        m <- which(model_list == "AR-RF")
        err_oos[pos,v,h,m] <- train_data[oos_pos,1] - pred[length(pred)]
        prediction_oos[pos,v,h,m] <- pred[length(pred)]
      }
      
      
      end_estim <- Sys.time()
      print(end_estim-start_estim)
      
      # -----------------------------------------------------------------------------------------
      pos <- pos + 1
      gc()
      
    } #oos
  } #h
} #v

save_path <- paste0(paths$rst,"/",OOS_params$save_path)
if(!dir.exists(save_path)){dir.create(save_path)}
save(prediction_oos, err_oos, data,
     file = paste0(paths$rst,"/",OOS_params$save_path,"/",OOS_params$targetName[var],"_h",hor,".RData"))


# ===========================================================================================================
# 3. RESULTS
# ===========================================================================================================

results <- process_results(paths,OOS_params = OOS_params, benchmark = "AR, BIC") # To use plain MSE put benchmark = NA

# Show MSE ratio
round(results$mse_table,3)
round(results$mse_table_2019,3)

# MSE ratio barplots and predictions plots (the plots are saved in 20_Figures)
mse_barplot_h1 <- list()
mse_barplot_h4 <- list()
pred_plot_h1 <- list()
pred_plot_h4 <- list()

for(var in 1:dim(results$mse_table)[3]) {
  
  # MSE
  mse_barplot_h1[[var]] <- quick_barplot(results, hor = 1, var = var)
  mse_barplot_h4[[var]] <- quick_barplot(results, hor = 4, var = var)
  
  # Predictions
  pred_plot_h1[[var]] <- quick_plot(results, hor = 1, var = var)
  pred_plot_h4[[var]] <- quick_plot(results, hor = 4, var = var)
  
  # Put the 2 graphs together
  p <- arrangeGrob(pred_plot_h1[[var]],mse_barplot_h1[[var]],
                   nrow = 2, ncol = 1)
  ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",1,".png")
  ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
  p <- arrangeGrob(pred_plot_h4[[var]],mse_barplot_h4[[var]],
                   nrow = 2, ncol = 1)
  ptitle = paste0(paths$fig,"/",OOS_params$targetName[var],"_h",4,".png")
  ggsave(ptitle, plot = p, dpi=72, dev='png', height=600, width=450, units="mm")
  
}

# Quick view, you need to choose the postion of the target you want to see
# targets order : (1) "CPIAUCSL", (2) "UNRATE", (3) "HOUST", (4) "PAYEMS", (5) "GDPC1"
mse_barplot_h1[[3]]
pred_plot_h1[[3]]
