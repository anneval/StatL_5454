
# ===========================================================================================================
# 1. WRAPPER FUNCTION THAT DOES THE OUT-OF-SAMPLE EXERCISE
# ===========================================================================================================

Forecast_all <- function(it_pos, all_options, paths, OOS_params, seed) {
  
  #
  # Description:
  # Function that does the OOS exercise as describe in the OOS_params parameters
  #
  # INPUTS:
  # it_pos : 
  # all_options : 
  # paths : the names of the subfolders
  # OOS_params : The parameters used in the OOS exercise
  # seed : 
  #
  # OUTPUTS:
  # prediction_oos: matrix of all predictions made in the OOS exercise (dimensions : OOS obs X targets X horizons X models)
  # err_oos :  matrix of all the errors made during OOS exercise (Y - Yhat, (dimensions : OOS obs X targets X horizons X models))
  # data : matrix containing the training set (all lagged values and the target)
  #
  
  set.seed(seed)
  
  # Load libraries
  library(torch)      # Neural Networks
  library(glmnet)     # Lasso, Ridge and Elastic-Net
  library(ranger)     # Random Forest
  library(MacroRF)    # Macro Random Forest
  library(caret)      # Gradient boosting machine (GBM)
  library(pracma)     # Utilities (for PCA)
  library(stringr)    # String manipulation
  library(doParallel) # Parallel estimation
  library(foreach)    # Parallel estimation
  
  # Load US data retreiver -----------------------------
  source(paste(paths$too, 'MakeDataUK_function.R', sep='/'))
  
  # Factor analysis (PCA) ------------------------------
  source(paste(paths$too, 'EM_sw.R', sep='/'))                                                       
  source(paste(paths$too, 'factor.R', sep='/'))                                                      
  source(paste(paths$too, 'ICp2.R', sep='/'))
  
  # Neural Network function ----------------------------
  source(paste(paths$too, 'MLP_function_v6b.R', sep='/'))
  torch_set_num_threads(1)
  
  # Variable and Horizon to forecast #############################################################
  # ==============================================================================================
  it_pos = 1
  var <- all_options$var[it_pos]
  hor <- all_options$hor[it_pos]
  
  # Get the data
  UKdata <- MakeDataUK(path = paste0(path,paths$dat,"/"), targetName = OOS_params$targetName[var], h = hor,
                       nFac = OOS_params$nFac, lag_y = OOS_params$lagY, lag_f = OOS_params$lagX, lag_marx = OOS_params$lagMARX,
                       versionName = "current",
                       download = F, EM_algorithm=F, EM_last_date=NA,
                       frequency = 1, target_new_tcode=OOS_params$target_tcode[var])
  data <- UKdata[[1]]$lagged_data
  
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
          nn_hyps$n_features <<- ncol(newtrain[,-1])
          
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
          
          # newtrain_df <- as.data.frame(newtrain)
          # S.pos_df <- newtrain_df %>%
          #   select(2:258, contains("_F_UK"))
          # 
      
          col_names <- colnames(newtrain)
          
          # Find indices for columns 2 to 258
          cols_2_258 <- 2:258
          
          # Find indices for columns containing '_F_UK'
          cols_F_UK <- grep("_F_UK", col_names)
          
          # Combine the indices
          selected_cols <- c(cols_2_258, cols_F_UK)
          
          # Select the columns from the matrix
          
          Spos = newtrain[, selected_cols]
          
          
          if(((pos-1) %% reEstimate) == 0) {
            
            mrf <- MRF(data=newtrain[train_pos,],
                       y.pos=1,
                       S.pos = Spos,
                       #+2:ncol(newtrain),
                       x.pos=MacroRF_hyps$x_pos,
                       oos.pos=c(),
                       minsize=MacroRF_hyps$minsize,
                       mtry.frac=MacroRF_hyps$mtry_frac,
                       min.leaf.frac.of.x=1.5,
                       VI=FALSE,
                       ERT=FALSE,
                       quantile.rate=0.33,
                       S.priority.vec=NULL,
                       random.x = FALSE,
                       howmany.random.x=1,
                       howmany.keep.best.VI=20,
                       cheap.look.at.GTVPs=FALSE,
                       prior.var=1/c(0.01,0.25,rep(1,length(MacroRF_hyps$x_pos)-1)),
                       prior.mean=coef(lm(as.matrix(newtrain[train_pos,1])~as.matrix(newtrain[train_pos,MacroRF_hyps$x_pos]))),
                       subsampling.rate=0.70,
                       rw.regul=0.95,
                       keep.forest=TRUE,
                       block.size=MacroRF_hyps$block_size,
                       trend.pos=ncol(newtrain),
                       trend.push=4,
                       fast.rw=TRUE,
                       ridge.lambda=0.5,HRW=0,B=MacroRF_hyps$B,resampling.opt=2,printb=FALSE)
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
  
}

# ===========================================================================================================
# 2. EXPLORE THE RESULTS
# ===========================================================================================================

process_results <- function(paths, OOS_params, benchmark = NA) {
  
  #
  # Description:
  # Function that store the results and create mse tables
  #
  # INPUTS:
  # paths : the names of the subfolders
  # OOS_params : The parameters used in the OOS exercise
  # benchmark : Name of the benchmark (ex; "AR, BIC")
  #
  # OUTPUTS:
  # predictions : matrix of all predictions made in the OOS exercise (dimensions : OOS obs X targets X horizons X models)
  # errors :  matrix of all the errors made during OOS exercise (Y - Yhat, (dimensions : OOS obs X targets X horizons X models))
  # targets : matrix of all the targets (dimensions : obs X targets X horizons)
  #
  
  ## Create storage --------------------------------------------------------------------------
  
  results_path <- paste(paths$rst,OOS_params$save_path, sep = "/")
  results_files <- list.files(results_path)
  load(paste(results_path,results_files[1], sep = "/"))
  model_list <- OOS_params$model_list
  
  predictions <- array(data = NA, dim = c(dim(prediction_oos)[1],
                                          dim(prediction_oos)[2],
                                          dim(prediction_oos)[3],
                                          length(model_list)))
  rownames(predictions) <- rownames(prediction_oos)
  dimnames(predictions)[[2]] <- colnames(prediction_oos)
  dimnames(predictions)[[3]] <- dimnames(prediction_oos)[[3]]
  dimnames(predictions)[[4]] <- model_list
  
  errors <- array(data = NA, dim = c(dim(prediction_oos)[1],
                                     dim(prediction_oos)[2],
                                     dim(prediction_oos)[3],
                                     length(model_list)))
  rownames(errors) <- rownames(prediction_oos)
  dimnames(errors)[[2]] <- colnames(prediction_oos)
  dimnames(errors)[[3]] <- dimnames(prediction_oos)[[3]]
  dimnames(errors)[[4]] <- model_list
  
  targets <- array(data = NA, dim = c(nrow(data),
                                      dim(prediction_oos)[2],
                                      dim(prediction_oos)[3]))
  rownames(targets) <- rownames(data)
  dimnames(targets)[[2]] <- colnames(prediction_oos)
  dimnames(targets)[[3]] <- dimnames(prediction_oos)[[3]]
  
  ## Store results ------------------------------------------------------------------------------
  
  H <- OOS_params$horizon
  for (var in 1:length(OOS_params$targetName)) {
    posH <- 1
    for(hor in H) {
      
      # Load results
      file <- paste0(results_path,"/",OOS_params$targetName[var],"_h",hor,".RData")
      
      tryCatch(
        {
         load(file)
          predictions[,var,hor,] <- prediction_oos[,which(dimnames(prediction_oos)[[2]] %in% OOS_params$targetName[var]),hor,]
          errors[,var,hor,] <- err_oos[,which(dimnames(err_oos)[[2]] %in% OOS_params$targetName[var]),hor,] 
          targets[hor:nrow(targets),var,hor] <- data[,1]
        },
        error = function(e) {
          print(file)
          predictions[,var,hor,] <- NA
          errors[,var,hor,] <- NA
          targets[,var,hor] <- NA
        }) # Some results may not be there
      
      posH <- posH + 1
    }
  }
  
  ## Calculate MSEs ---------------------------------------------------------------------------
  
  # All Sample
  mse_table <- array(data = NA, c(dim(predictions)[3],
                                  dim(predictions)[4],
                                  dim(predictions)[2]))
  dimnames(mse_table)[[1]] <- dimnames(predictions)[[3]]
  dimnames(mse_table)[[2]] <- dimnames(predictions)[[4]]
  dimnames(mse_table)[[3]] <- dimnames(predictions)[[2]]
  
  # Until 2019
  end <- which(dimnames(predictions)[[1]]=="2019-12-01")
  mse_table_2019 <- array(data = NA, c(dim(predictions)[3],
                                       dim(predictions)[4],
                                       dim(predictions)[2]))
  dimnames(mse_table_2019)[[1]] <- dimnames(predictions)[[3]]
  dimnames(mse_table_2019)[[2]] <- dimnames(predictions)[[4]]
  dimnames(mse_table_2019)[[3]] <- dimnames(predictions)[[2]]
  
  for (var in 1:dim(predictions)[2]) { #1:dim(predictions)[2]
   for(hor in 1:dim(predictions)[3]) {
    #for(hor in H) {
  
      bench <- which(dimnames(predictions)[[4]] == benchmark)
      
      mse <- apply(errors[,var,hor,]^2,2,mean, na.rm=TRUE)
      if(!is.na(benchmark)) {
        mse <- mse <- mse/mse[bench]
      }
      mse_table[hor,,var] <- mse
      
      mse <- apply(errors[1:end,var,hor,]^2,2,mean, na.rm=TRUE)
      if(!is.na(benchmark)) {
        mse <- mse <- mse/mse[bench]
      }
      mse_table_2019[hor,,var] <- mse
      
    }
  }
  
  ## Output results -------------------------------------------------------------------------------
  
  output <- list("predictions" = predictions,
                 "errors" = errors,
                 "targets" = targets,
                 "mse_table" = mse_table,
                 "mse_table_2019" = mse_table_2019)
  
  return(output)
  
}


quick_barplot <- function(results, hor, var) {
  
  #
  # Description:
  # Function that generates barplots for out-of-sample MSEs
  #
  
  # Get MSE from both periods
  mse <- results$mse_table[hor,,var]
  mse_2019 <- results$mse_table_2019[hor,,var]
  mse <- rbind(mse,mse_2019)
  rownames(mse) <- c("All Sample", "Until 2019")
  
  # Graph
  mse_long <- reshape2::melt(mse)
  title <- paste0(dimnames(results$mse_table)[[3]][var]," - ",dimnames(results$mse_table)[[1]][hor])
  
  p <- ggplot(data=mse_long, aes(x=Var2, y=value,fill=Var1)) +
    geom_bar(stat="identity",position=position_dodge()) +
    ggtitle(title)+
    theme_bw()+
    theme(legend.position="bottom",
          legend.text=element_text(size=25),
          legend.key.size = unit(3,"line"),
          strip.text = element_text(face="bold", colour = "white",size=23,family="Arial"),
          legend.title=element_blank(),axis.text.x = element_text(face = c('plain', 'plain', 'plain', 'plain', 'plain')),
          strip.background=element_rect(colour="black",fill="black"),
          axis.text=element_text(size=23),
          plot.title = element_text(size = 25, face = "bold", hjust = 0.5)) +
    scale_fill_manual(values=c("#386cb0","#ef3b2c")) +
    xlab("")+
    ylab("")+ 
    geom_hline(yintercept = 1)
  
  return(p)
}


quick_plot <- function(results, hor, var) {
  
  #
  # Description:
  # Function that generates graphs for the out-of-sample predictions
  #
  # INPUTS :
  # results : it's the output from the "process_results" function
  # var : position of the variable of interest in "OOS_params$targetName"
  # hor : horizon
  #
  
  # Get predictions from each models
  target = results$targets[((nrow(results$targets)-nrow(results$predictions))+1):nrow(results$targets),var,hor]
  pred <- cbind(results$predictions[,var,hor,], "Target" = target)
  pred_long <- reshape2::melt(pred)
 # pred_long$Var1 =  as.Date(as.character(pred_long$Var1), format = c("%m/%d/%Y"))
  # pred_long$Var1 =  as.Date(as.character(pred_long$Var1), format = c("%m/%d/%Y"))
  pred_long$Var1 <- as.character(pred_long$Var1)
  pred_long$Var1 <- as.Date(pred_long$Var1, format = "%Y-%m-%d")
  
  # Graph parameters
  target_pos <- which(unique(pred_long$Var2) == "Target")
  sizeline <- rep(1.2,length(unique(pred_long$Var2)))
  sizeline[target_pos] <- 2
  colvector <- rainbow(length(unique(pred_long$Var2)))
  colvector[target_pos] <- "black"
  title <- paste0(dimnames(results$predictions)[[2]][var]," - ",dimnames(results$predictions)[[3]][hor])
  
  # Geaph
  p <- ggplot(data = pred_long, aes(x = Var1, y = value, color = Var2)) + 
    geom_line(aes(size = Var2))+
    scale_size_manual(values = sizeline) +
    scale_color_manual(values=c(colvector))+
    xlab('')+ylab('')+
    ggtitle(title)+
    theme_bw() +
    theme(legend.position="bottom",
          legend.text=element_text(size=25),
          legend.key.size = unit(3,"line"),
          strip.text = element_text(face="bold", colour = "white",size=23,family="Arial"),
          legend.title=element_blank(),axis.text.x = element_text(face = c('plain', 'plain', 'plain', 'plain', 'plain')),
          strip.background=element_rect(colour="black",fill="black"),
          axis.text=element_text(size=25),
          plot.title = element_text(size = 25, face = "bold", hjust = 0.5))
  return(p)
}
