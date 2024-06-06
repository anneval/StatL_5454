MLP <- function(X,Y,Xtest,Ytest,nn_hyps,standardize,seed) {
  
  # V5 : Return best_model fixed
  # V6 : Add batch size option
  
  set.seed(seed)
  
  show_train=nn_hyps$show_train
  
  if(show_train < 3) {
    
    cat("\nProgress : \n")
    cat(rep("-",40), sep = "")
    cat("\n \n")
    
  }
  
  # If needed scale data
  temp <- c()
  
  if(standardize==T) {
    
    if(show_train < 3) {
      cat("Standardize Data !!!! \n \n")
    }
    
    temp=scale_data(Xtrain = X, Ytrain = Y, Xtest = Xtest, Ytest = Ytest)
    X=temp$Xtrain
    Xtest=temp$Xtest
    
    Ytest = temp$Ytest
    Y = temp$Ytrain
    
  }
  
  # Convert our input data and labels into tensors.
  x_train = torch_tensor(X, dtype = torch_float())
  y_train = torch_tensor(Y, dtype = torch_float())
  x_test = torch_tensor(Xtest, dtype = torch_float())
  y_test = torch_tensor(Ytest, dtype = torch_float())
  
  # =====================================================================================================
  ## MNN MODEL
  # =====================================================================================================
  
  if(show_train < 3) {
    cat("Initialize Model !!!!! \n \n") 
  }
  
  BuildNN <- function(X,Y,training_index,nn_hyps) {
    
    # Setting up hyperparameters
    lr=nn_hyps$lr
    epochs=nn_hyps$epochs
    patience=nn_hyps$patience
    tol=nn_hyps$tol
    
    #training_index=training_index  
    show_train=nn_hyps$show_train
    
    # Build Model
    net = nn_module(
      "nnet",
      
      initialize = function(n_features=nn_hyps$n_features, nodes=nn_hyps$nodes,dropout_rate=nn_hyps$dropout_rate){
        
        self$n_layers = length(nodes)
        
        self$input = nn_linear(n_features,nodes[1])
        
        if(length(nodes)==1) {
          self$first = nn_linear(nodes,nodes)
        }else{
          self$first = nn_linear(nodes[1],nodes[1])
          self$hidden <- nn_module_list(lapply(1:(length(nodes)-1), function(x) nn_linear(nodes[x], nodes[x+1])))
        }
        
        self$output = nn_linear(nodes[length(nodes)],1)
        self$dropout = nn_dropout(p=dropout_rate)
        
      },
      
      forward = function(x){
        
        # Input
        x = torch_selu(self$input(x))
        
        # Hidden
        x = torch_selu(self$first(x))
        x = self$dropout(x)
        
        if(self$n_layers>1) {
          
          for(layer in 1:(self$n_layers-1)) {
            x = torch_selu(self$hidden[[layer]](x))
            x = self$dropout(x)
          }
          
        }
        
        # Output
        yhat = torch_squeeze(self$output(x))
        
        result <- list(yhat)
        return(result)
      }
      
    )
    
    model = net()
    
    
    ## Train model ---------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    patience = patience
    wait = 0
    
    oob_index <- c(1:x_train$size()[1])[-training_index]
    
    batch_size <- nn_hyps$batch_size
    num_data_points <- length(training_index)
    num_batches <- floor(num_data_points/batch_size)
    
    if(!is.na(nn_hyps$num_batches)){
      num_batches <- nn_hyps$num_batches
    }
    
    best_epoch = 0
    best_loss = NA
    
    criterion = nn_mse_loss()
    optimizer = optim_adam(model$parameters, lr = lr)
    
    
    for (i in 1:epochs) {
      
      # manually loop through the batches
      training_index <- sample(training_index)
      model$train()
      for(batch_idx in 1:num_batches) {
        
        optimizer$zero_grad() # Start by setting the gradients to zero
        
        # here index is a vector of the indices in the batch
        idx <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
        
        # train
        y_pred=model(x_train[training_index[idx],])[[1]]
        loss=criterion(y_pred,y_train[training_index[idx]])
        
        loss$backward()  # Backpropagation step
        optimizer$step() # Update the parameters
        
        # Check Training
        if(show_train==1) {
          if(i %% 1 == 0) {
            #cat(" Batch number: ",batch_idx," on ", num_batches, "\n")
            # cat(" Epoch:", i, "Loss: ", round(loss$item(),5),", Val Loss: ",round(loss_oob$item(),5), "\n")
            
          }
        }
        
      }
      
      model$eval()
      with_no_grad({
        y_pred_oob=model(x_train[oob_index,])[[1]]
        loss_oob=criterion(y_pred_oob,y_train[oob_index])
        
        if(x_train$size()[1] == length(training_index)) {
          loss_oob=criterion(y_pred,y_train[training_index])
        }
      })
      
      percentChange <- ((best_loss - loss_oob$item())/loss_oob$item())
      
      # Early Stopping
      if(best_loss > loss_oob$item() | i == 1) { #best_loss > loss_oob$item()
        best_loss=loss_oob$item()
        best_epoch=i
        state_best_model <- lapply(model$state_dict(), function(x) x$clone()) 
        
        if(percentChange > tol | i == 1) {
          wait=0
        }else {
          wait=wait+1
        }
        
      }else{
        
        wait=wait+1
        
      }
      
      if(show_train==1) {
        
        # Check Training
        if(i %% 1 == 0) {
          cat(" Epoch:", i, ", Loss: ", loss$item(),", Val Loss: ",loss_oob$item(), "(PercentChange: ",round(percentChange,3),")", "\n")
          # cat(" Epoch:", i, "Loss: ", round(loss$item(),5),", Val Loss: ",round(loss_oob$item(),5), "\n")
          
        }
        
      }
      
      if(wait > patience) {
        if(show_train==1) {
          cat("Best Epoch at:", best_epoch, "\n")
        }
        break
      }
      
    }
    
    model$load_state_dict(state_best_model)
    return(model) # Return the model with the best val loss
    
  }
  
  # =====================================================================================================
  ## MODEL AVERAGING
  # =====================================================================================================
  
  num_average <- nn_hyps$num_average
  sampling_rate <- nn_hyps$sampling_rate
  
  pred.in.ensemble <- array(data = NA, dim = c(nrow(X),num_average))
  pred.ensemble <- array(data = NA, dim = c(nrow(Xtest),num_average))
  
  if(show_train==2) {
    pb <- txtProgressBar(min = 0, max = num_average, style = 3) # Progress bar
  }
  
  trained_model <- list()
  for(j in 1:num_average) {
    
    # Bootstrap parameters
    set.seed(seed+j)
    boot <- sample(1:nrow(X), size = sampling_rate*nrow(X), replace = F) # training
    oob <- (nrow(X)+1):(nrow(X)+nrow(Xtest))                             # out of bag
    
    ## Estimation -------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    model <- BuildNN(x_train,y_train,boot,nn_hyps)
    trained_model[[j]] <- model
    
    if(show_train==1) {
      cat("Done with model :",j, "\n \n")
    }
    
    ## Storage ----------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    model$eval()
    
    if(standardize==T) {
      
      pred.in.ensemble[,j] <- invert_scaling(as.matrix(model(x_train[,])[[1]]),temp)
      pred.ensemble[,j] <- invert_scaling(as.matrix(model(x_test)[[1]]),temp)
      
    }else{
      
      pred.in.ensemble[,j] <- as.matrix(model(x_train[,])[[1]])
      pred.ensemble[,j] <- as.matrix(model(x_test)[[1]])
      
    }
    
    model$train()
    
    if(show_train==2) {
      setTxtProgressBar(pb, j)
    }
    
  } # j, bootstrap end
  
  if(show_train==2) {
    close(pb)
  }
  
  # Output
  pred.in <- rowMeans(pred.in.ensemble, na.rm = T)
  pred <- rowMeans(pred.ensemble, na.rm = T)
  
  results <- list(pred.in.ensemble=pred.in.ensemble,
                  pred.ensemble=pred.ensemble,
                  pred.in=pred.in,
                  pred=pred,
                  trained_model=trained_model,
                  scaler = temp,
                  standardize=standardize)
  
  return(results)
  
} # MLP FUNCTION END

# =====================================================================================================
## PREDICT (NOT WORKING !)
# =====================================================================================================

predict_nn <- function(mlp,Xtest,nn_hyps) {
  
  
  EmptyNN <- function(nn_hyps) {
    # Build Model
    net = nn_module(
      "nnet",
      
      initialize = function(n_features=nn_hyps$n_features, nodes=nn_hyps$nodes,dropout_rate=nn_hyps$dropout_rate){
        
        self$n_layers = length(nodes)
        
        self$input = nn_linear(n_features,nodes[1])
        
        if(length(nodes)==1) {
          self$first = nn_linear(nodes,nodes)
        }else{
          self$first = nn_linear(nodes[1],nodes[1])
          self$hidden <- nn_module_list(lapply(1:(length(nodes)-1), function(x) nn_linear(nodes[x], nodes[x+1])))
        }
        
        self$output = nn_linear(nodes[length(nodes)],1)
        self$dropout = nn_dropout(p=dropout_rate)
        
      },
      
      forward = function(x){
        
        # Input
        x = torch_selu(self$input(x))
        
        # Hidden
        x = torch_selu(self$first(x))
        x = self$dropout(x)
        
        if(self$n_layers>1) {
          
          for(layer in 1:(self$n_layers-1)) {
            x = torch_selu(self$hidden[[layer]](x))
            x = self$dropout(x)
          }
          
        }
        
        # Output
        yhat = torch_squeeze(self$output(x))
        
        result <- list(yhat)
        return(result)
      }
    )
    
    model = net()
    return(model)
  }
  
  
  # Create empty matrix
  num_bootstrap <- dim(mlp$pred.in.ensemble)[2]
  
  if(!is.null(nrow(Xtest))) {
    obs <- nrow(Xtest)
  } else{
    obs <- 1
  }
  
  forecasts <- matrix(data = NA, nrow = obs, ncol = num_bootstrap)
  forecasts[,] <- NA
  
  if(mlp$standardize == T) {
    scaler <- mlp$scaler
    newx <- predict_scale_data(scaler, Xtest)
  }
  
  
  for(i in 1:num_bootstrap) {
    
    # Format data
    Xtest <- torch_tensor(newx, dtype = torch_float(), requires_grad = F)
    
    # Load trained models
    state_dict <- mlp$trained_model[[i]]
    state_model <- lapply(state_dict$state_dict(), function(x) x$clone()) 
    model <- EmptyNN(nn_hyps)
    model$load_state_dict(state_model)
    
    # Predict
    model$eval()
    if(mlp$standardize == T) {
      
      forecasts[,i] <- invert_scaling(as.matrix(model(Xtest)[[1]]),scaler)
      
    }else{
      
      forecasts[,i] <- as.matrix(model(Xtest)[[1]])
      
    }
    
  }
  
  return(forecasts)
  
}

# =====================================================================================================
## STANDARDIZATION
# =====================================================================================================

scale_data <- function(Xtrain, Ytrain, Xtest, Ytest) {
  
  # Features
  sigma_x <- apply(Xtrain,2,sd)
  mu_x <- apply(Xtrain,2,mean)
  
  if(is.null(dim(Xtest))==TRUE) {
    dim(Xtest) <- c(1, length(Xtest))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtest[,x] - mu_x[x])/sigma_x[x]))
  Xtrain <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtrain[,x] - mu_x[x])/sigma_x[x]))
  
  # Target
  sigma_y <- sd(Ytrain)
  mu_y <- mean(Ytrain)
  
  Ytrain <- (Ytrain-mu_y)/sigma_y
  Ytest <- (Ytest-mu_y)/sigma_y
  
  return(list(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest , sigma_y = sigma_y, mu_y = mu_y,
              sigma_x = sigma_x, mu_x = mu_x))
  
}

invert_scaling <- function(scaled, scaler) {
  
  sigma_y <- scaler$sigma_y
  mu_y <- scaler$mu_y
  
  
  inverted <- sigma_y*scaled + mu_y
  
  return(inverted)
  
}

rescale_data <- function(results,newx) {
  
  # Features
  sigma_x <- results$scaler$sigma_x
  mu_x <- results$scaler$mu_x
  
  if(is.null(dim(newx))==TRUE) {
    dim(newx) <- c(1, length(newx))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (newx[,x] - mu_x[x])/sigma_x[x]))
  
  return(newx = Xtest)
  
}

predict_scale_data <- function(scaler, Xtest) {
  
  # Features
  sigma_x <- scaler$sigma_x
  mu_x <- scaler$mu_x
  
  if(is.null(dim(Xtest))==TRUE) {
    dim(Xtest) <- c(1, length(Xtest))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtest[,x] - mu_x[x])/sigma_x[x]))
  
  return(Xtest)
}