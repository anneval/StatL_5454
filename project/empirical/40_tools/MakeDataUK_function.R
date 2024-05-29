MakeDataUK <- function(path, targetName, h, nFac, lag_y,lag_f, lag_marx, versionName="current", frequency, download=TRUE, EM_algorithm=T,
                       EM_last_date=NA,target_new_tcode) {
  
  library(pracma)
  library(stringr)
  
  
 path = paste0(path,paths$dat,"/")
 targetName = OOS_params$targetName[var]
 h = hor
    nFac = OOS_params$nFac
    lag_y = OOS_params$lagY
    lag_f = OOS_params$lagX
    lag_marx = OOS_params$lagMARX
    
   versionName = "current"

   download = F
   EM_algorithm=F
   EM_last_date=NA

   frequency = 1

   target_new_tcode=OOS_params$target_tcode[var]
  
  
  # Download FRED-QD if needed
  if(substr(path,nchar(path),nchar(path)) != "/") {path <- paste0(path,"/")}

  # if(download == TRUE) {
  #   if(frequency==1) {
  #     url <- paste0("https://files.stlouisfed.org/files/htdocs/fred-md/monthly/",versionName,".csv")
  #     download.file(url,
  #                   destfile = paste(path,paste0(versionName,"_monthly.csv"),sep='/'),
  #                   mode = "wb")
  #   }else if(frequency==2) {
  #     url <- paste0("https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/",versionName,".csv")
  #     download.file(url,
  #                   destfile = paste(path,paste0(versionName,"_quarterly.csv"),sep='/'),
  #                   mode = "wb")
  #   }
  # 
  # else {
  #   local_path <- "C:/Users/avalder/OneDrive - WU Wien/Documents/Study/SoSe_24/Statistical Learning/assignments/StatL_5454/project/empirical/10_data/UKMD_April_2024"
  #   if (frequency == 1) {
  #     data <- read.csv(file.path(local_path, paste0("raw_uk_md.csv")))
  #   }
  #   return(data)
  # }}


  output <- vector("list", length = length(targetName))
 numOfTarget <-1
  
  for (numOfTarget in 1:length(output)) {
    
    # ## TRANSFORM DATA ---------------------------------------------------------------
    # if(frequency==1){
    #   data <- transformFRED(file = paste0(path,"raw_uk_md.csv"),date_start =as.Date("1998-01-01"), date_end = NULL,
    #                         transform = TRUE, frequency, targetName[numOfTarget], target_new_tcode[numOfTarget], h)
    # } else if(frequency==2){
    #   data <- transformFRED(file = paste0(path,versionName,"_quarterly.csv"), date_start = NULL, date_end = NULL,
    #                         transform = TRUE, frequency, targetName[numOfTarget], target_new_tcode[numOfTarget], h)
    # }
    
    local_path <- "C:/Users/avalder/OneDrive - WU Wien/Documents/Study/SoSe_24/Statistical Learning/assignments/StatL_5454/project/empirical/10_data/UKMD_April_2024"
    if (frequency == 1) {
      data <- read.csv(file.path(local_path, paste0("balanced_uk_md.csv")))
    }
    # 
  
    
     targetedSerie <- data[[targetName]]
       # rawdata <- data[[2]]
    # all_tcodes <- data[[4]]
    # data <- data[[1]]
    # transdata <- data
    # 
    # Change variables names
    # colnames(data)[grep(" ", colnames(data))] <- sub(" ",".",grep(" ", colnames(data), value = TRUE))
    # colnames(data)[grep(" ", colnames(data))] <- sub(" ",".",grep(" ", colnames(data), value = TRUE))
    # colnames(data)[grep("&", colnames(data))] <- sub("&",".",grep("&", colnames(data), value = TRUE))
    # colnames(data)[grep(":", colnames(data))] <- sub(":","",grep(":", colnames(data), value = TRUE))
    # 
    # 
    data <- data[,-1]
    
    date <- as.character(data[,1])
    if(length(targetedSerie)>1) {
      names(targetedSerie) <- date
    }
    
    # Use EM Algorithm
    # if(EM_algorithm == TRUE) {
    #   
    #   if(!is.na(EM_last_date)) {
    #     toUse <- 1:which(date == EM_last_date)
    #     part <- EM_sw(data=data[toUse,], n=8, it_max=1000)$data
    #     data <- rbind(part,as.matrix(data[-toUse,]))
    #   }else{
    #     data <- EM_sw(data=data[,], n=8, it_max=1000)$data
    #   }
    # }
    
    # Get the target
    rownames(data) <- date
    target <- data[,which(colnames(data)==targetName[numOfTarget])]
    if(length(targetedSerie)<2) {
      targetedSerie <- target
    }
    data <- data[,-which(colnames(data)==targetName[numOfTarget])]
    varNames <- colnames(data)
    
    data <- data[1:313,-1] # drop 2024 because of NAs 
    target <- target[1:313]
    targetedSerie <- targetedSerie[1:313]
    #### LIBOR imputation
    
    na_id <- which(is.na(data$LIBOR_3mth))
    good_id <- sapply(na_id, function(x) x + -3:3)
    imputation <- apply(good_id, 2, function(x) mean(data[x,"LIBOR_3mth"], na.rm = TRUE))
    
    data[na_id,"LIBOR_3mth"] <- imputation
    # 
    # data_m <- as.numeric(as.matrix(data))
    # which(is.na(data_m))
    ###### 

    # 
    # X <- standard(as.matrix(data))
    #  r <- nFac
    #  bign <- dim(X)[2]
    #  bigt <- dim(X)[1]
    #  #X <- as.matrix(data)
    # #  t(data_m)%*%data_m
    #  
    # # data_m <- as.matrix()
    #  test_xx <- t(X)%*%X
     
   #  svd <- svd(test_xx)
    # lambda <- svd$u[,1:r]*sqrt(bign) # r th column times r th biggest eigenvalue
    # f_hat <- X%*%lambda/bign         # factors
    # e_hat <- X - f_hat%*%t(lambda)   # errors
    # mse <- sum(e_hat^2)/(bign*bigt)
    # 
    # results <- list(factors = f_hat, lambda = lambda, mse = mse)
    # 
    ## MAKE FACTORS (if needed) -----------------------------------------------------
     # 
     # data_s <- standard(data)
     # 
     # 
     data_m <- as.matrix(data)
     # 
    if(nFac > 0) {
      facs <- factorize(standard((data_m))$Y, n_fac = nFac)$factor
      colnames(facs) <- paste0("F_US",1:nFac)
      data = cbind(facs,data)
    }else{
      facs = NA
    }
    

   # factorize(as.matrix(test),n_fac=5) 
    
    ## MAKE LAGS --------------------------------------------------------------------
    maxLag <- max(lag_y, lag_f)
    maxLag_marx = max(lag_marx) 
    maxLag_all = max(maxLag_marx,maxLag, na.rm = T)
    
    
    newtrain <- make_reg_matrix(y=targetedSerie,Y=target,factors = data, h=h, max_y = lag_y+h-1, max_f = lag_f+h-1)
    newtrain <- newtrain[(maxLag_all+h+1):nrow(newtrain),]
    
    ## MAKE MARX --------------------------------------------------------------------
    if(!is.na(lag_marx)) {
      if(nFac > 0) {
        X_alt <- data[,-c(1:nFac)]
      }else{
        X_alt <- data
      }
      
      lags <- make_reg_matrix(y=targetedSerie,Y=target,factors = X_alt , h=h, max_y = maxLag_marx+h-1, max_f = maxLag_marx+h-1)
      lags <- lags[(maxLag_all+h+1):nrow(lags),-c(1:(maxLag_marx+1))]
      rownames(lags) <- c()
      
      names.cs = colnames(X_alt)
      bigX <- lags
      
      names.bigX = 1:dim(bigX)[2]
      
      lags_names <- paste0("L",0:(maxLag_marx-1),"_")
      lags_names <- paste(lags_names, collapse = "|")
      for(jj in 1:dim(bigX)[2]){
        names.bigX[jj] = str_replace(colnames(bigX)[[jj]],lags_names,"")#substr(colnames(bigX)[jj],start=4,stop=nchar(colnames(bigX)[jj]))
      }
      new.facs = c()
      new.marx <- c()
      
      for(jj in 1:length(names.cs)){
        subset=bigX[,names.bigX==names.cs[jj]]
        
        marx <- do.call(cbind,
                        lapply(1:(maxLag_marx-1), function(x) (rowMeans(subset[,1:(x+1)]))) )
        colnames(marx) <- paste0(paste0("L",1:(maxLag_marx-1),"_MARX"),"_",names.cs[jj])
        new.marx <- cbind(new.marx,marx)
      }
      
      # MAF and MARX for Y
      subset=newtrain[,2:(maxLag_marx+1)]
      
      marx <- do.call(cbind,
                      lapply(1:(maxLag_marx-1), function(x) (rowMeans(subset[,1:(x+1)]))) )
      colnames(marx) <- paste0(paste0("L",1:(maxLag_marx-1),"_MARX"),"_","YLag")
      new.marx <- cbind(new.marx,marx)
      
      # Keep only marx selected lags
      # toKeep <- unlist(lapply(lag_marx, function(x) paste0("L",x-1,"_|")))
      lag_marx_alt <- lag_marx[-length(lag_marx)]
      toKeep <- paste0("L",lag_marx_alt-1,"_|")
      toKeep <- paste(toKeep, collapse = '')
      toKeep <- paste0(toKeep,"L",lag_marx[length(lag_marx)]-1,"_")
      new.marx <- new.marx[,grep(toKeep, colnames(new.marx))]
      
      newtrain <- cbind(newtrain,new.marx,trend=1:nrow(newtrain))
      
    }else{
      
      newtrain <- cbind(newtrain,trend=1:nrow(newtrain))
      
    }
    
    ### COMBINE ALL ------------------------------------------------------------------
    
    if(frequency==1) {
      frequencyString = "Monthly"
    }else if(frequency==2){
      frequencyString = "Quarterly"
    }
    
    returns <- list(#trans_data=transdata,
                    #raw_data=rawdata,
                    lagged_data=newtrain,
                    factors=facs,
                    targetName=targetName[numOfTarget],
                  #  versionName=versionName,
                    horizon=h,
                    frequency=frequencyString,
                    varNames=c(varNames))
                   # tcodes=all_tcodes)
    
    output[[numOfTarget]] <- returns
    
  } #numOfTarget
  
  names(output) <- targetName
  return(output)
  
}

## ========================================================================== ##
## SUBFUNCTIONS                                                               ##
## ========================================================================== ##

make_reg_matrix <- function(y,Y,factors,h,max_y,max_f){
  # Author: Stephane Surprenant
  # Creation: 12/02/2018
  #
  # Description: This function creates a regression matrix
  # containing the dependent variable, y, its lagged values
  # from h to max_y and lagged exogenous regressors,
  # from lag h to max_f
  #
  # NOTE: y and factors must of same time dimension.
  #
  # OUTPUT
  # First column is dependent variable. All others are
  # regressors.
  
  bigtY <- nrow(as.matrix(Y))       # Time dimension
  bigtF <- nrow(as.matrix(factors))       # Time dimension
  bign <- ncol(as.matrix(factors)) # Number of factors
  
  lags <- sapply(h:max_y, function(i) c(array(NA,dim=i),Y[1:(bigtY-i)]))
  f <- do.call(cbind, lapply(h:max_f, function(i)
    rbind(array(NA,dim=c(i,bign)),
          as.matrix(factors[1:(bigtF-i),])))
  )
  reg <- cbind(y,lags,f)
  
  # Add names
  colnames(reg)[2:(max_y-h+2)] <- paste(paste("L", h:max_y-h, sep="_"),
                                        "y", sep="")
  count <- max_y-h+3
  name_fac=gsub(':','',gsub('&','.',gsub(' ','_',colnames(factors))))
  for (i in h:max_f){
    colnames(reg)[count:(count+bign-1)] <- paste(paste("L", i-h, sep=""), name_fac, sep="_")
    count <- count + bign
  }
  
  return(reg)
}

factorize <- function(data,n_fac){
  
  X <- data  # Reassign
  r <- n_fac # Reassign
  
  bign <- dim(X)[2]
  bigt <- dim(X)[1]
  
  svd <- svd(t(X)%*%X)         # svd$u = svd$v (symmetry); svd$d = eigenvalues
  lambda <- svd$u[,1:r]*sqrt(bign) # r th column times r th biggest eigenvalue
  f_hat <- X%*%lambda/bign         # factors
  e_hat <- X - f_hat%*%t(lambda)   # errors
  mse <- sum(e_hat^2)/(bign*bigt)
  
  results <- list(factors = f_hat, lambda = lambda, mse = mse)
  
  return(results)
}

EM_sw <- function(data, n, it_max=50){
  #
  # Author: Stephane Surprenant
  # Creation: 16/11/2017
  #
  # Description:
  # This function inputs missing values in 'data' by using an
  # Expectation-Maximization algorithm. The data must is standerdized and
  # centered at zero. The algorithm begins by replacing the missing values
  # by 0 (the unconditional mean of the available observations). Then, it
  # estimates a factor model through principal component. Missing values
  # are then replaced by the predicted common component of the series.
  #
  # INPUTS: data: a matrix of data with time series as column vectors;
  #         n: number of factors considered
  # OUTPUT: X: a balanced panel
  #
  # Note: Requires package pracma (for repmat) and factor.R for PCA.
  #
  # ========================================================================= #
  #                               ESTIMATION
  # ========================================================================= #
  
  X <- as.matrix(data)
  
  # 1. Standerdize data
  size <- dim(X)
  a <- standard(X)
  X0 <- a$Y
  std <- a$std
  mu <- a$mean
  
  # 2. First replacement and PCA
  X0[is.na(X)] <- 0 # Unconditional mean of standard data is zero
  X0[is.na(X0)] <- 0 # Unconditional mean of standard data is zero
  
  a <- factorize(X0, n_fac=n) # PCA
  f_1 <- a$factors
  L_1 <- a$lambda
  x_hat <- X0
  cc <- f_1%*%t(L_1) # Common component 
  x_hat[is.na(X)] <- cc[is.na(X)]
  x_hat <- x_hat*repmat(std,size[1],1) + repmat(mu,size[1],1)
  
  # 3. Initialize algorithm
  it <- 0      # Initialize iterations
  err <- 999   # Initialize error criterion to arbitrarily high value
  # it_max <- 50 # Selection maximum iterations
  con <- 1e-6  # Set convergence criterion
  old_f <- array(0, dim=c(size[1],1))
  
  while(it < it_max && err > con){
    # 1. Standerdize dataset
    a <- standard(x_hat) 
    std_new <- a$std
    mu_new <- a$mean
    x_new <- a$Y
    
    old_f <- f_1 # Save factors from previous iteration
    
    # 2. Extract factor through PCA
    a <- factorize(x_new, n_fac=n)
    f_1 <- a$factors
    L_1 <- a$lambda
    mse <- a$mse
    cc <- f_1%*%t(L_1) # Common component 
    x_hat <- x_new
    x_hat[is.na(X)] <- cc[is.na(X)]
    x_hat <- x_hat*repmat(std_new,size[1],1) + repmat(mu_new,size[1],1)
    
    mean_old_f2 <- apply(old_f^2, c(2), mean)
    mean_new_f2 <- apply(f_1^2, c(2), mean)
    
    err <- abs(mean(mean_old_f2 - mean_new_f2))
    
    it <- it + 1;
  }
  x_hat[!is.na(X)] <- X[!is.na(X)]
  results <- list(data=x_hat, factors=f_1, lambda=L_1, iterations=it, mse=mse)
  return(results)
}

standard <- function(Y){
  Y < as.matrix(Y)
  size <- dim(Y)
  
  mean_y <-apply(Y, c(2), mean, na.rm=TRUE)
  sd_y <- apply(Y, c(2), sd, na.rm=TRUE)
  # print(Y)
  # print(mean_y)
  #  print(size[1])
  # print(repmat(mean_y, size[1],1))
  
  Y0 <- (Y - repmat(mean_y, size[1],1))/repmat(sd_y, size[1],1)
  return(list(Y=Y0, mean=mean_y, std=sd_y))
}




# data <- transformFRED(file = paste0(path,"raw_uk_md.csv"),date_start =as.Date("1998-01-01"), date_end = NULL,
#                       transform = TRUE, frequency, targetName[numOfTarget], target_new_tcode[numOfTarget], h)

#targetNames <- targetName[numOfTarget]
#target_tcode <- target_new_tcode[numOfTarget]
  
transformFRED <- function (file = "", date_start = NULL, date_end = NULL, transform = TRUE, frequency, targetNames, target_tcode, h) 
{

  # Function based on fbi R Package (https://github.com/cykbennie/fbi)
  
  if (!is.logical(transform)) 
    stop("'transform' must be logical.")
  if ((class(date_start) != "Date") && (!is.null(date_start))) 
    stop("'date_start' must be Date or NULL.")
  if ((class(date_end) != "Date") && (!is.null(date_end))) 
    stop("'date_end' must be Date or NULL.")
  if (class(date_start) == "Date") {
    if (as.numeric(format(date_start, "%d")) != 1) 
      stop("'date_start' must be Date whose day is 1.")
    if (date_start < as.Date("1959-01-01")) 
      stop("'date_start' must be later than 1959-01-01.")
  }
  if (class(date_end) == "Date") {
    if (as.numeric(format(date_end, "%d")) != 1) 
      stop("'date_end' must be Date whose day is 1.")
  }
  print(targetNames)
  
  rawdata <- read.csv(file)
  
  if(frequency==2) {rawdata <- rawdata[-1,]}
  
  ind <- !(rowSums(is.na(rawdata[,2:ncol(rawdata)])) == ncol(rawdata[,2:ncol(rawdata)])) # detect if all elements of a row are NA
  rawdata <- rawdata[ind,]
  header <- c("date", colnames(rawdata)[-1])[1:ncol(rawdata)]
  
  tcode <- rawdata[1,-1]
  
  if(!is.na(target_tcode)) {
    tcode[which(names(tcode) == targetNames)] <- target_tcode
    targetPos <- which(names(tcode) == targetNames)
  }
  
  cat("Target transformation code : ",as.integer(tcode[which(names(tcode) == targetNames)]),"\n", sep = "")
  cat("\n")
  
  date <- as.character(rawdata[-1,1])
  
  rawdata <- as.data.frame(rawdata[-1,])
  rawdata[,1] <- as.character(rawdata[,1])
  colnames(rawdata) <- header
  
  transxf <- function(x, tcode, h) {
    n <- length(x)
    small <- 1e-06
    y <- rep(NA, n)
    y1 <- rep(NA, n)
    if (tcode == 1) {
      y <- x
    }
    else if (tcode == 2) {
      y[2:n] <- x[2:n] - x[1:(n - 1)]
    }
    else if (tcode == 3) {
      y[3:n] <- x[3:n] - 2 * x[2:(n - 1)] + x[1:(n - 2)]
    }
    else if (tcode == 4) {
      if (min(x, na.rm = TRUE) > small) 
        y <- log(x)
    }
    else if (tcode == 5) {
      if (min(x, na.rm = TRUE) > small) {
        x <- log(x)
        y[2:n] <- x[2:n] - x[1:(n - 1)]
      }
    }
    else if (tcode == 6) {
      if (min(x, na.rm = TRUE) > small) {
        x <- log(x)
        y[3:n] <- x[3:n] - 2 * x[2:(n - 1)] + x[1:(n - 
                                                     2)]
      }
    }
    else if (tcode == 7) {
      y1[2:n] <- (x[2:n] - x[1:(n - 1)])/x[1:(n - 1)]
      y[3:n] <- y1[3:n] - y1[2:(n - 1)]
    }
    else if (tcode == 8) {
      if (min(x, na.rm = TRUE) > small) {
        if(h==0) {h=1}
        x <- log(x)
        y[(h+1):n] <- (x[(h+1):n] - x[1:(n - h)])/h
      }
    }
    return(y)
  }
  
  target <- NA
  if (transform) {
    N <- ncol(rawdata)
    data <- rawdata
    data[, 2:N] <- NA
    for (i in 2:N) {
      if(tcode[i-1]==8) {
        temp <- transxf(rawdata[, i], tcode[i - 1], h)
        target <- temp
        tcode[i - 1] <- 5
      }
      temp <- transxf(rawdata[, i], tcode[i - 1], h)
      data[, i] <- temp
    }
  }
  else {
    data <- rawdata
  }
  if (is.null(date_start)) 
    date_start <- as.Date("1959-01-01")
  if (is.null(date_end)) 
    date_end <- data[, 1][nrow(data)]
  # index_start <- which.max(data[, 1] == date_start)
  # index_end <- which.max(data[, 1] == date_end)
  # outdata <- data[index_start:index_end, ]
  outdata <- data[, ]
  
  #class(outdata) <- c("data.frame", "fredmd")
  returnList <- list(outdata,rawdata,target,tcode)
  
  return(returnList)
}
