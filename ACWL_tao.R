
library(rpart)
require(nnet)



######### functions for ACWL and simulations ########

# function to sample treatment A
# input matrix.pi as a matrix of sampling probabilities, which could be non-normalized
A.sim<-function(matrix.pi){
  N<-nrow(matrix.pi) # sample size
  K<-ncol(matrix.pi) # treatment options
  if(N<=1 | K<=1) stop("Sample size or treatment options are insufficient!")
  if(min(matrix.pi)<0) stop("Treatment probabilities should not be negative!")

  # normalize probabilities to add up to 1 and simulate treatment A for each row
  pis<-apply(matrix.pi,1,sum)
  probs<-matrix(NA,N,K)
  A<-rep(NA,N)
  for(i in 1:N){
    probs[i,]<-matrix.pi[i,]/pis[i]
    A[i]<-sample(0:(K-1),1,prob=probs[i,])
  }
  class.A<-sort(unique(A))
  colnames(probs)<-paste("A=",class.A,sep="")
  return(list(A = A, probs = probs))
}

# function to estimate propensity score
# input treatment vector A and covariate matrix Xs
M.propen<-function(A,Xs){
  if(ncol(as.matrix(A))!=1) stop("Cannot handle multiple stages of treatments together!")
  if(length(A)!= nrow(as.matrix(Xs))) stop("A and Xs do not match in dimension!")
  if(length(unique(A))<=1) stop("Treament options are insufficient!")
  class.A<-sort(unique(A))

  # require(nnet)
  s.data<-data.frame(A,Xs)
  # multinomial regression with output suppressed
  model<-capture.output(mlogit<-multinom(A ~., data=s.data))
  s.p<-predict(mlogit,s.data,"probs")
  colnames(s.p)<-paste("A=",class.A,sep="")

  s.p
}

# function to estimate conditional means for multiple stages
# input Y as a continous outcome of interest
# input As = (A1, A2, ...) as a matrix of treatments at multiple stages; Stage t has treatment K_t options labeled as 0, 1, ..., K_t-1.
# input H as a matrix of covariates before assigning final treatment, excluding previous treatment variables
Reg.mu<-function(Y,As,H){
  if(nrow(as.matrix(As))!=nrow(as.matrix(H))) stop("Treatment and Covariates do not match in dimension!")
  Ts<-ncol(as.matrix(As)) # number of stages
    
  N<-nrow(as.matrix(As))
  if(Ts<0 || Ts>3) stop("Only support 1 to 3 stages!")
  H<-as.matrix(H)

  if(Ts==1){
    A1<-as.matrix(As)[,1]
    KT<-length(unique(A1)) # treatment options at last stage
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ H*factor(A1))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1=rep(sort(unique(A1))[k],N)))
  }
  if(Ts==2){
    A1<-as.matrix(As)[,1];A2<-as.matrix(As)[,2]
    KT<-length(unique(A2))
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ (H + factor(A1))*factor(A2))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1,A2=rep(sort(unique(A2))[k],N)))
  }
  if(Ts==3){
    A1<-as.matrix(As)[,1];A2<-as.matrix(As)[,2];A3<-as.matrix(As)[,3]
    KT<-length(unique(A3))
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ (H + factor(A1) + factor(A2))*factor(A3))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1,A2,A3=rep(sort(unique(A3))[k],N)))
  }

  output<-list(mus.reg, RegModel)
  names(output)<-c("mus.reg","RegModel")
  output
}

# function to calculate AIPW adaptive contrasts and working orders
# input outcome Y, treatment vector A, estimated propensity matrix pis.hat and regression-based conditional means mus.reg
CL.AIPW<-function(Y,A,pis.hat,mus.reg){
  class.A<-sort(unique(A))
  K<-length(class.A)
  N<-length(A)
  if(K<2 | N<2) stop("No multiple treatments or samples!")
  if(ncol(pis.hat)!=K | ncol(mus.reg)!=K | nrow(pis.hat)!=N | nrow(mus.reg)!=N) stop("Treatment, propensity or conditional means do not match!")

  #AIPW estimates; this is bias correction step
  mus.a<-matrix(NA,N,K)
  for(k in 1:K){
    mus.a[,k]<-(A==class.A[k])*Y/pis.hat[,k]+(1-(A==class.A[k])/pis.hat[,k])*mus.reg[,k]
  }
  # C.a1 and C.a2 are AIPW contrasts; l.a is AIPW working order
  C.a1<-C.a2<-l.a<-rep(NA,N)
  for(i in 1:N){
    # largest vs. second largest
    C.a1[i]<-max(mus.a[i,])-sort(mus.a[i,],decreasing=T)[2]
    # largest vs. smallest
    C.a2[i]<-max(mus.a[i,])-min(mus.a[i,])
    # minus 1 to match A's range of 0,...,K-1
    l.a[i]<-which(mus.a[i,]==max(mus.a[i,])) # -1
  }
  output<-data.frame(C.a1, C.a2, l.a)
  output
}

# function to summarize simulation results
summary2<-function(x){
  s1<-summary(x)
  if(is.matrix(x)){
    SD<-apply(x,2,sd)
  } else{
    SD<-sd(x)
  }
  s2<-list(s1,SD)
  names(s2)<-c("summary","SD")
  return(s2)
}


ensure_vector <- function(var) {
  if (is.null(var)) {
    stop("Error: Variable is NULL")
  }
  if (!is.vector(var) || dim(var) != NULL) {
    var <- as.vector(var)  # Convert to vector if it's not
  }
  return(var)
}




# Work with  actions 1, 2, 3,  
train_ACWL <- function(job_id, O1, O2, A1, A2, probs1, probs2, R1, R2, g1.opt, g2.opt, config_number, contrast = 1, method = "tao") {
  cat("Train model: ", method, "\n")

  # N <- length(x1)  
  N <- nrow(O1) 
  cat("Number of row in O1 is: ", nrow(O1), "\n ")
           
  # cat("Debug: Dimensions of train_input_np are", dim(O1), "and the data type is", class(O1), "\n")

  # Directly call ensure_vector for each variable and update 
  # x1 <- ensure_vector(x1)
  # x2 <- ensure_vector(x2)
  # x3 <- ensure_vector(x3)
  # x4 <- ensure_vector(x4)
  # x5 <- ensure_vector(x5)

  A1 <- ensure_vector(A1)
  A2 <- ensure_vector(A2)
  R1 <- ensure_vector(R1)
  R2 <- ensure_vector(R2)
  g1.opt <- ensure_vector(g1.opt)
  g2.opt <- ensure_vector(g2.opt)

  colnames_O1= paste("x", 1:ncol(O1), sep="")
  # colnames_O2= paste("x", 1:ncol(O2), sep="")

  ############### stage 2 Estimation #####################
  # Stage 2 Estimation (Backward induction) : estimate conditional means by regression

  # O2 will possibly be in this estimation also in H = cbind(O1, R1, O2) # DISCUSS **********************************************
  REG2 <- Reg.mu(Y = R2, As = cbind(A1, A2), H = cbind(O1, R1))
  mus2.reg <- REG2$mus.reg
  CLs2.a <- CL.AIPW(Y = R2, A = A2, pis.hat = probs2, mus.reg = mus2.reg)

  # main difference betwen two contrasts here!!!  
  C2.a1<-CLs2.a$C.a1 
  if(contrast == 2){
      C2.a1<-CLs2.a$C.a2
  }
  l2.a<-CLs2.a$l.a
    
 
  # ################################    STRAIGHT regression / Q learning  ################################
  # # contrasts C and working order l.reg
  # l2.reg<-rep(NA,N)
  # for(j in 1:N){
  #   # straight reg, similar to Q-learning
  #   l2.reg[j]<-which(mus2.reg[j,]==max(mus2.reg[j,]))
  # }
    

  # Weighted classification using CART
  # fit2.a1 <- rpart(l2.a ~ x1 + x2 + x3 + x4 + x5 + A1 + R1, weights = C2.a1, method = "class")

  # Convert matrix to a data frame
  train_input_df <- as.data.frame(O1)
  names(train_input_df) <- colnames_O1 # c("x1", "x2", "x3", "x4", "x5")  

  # Add additional variables
  train_input_df$A1 <- A1
  train_input_df$R1 <- R1
  # train_input_df$O2 <- O2 # O2 will go here if we have one    # DISCUSS **********************************************

  # Fit the model
  fit2.a1 <- rpart(l2.a ~ ., data = train_input_df, weights = C2.a1, method = "class")
  g2.a1 <- as.numeric(predict(fit2.a1, type = "class")) # - 1
    
  ############### stage 1 Estimation #####################
  # Stage 1 Estimation : expected optimal stage 2 outcome
    
  # E.R2.reg<-rep(NA, N) #    STRAIGHT regression / Q learning

  E.R2.a1 <- rep(NA, N)
    
  for (m in 1:N){
      # E.R2.reg[m]<-R2[m] + mus2.reg[m,l2.reg[m]]-mus2.reg[m,A2[m]]  #    STRAIGHT regression / Q learning
      E.R2.a1[m] <- R2[m] + mus2.reg[m, g2.a1[m]] - mus2.reg[m,A2[m]]
  }
    
  # calculate pseudo outcome (PO)
  # PO.reg<- R1 + E.R2.reg  # STRAIGHT regression / Q learning
  PO.a1 <- R1 + E.R2.a1
    
  # ########### straight regression / Linear Q-learning #########
  # REG1<-Reg.mu(Y=PO.reg,As=A1,H=O1)
  # mus1.reg<-REG1$mus.reg
  # l1.reg<-rep(NA,N)
  # for(j in 1:N) l1.reg[j]<-which(mus1.reg[j,]==max(mus1.reg[j,]))
    
    
  ####### ACWL-Contrast #########
  REG1.a1 <- Reg.mu(Y = PO.a1, As = A1, H = O1)
  mus1.reg.a1 <- REG1.a1$mus.reg
  CLs1.a1 <- CL.AIPW(Y = PO.a1, A = A1, pis.hat = probs1, mus.reg = mus1.reg.a1)

  C1.a1<-CLs1.a1$C.a1
  if(contrast == 2){
      C1.a1<-CLs1.a1$C.a2
  }
  l1.a1<-CLs1.a1$l.a

  # fit1.a1 <- rpart(l1.a1 ~ x1 + x2 + x3 + x4 + x5, weights = C1.a1, method = "class")
  # Convert matrix to a data frame
  train_input <- as.data.frame(O1)
  names(train_input) <-  colnames_O1 # paste("x", 1:ncol(train_input), sep="")

  fit1.a1 <- rpart(l1.a1 ~ ., data = train_input, weights = C1.a1, method = "class")
  g1.a1 <- as.numeric(predict(fit1.a1, type = "class")) # - 1
    
    
  # if(method=="linear"){
  #     select2 <- mean(l2.reg == g2.opt)
  #     select1 <- mean(l1.reg == g1.opt)
  #     selects <- mean(l1.reg == g1.opt & l2.reg == g2.opt)}
  # else{
  #     select2 <- mean(g2.a1 == g2.opt)
  #     select1 <- mean(g1.a1 == g1.opt)
  #     selects <- mean(g1.a1 == g1.opt & g2.a1 == g2.opt)}

  select2 <- mean(g2.a1 == g2.opt)
  select1 <- mean(g1.a1 == g1.opt)
  selects <- mean(g1.a1 == g1.opt & g2.a1 == g2.opt)

  cat("Saving Tao's model now \n")
  # Save models 
  model_dir <- sprintf("models/%s", job_id)
  if (!dir.exists(model_dir)) {
    dir.create(model_dir)
  }

   
  # # Define and save the file paths for the first set of regression models
  # file_path_fit1_reg <- sprintf("%s/best_model_ACWL_stage1_tao_fit1_reg_config_number_%d.rds", model_dir, config_number)
  # file_path_fit2_reg <- sprintf("%s/best_model_ACWL_stage2_tao_fit2_reg_config_number_%d.rds", model_dir, config_number)
  # saveRDS(REG1, file = file_path_fit1_reg)
  # saveRDS(REG2, file = file_path_fit2_reg)
  
  # Define and save the file paths for the second set of models
  file_path_fit1 <- sprintf("%s/best_model_ACWL_stage1_tao_fit1_config_number_%d.rds", model_dir, config_number)
  file_path_fit2 <- sprintf("%s/best_model_ACWL_stage2_tao_fit2_config_number_%d.rds", model_dir, config_number)
  saveRDS(fit1.a1, file = file_path_fit1)
  saveRDS(fit2.a1, file = file_path_fit2)


  # Return the proportions of correct decisions
  return(list(select2 = select2, select1 = select1, selects = selects))
}






test_ACWL <- function(O1, O2, x1, x2, x3, x4, x5, g1k, g2k, noiseless, config_number, job_id, method= "tao") {
  cat("Test model: ", method, "\n")

  ni <- nrow(O1) 
  cat("Number of row in O1 is: ", nrow(O1), "\n ")

  E0.yi <- matrix(NA, ni, 2)  # a matrix of estimated counterfactual mean by different methods
 
  # Initializing vectors for storage
  g1.a1 <- g2.a1 <- R1.a1 <- R2.a1 <- numeric(ni)
  g1.reg <- g2.reg <- R1.reg <- R2.reg <- numeric(ni)

  model_dir <- sprintf("models/%s", job_id)

  # Loading saved models with config number in the file names
  # fit1.reg <- readRDS(sprintf("%s/best_model_ACWL_stage1_tao_fit1_reg_config_number_%d.rds", model_dir, config_number))
  # fit2.reg <- readRDS(sprintf("%s/best_model_ACWL_stage2_tao_fit2_reg_config_number_%d.rds", model_dir, config_number))
  fit1.a1 <- readRDS(sprintf("%s/best_model_ACWL_stage1_tao_fit1_config_number_%d.rds", model_dir, config_number))
  fit2.a1 <- readRDS(sprintf("%s/best_model_ACWL_stage2_tao_fit2_config_number_%d.rds", model_dir, config_number))
        
  # # straight regression models
  # fit1.reg<-fit1.reg$RegModel
  # fit2.reg<-fit2.reg$RegModel

  colnames_O1 = paste("x", 1:ncol(O1), sep="")

  # Predicting the optimal g and plug in the true outcome model for counterfactual mean
  for (k in 1:ni) {
    X0.k <- O1[k, ] #c(x1[k], x2[k], x3[k], x4[k], x5[k])

    z1 <- rnorm(1, mean = 0, sd = ifelse(noiseless, 0, 1))
    z2 <- rnorm(1, mean = 0, sd = ifelse(noiseless, 0, 1))
    
    ################################    STRAIGHT regression / Linear Q learning  ################################
    # # Estimate outcomes under different treatments
    # mu10.reg<-sum(c(1,X0.k,0,0,X0.k*0,X0.k*0)*coef(fit1.reg))
    # mu11.reg<-sum(c(1,X0.k,1,0,X0.k*1,X0.k*0)*coef(fit1.reg))
    # mu12.reg<-sum(c(1,X0.k,0,1,X0.k*0,X0.k*1)*coef(fit1.reg))
    # mus1.reg<-c(mu10.reg,mu11.reg,mu12.reg)
    # g1.reg[k]<-which(mus1.reg==max(mus1.reg)) #-1
    # R1.reg[k]<- exp(1.5 - abs(1.5 * x1[k] + 2) * (g1.reg[k] - g1k[k])^2) + z1                
      
    # X1.k <- c(X0.k, R1.reg[k])
    # l1.bi <- c(I(g1.reg[k]==1),I(g1.reg[k]==2))
                          
    # # straight regression method
    # mu20.reg<-sum(c(1,X1.k,l1.bi,0,0,X1.k*0,X1.k*0,l1.bi*0,l1.bi*0)*coef(fit2.reg))
    # mu21.reg<-sum(c(1,X1.k,l1.bi,1,0,X1.k*1,X1.k*0,l1.bi*1,l1.bi*0)*coef(fit2.reg))
    # mu22.reg<-sum(c(1,X1.k,l1.bi,0,1,X1.k*0,X1.k*1,l1.bi*0,l1.bi*1)*coef(fit2.reg))
    # mus2.reg <- c(mu20.reg,mu21.reg,mu22.reg)
    # g2.reg[k] <- which(mus2.reg == max(mus2.reg))
    # R2.reg[k] <- exp(1.26 - abs(1.5 * x3[k] - 2) * (g2.reg[k] - g2k[k])^2) + z2 

    ################################    ACWL  ################################

    newdata1 <- data.frame(t(O1[k, ] ))
    colnames(newdata1) <- colnames_O1
  
    g1.a1[k] <- as.numeric(predict(fit1.a1, newdata = newdata1, type = "class")) #- 1
    R1.a1[k] <- exp(1.5 - abs(1.5 * x1[k] + 2) * (g1.a1[k] - g1k[k])^2) + z1
                       
    newdata2.a1 <- data.frame(x1=x1[k], x2=x2[k], x3=x3[k], x4=x4[k], x5=x5[k], A1=g1.a1[k], R1=R1.a1[k])
    # newdata2.a1 <- data.frame(O1[k, ], A1=g1.a1[k], R1=R1.a1[k])
    g2.a1[k] <- as.numeric(predict(fit2.a1, newdata = newdata2.a1, type = "class")) #- 1
                       
    R2.a1[k] <- exp(1.26 - abs(1.5 * x3[k] - 2) * (g2.a1[k] - g2k[k])^2) + z2
  }
  
  # if (method == "linear") {
  #   R1.a1 <- R1.reg
  #   R2.a1 <- R2.reg
  #   g1.a1 <- g1.reg
  #   g2.a1 <- g2.reg
  #   select2 <- mean(g2.reg == g2k)
  #   select1 <- mean(g1.reg == g1k)
  #   selects <- mean(g1.reg == g1k & g2.reg == g2k)
  # } else {
  #   select2 <- mean(g2.a1 == g2k)
  #   select1 <- mean(g1.a1 == g1k)
  #   selects <- mean(g1.a1 == g1k & g2.a1 == g2k)
  # }


  select2 <- mean(g2.a1 == g2k)
  select1 <- mean(g1.a1 == g1k)
  selects <- mean(g1.a1 == g1k & g2.a1 == g2k)

  return(list(
    R1.a1 = R1.a1,
    R2.a1 = R2.a1,
    g1.a1 = g1.a1,
    g2.a1 = g2.a1,
    select2 = select2,
    select1 = select1,
    selects = selects
  ))
}










