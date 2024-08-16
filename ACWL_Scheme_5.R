
#########################################################################################
# 2 stages and 3 treatments

N<-1000 # sample size
ni<-1000 # sample counterfactual outcome using estimated g.opt and true outcome model
iter<-10 # replication 500

c1 = 3
c2 = 3
beta = 1

select1<-selects<-matrix(NA,iter,3) # percent of optimality at stage 1 and both stages
select2<-matrix(NA,iter,3) # percent of optimality at stage 2
E0.ys<-matrix(NA,iter,4) # estimated mean counterfactual outcome

accuracy_S1 <- accuracy_S2 <- numeric(iter)

for(i in 1:iter){
  # simulate baseline covariates
  x1<-rnorm(N, mean = 0, sd = 10) #rnorm(N)
  x2<-rnorm(N, mean = 0, sd = 10) #rnorm(N)
  x3<-rnorm(N, mean = 0, sd = 10) # rnorm(N)
  X0<-cbind(x1,x2,x3) # O1

  # X0 <- matrix(rnorm(3 * N, mean = 0, sd = 1), nrow = N, ncol = 3) 
  # X0 = data.frame(x1 = rnorm(N, mean = 0, sd = 1), x2 = rnorm(N, mean = 0, sd = 10), x3 = rnorm(N, mean = 0, sd = 10))


  O2 = rnorm(N) 

  ############### stage 1 data simulation ##############
  # simulate A1, stage 1 treatment with K1=3
  pi10<-rep(1/3,N); pi11<-rep(1/3,N); pi12<-rep(1/3,N)


  matrix.pi1<-cbind(pi10,pi11,pi12)
  # A1<-A.sim(matrix.pi1)
  result1 <-A.sim(matrix.pi1) # normalizing inside A.sim
  A1 <- result1$A; probs1 <- result1$probs


  # optimal g1.opt
  # g1.opt<-(x1>-1)*((x2>-0.5)+(x2>0.5))
  # Calculate the sum of each row in X0
  sums <- rowSums(X0)
  # Calculate g1_opt based on the condition of sums
  g1_opt <- ifelse(sums > 0, 2, 0)

  # stage 1 outcome R1
  z1 = rnorm(N,0,1)
  R1<-A1*sums + c1 + z1


  ############### stage 2 data simulation ##############
  # A2, stage 2 treatment with K2=3
  pi20<-rep(1/3,N); pi21<-rep(1/3,N); pi22<-rep(1/3,N)
  matrix.pi2<-cbind(pi20,pi21,pi22)
  # A2<-A.sim(matrix.pi2)
  result2 <-A.sim(matrix.pi1) # normalizing inside A.sim
  A2 <- result2$A; probs2 <- result2$probs

  # optimal g2.opt    
  # g2.opt<-(x3>-1)*((R1>0.5)+(R1>3)) 
  g2_opt <- apply(X0^2, 1, which.max) - 1 


  # stage 2 outcome R2
  z2 = rnorm(N,0,1)
  R2 <- (X0[cbind(1:N, A2+1)])^2+ O2*beta+ c2+ z2 

  ############### stage 2 Estimation #####################
  # Backward induction

  # estimate conditional means by regression
  REG2<-Reg.mu(Y=R2,As=cbind(A1,A2),H=cbind(X0,R1))
  mus2.reg<-REG2$mus.reg

  # calculate AIPW adaptive contrasts C and working orders l
  # input outcome and treatment vectors, estimated propensity and conditional means
  CLs2.a<-CL.AIPW(Y=R2,A=A2,pis.hat=probs2,mus.reg=mus2.reg)
  C2.a1<-CLs2.a$C.a1
  C2.a2<-CLs2.a$C.a2
  l2.a<-CLs2.a$l.a

  # Competing methods: straight regression and BOWL
  # contrasts C and working order l.reg
  l2.reg<-rep(NA,N)

  for(j in 1:N){
  	# straight reg, similar to Q-learning
    l2.reg[j]<-which(mus2.reg[j,]==max(mus2.reg[j,]))-1
  }

  # AIPW contrasts
  fit2.a1<-rpart(l2.a ~ x1+x2+x3+A1+R1, weights=C2.a1, method="class")
  fit2.a2<-rpart(l2.a ~ x1+x2+x3+A1+R1, weights=C2.a2, method="class")

  # predicted optimal treatments, matching scale of A2
  # g2.ow<-as.numeric(predict(fit2.ow,type="class"))-1
  g2.a1<-as.numeric(predict(fit2.a1,type="class"))-1
  g2.a2<-as.numeric(predict(fit2.a2,type="class"))-1

  # select2[i,]<-c(mean(l2.reg==g2.opt),mean(g2.ow==g2.opt),mean(g2.a1==g2.opt),mean(g2.a2==g2.opt))
  select2[i,]<-c(mean(l2.reg==g2.opt),mean(g2.a1==g2.opt),mean(g2.a2==g2.opt))


  ############### stage 1 Estimation #####################
  # calculate pseudo outcome (PO)

  # expected optimal stage 2 outcome
  E.R2.reg<-E.R2.a1<-E.R2.a2<-rep(NA,N)

  ## use observed R2 + E(loss), modified Q learning as in Huang et al.2015
  for(m in 1:N){
	E.R2.reg[m]<-R2[m] + mus2.reg[m,l2.reg[m]+1]-mus2.reg[m,A2[m]+1]
	E.R2.a1[m]<-R2[m] + mus2.reg[m,g2.a1[m]+1]-mus2.reg[m,A2[m]+1]
	E.R2.a2[m]<-R2[m] + mus2.reg[m,g2.a2[m]+1]-mus2.reg[m,A2[m]+1]
  }

  # pseudo outcomes
  PO.reg<-R1+E.R2.reg
  PO.a1<-R1+E.R2.a1
  PO.a2<-R1+E.R2.a2

  ########### straight regression #########
  REG1<-Reg.mu(Y=PO.reg,As=A1,H=X0)
  mus1.reg<-REG1$mus.reg
  l1.reg<-rep(NA,N)
  for(j in 1:N) l1.reg[j]<-which(mus1.reg[j,]==max(mus1.reg[j,]))-1


  ####### ACWL-C1 #########
  REG1.a1<-Reg.mu(Y=PO.a1,As=A1,H=X0)
  mus1.reg.a1<-REG1.a1$mus.reg
  CLs1.a1<-CL.AIPW(Y=PO.a1,A=A1,pis.hat=probs1, mus.reg=mus1.reg.a1)
  C1.a1<-CLs1.a1$C.a1
  l1.a1<-CLs1.a1$l.a

  fit1.a1<-rpart(l1.a1 ~ x1+x2+x3, weights=C1.a1,method="class")
  g1.a1<-predict(fit1.a1,type="class")


  ####### ACWL-C2 #########
  REG1.a2<-Reg.mu(Y=PO.a2,As=A1,H=X0)
  mus1.reg.a2<-REG1.a2$mus.reg
  CLs1.a2<-CL.AIPW(Y=PO.a2,A=A1,pis.hat=probs1,mus.reg=mus1.reg.a2)
  C1.a2<-CLs1.a2$C.a2
  l1.a2<-CLs1.a2$l.a

  fit1.a2<-rpart(l1.a2 ~ x1+x2+x3, weights=C1.a2,method="class")
  g1.a2<-predict(fit1.a2,type="class")

  select1[i,]<-c(mean(l1.reg==g1.opt),mean(g1.a1==g1.opt),mean(g1.a2==g1.opt))

  selects[i,]<-c(mean(l1.reg==g1.opt & l2.reg==g2.opt), mean(g1.a1==g1.opt & g2.a1==g2.opt),mean(g1.a2==g1.opt & g2.a2==g2.opt))

  #####################################################################################
  ######## estimate the counterfactual mean using g.hat and true outcome model ########

  E0.yi<-matrix(NA,ni,4) # a matrix of estimated counterfactual mean by different methods
  # straight regression models
  RegModel1<-REG1$RegModel
  RegModel2<-REG2$RegModel

  g1.a1 <- g2.a1 <- g1.opt <- g2.opt <- numeric(ni)

  for(k in 1:ni){
    # simulate one sample (a new patient)
    x1<-rnorm(1, mean = 0, sd = 10);x2<-rnorm(1, mean = 0, sd = 10);x3<-rnorm(1, mean = 0, sd = 10);O2.k<-rnorm(1)
    X0.k<-c(x1,x2,x3)
    # true optimal g and outcome at stages 1 and 2 
    sums <- sum(X0.k)
    g1k<-ifelse(sums > 0, 2, 0)
    z1.k = rnorm(1,0,1)
    R1k<-g1k*sums + c1 + z1.k
    g2k<- which.max(X0.k^2) - 1 
    z2.k = rnorm(1,0,1)
    R2k = X0.k[g2k+1]^2+O2.k*beta+ c2+ z2.k 


    ##### predicting the optimal g and plug in the true outcome model for counterfactual mean

    ####### stage 1 prediction #######

    # straight regression method
    # estimate outcome under different treatments
    mu10.reg<-sum(c(1,X0.k,0,0,X0.k*0,X0.k*0)*coef(RegModel1))
    mu11.reg<-sum(c(1,X0.k,1,0,X0.k*1,X0.k*0)*coef(RegModel1))
    mu12.reg<-sum(c(1,X0.k,0,1,X0.k*0,X0.k*1)*coef(RegModel1))
    mus1.reg<-c(mu10.reg,mu11.reg,mu12.reg)
    l1.reg.k<-which(mus1.reg==max(mus1.reg))-1
    R1.reg.k<- l1.reg.k*sums + c1 + z1.k #X0.k^2+ O2.k*beta+ c2+ z2 
      
    # newdata for stage 2 prediction
    X1.k<-c(X0.k,R1.reg.k); l1.bi<-c(I(l1.reg.k==1),I(l1.reg.k==2))

    newdata1<-data.frame(x1,x2,x3)

    # ACWL-C1
    g1.a1.k<-as.numeric(predict(fit1.a1,newdata=newdata1,type="class"))-1
    R1.a1.k<-g1.a1.k*sums + c1 + z1.k 

    # newdata for stage 2 prediction
    newdata2.a1<-data.frame(newdata1,g1.a1.k,R1.a1.k)

    g1.a1[k] = g1.a1.k
    g1.opt[k] = g1k

    # ACWL-C2
    g1.a2.k<-as.numeric(predict(fit1.a2,newdata=newdata1,type="class"))-1
    R1.a2.k<-g1.a2.k+ O2.k*beta+ c2+ z2.k 
    # newdata for stage 2 prediction
    newdata2.a2<-data.frame(newdata1,g1.a2.k,R1.a2.k)

    ####### stage 2 prediction #######
    colnames(newdata2.a1)<-colnames(newdata2.a2)<-c("x1","x2","x3","A1","R1") # c("x1","x2","x3","x4","x5","A1","R1")

    # straight regression method
    mu20.reg<-sum(c(1,X1.k,l1.bi,0,0,X1.k*0,X1.k*0,l1.bi*0,l1.bi*0)*coef(RegModel2))
    mu21.reg<-sum(c(1,X1.k,l1.bi,1,0,X1.k*1,X1.k*0,l1.bi*1,l1.bi*0)*coef(RegModel2))
    mu22.reg<-sum(c(1,X1.k,l1.bi,0,1,X1.k*0,X1.k*1,l1.bi*0,l1.bi*1)*coef(RegModel2))
    mus2.reg<-c(mu20.reg,mu21.reg,mu22.reg)
    l2.reg.k<-which(mus2.reg==max(mus2.reg))-1
    R2.reg.k<-X0.k[l2.reg.k+1]^2+ O2.k*beta+ c2+ z2.k  # exp(1.26-abs(1.5*x3-2)*(l2.reg.k-g2k)^2)#+rnorm(1,0,1)


    # ACWL-C1
    g2.a1.k<-as.numeric(predict(fit2.a1,newdata=newdata2.a1,type="class"))-1
    R2.a1.k<-X0.k[g2.a1.k+1]^2+ O2.k*beta+ c2+ z2.k   # exp(1.26-abs(1.5*x3-2)*(g2.a1.k-g2k)^2)#+rnorm(1,0,1)

    g2.a1[k] = g2.a1.k
    g2.opt[k] = g2k

    # ACWL-C2
    g2.a2.k<-as.numeric(predict(fit2.a2,newdata=newdata2.a2,type="class"))-1
    R2.a2.k<-X0.k[g2.a2.k+1]^2+ O2.k*beta+ c2+ z2.k  # exp(1.26-abs(1.5*x3-2)*(g2.a2.k-g2k)^2)#+rnorm(1,0,1)

    E0.yi[k,1]<-R1k+R2k
    E0.yi[k,2]<-R1.reg.k+R2.reg.k
    E0.yi[k,3]<-R1.a1.k+R2.a1.k
    E0.yi[k,4]<-R1.a2.k+R2.a2.k

  }
 

  # Calculate the accuracy for stage 1
  accuracystage1 <- mean(g1.a1 == g1.opt)
  cat("Accuracy for Stage 1: ", accuracystage1, "\n")

  # Calculate the accuracy for stage 2
  accuracystage2 <- mean(g2.a1 == g2.opt)
  cat("Accuracy for Stage 2: ", accuracystage2, "\n")


  # Calculate the overall accuracy for both stages
  accuracystageOverall <- mean(g1.a1 == g1.opt & g2.a1 == g2.opt)
  cat("Overall Accuracy: ", accuracystageOverall, "\n")

  accuracy_S1[i] = accuracystage1
  accuracy_S2[i] = accuracystage2


  E0.ys[i,1]<-mean(E0.yi[,1])
  E0.ys[i,2]<-mean(E0.yi[,2])
  E0.ys[i,3]<-mean(E0.yi[,3])
  E0.ys[i,4]<-mean(E0.yi[,4])


  cat("========================= Iteration : ", i , " completed =========================\n")
}

colnames(select1)<-colnames(selects)<-c("Reg","ACWL-C1","ACWL-C2")
colnames(select2)<-c("Reg","ACWL-C1","ACWL-C2")

colnames(E0.ys)<-c("Truth","Reg","ACWL-C1","ACWL-C2")

# summary2(select1)
# summary2(select2)
# summary2(selects)
summary2(E0.ys)
print("\n\n")
cat("Mean accuracy for Stage 1: ", mean(accuracy_S1), "\n")
cat("Mean accuracy for Stage 2: ", mean(accuracy_S2), "\n")

