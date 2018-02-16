library(gamlss)

# Example from gamlss manual p.83
data(db)
#subdata <- subset(db,(age > 1 & age < 2))
#hist(subdata$head, col ="blue", border="red")

start_time <- Sys.time()

model_BCPE <- gamlss(head ~ age,
             sigma.formula = ~age,
             # sigma.fix = TRUE,
             # nu.fix = TRUE,
             # tau.fix = TRUE,
             # mu.start = c(1.9,45),
             # sigma.start = 1.0,
             # nu.start = 0.1,
             # tau.start= 1.0,
             family=BCPE, 
             data = subdata)

centiles(model_BCPE,xvar = subdata$age)

end_time <- Sys.time()

end_time - start_time

model_BCPE

fitted(model_BCPE, "mu")[1]
fitted(model_BCPE, "sigma")[1]
fitted(model_BCPE, "nu")[1]
fitted(model_BCPE, "tau")[1]

