library(gamlss)

# Example from gamlss manual p.83
data(db)
subdata <- subset(db,(age > 1 & age < 2))
#hist(subdata$head, col ="blue", border="red")

start_time <- Sys.time()
# p.89
#h5 <- gamlss(subdata$head~1, family=BCPE, nu.start=1, tau.start=1.7, sigma.start = 0.1, mu.start= 40)
h5 <- gamlss(head ~ poly(age,3),
             sigma.formula = ~poly(age,3),
             family=NO, data = subdata)

centiles(h5,xvar = subdata$age)



end_time <- Sys.time()

end_time - start_time


