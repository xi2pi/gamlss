library(gamlss)

# Example from gamlss manual p.83
data(db)
subdata <- subset(db,(age > 1 & age < 2))
hist(subdata$head, col ="blue", border="red")

start_time <- Sys.time()
# p.89
h5 <- gamlss(subdata$head~1, family=BCPE, nu.start=1, tau.start=1.7, sigma.start = 0.1, mu.start= 40)

pdf.plot(h5,1,min=40,max=60,step=0.2)

end_time <- Sys.time()

end_time - start_time

# Save data frame
setwd("C:/Users/CatOnTour/Documents/GitHub/reference-curves-pub/translating_R2Python")
write.csv(subdata, file = "example_data.csv")
