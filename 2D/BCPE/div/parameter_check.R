data(abdom)

abd7 <- gamlss(y~1, sigma.formula=~1, nu.formula=~1,
               tau.formula=~1, family=BCPE, data=abdom, n.cyc=50)
abd7

pdf.plot(abd7,1,min=1,max=500,step=1)


# Mu Coefficients:
#   (Intercept)  
# 226.2  
# Sigma Coefficients:
#   (Intercept)  
# -0.9291  
# Nu Coefficients:
#   (Intercept)  
# 1.014  
# Tau Coefficients:
#   (Intercept)  
# 2.327

fitted(abd7, "mu")[1]
fitted(abd7, "sigma")[1]
fitted(abd7, "nu")[1]
fitted(abd7, "tau")[1]