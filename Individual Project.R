
# title: "Individual Project"
# author: "Zhijiao Chen"
# date: "2020/4/15"


# Individual Project

### Zhijiao Chen

############################################################################################

### Part One: Data Visualization 

# The time series I'm going to analyze is S&P/Case-Shiller MA-Boston Home Price Index. Since it is a seasonally adjusted time series, there's no seasonality in this time series.

#     S&P/Case-Shiller MA-Boston Home Price Index
#     Source: S&P Dow Jones Indices LLC    
#     Release: S&P/Case-Shiller Home Price Indices    
#     Units:  Index Jan 2000=100, Seasonally Adjusted 
#     Frequency:  Monthly

rm(list=ls())
library(forecast)
library(ggplot2)
setwd("C:/Users/Catherine Chen/Dropbox/Brandeis/★ECONFIN 250 Forecasting/final project")



BOXRSA.data <- read.csv("BOXRSA.csv")
BOXRSA.ts <- ts(BOXRSA.data$BOXRSA,start=c(1987,01), freq=12)

plot(BOXRSA.ts)
Acf(BOXRSA.ts)
Pacf(BOXRSA.ts)


# From Figure 2 and Figure 3, we can see that ACF decays slowly while PACF drops rapidly. Note that the PACF plot has a significant spike only at lag 1. We can see a very obvious trend in this Seasonally Adjusted data.



############################################################################################

### Part Two: Splitting the data

# For this topic, I care about a long term trend of Boston Home Price Index. When dealing with these topics, a train/validation structure is better, and I can predict for a period of time. So I chose use a train/validation structure.

# I split the whole time series into two parts: Training part (From 1987-01 to 2013-12) and validation part (From 2014-01 to 2020-01).


BOXRSA.trend <- tslm(BOXRSA.ts ~ trend)
# set train and validation
BOXRSA.train <- window(BOXRSA.ts,end=c(2013,12))
BOXRSA.valid <- window(BOXRSA.ts,start=c(2014,1))
nValid <- length(BOXRSA.valid)



############################################################################################

### Part Three: ARIMA model


# ARIMA is a very popular technique for time series modelling. It describes the correlation between data points and takes into account the difference of the values.

# First of all, I ran a Dickey-Fuller test to see whether the time series is stationary or not. Since the null hypothesis cannot be rejected, so random walk plus drift exists, which make sense given that there is a growth rate. So I add a first difference and use an ARIMA model. The model with only one order of differencing assumes a constant average trend--it is essentially a fine-tuned random walk model with growth--and it, therefore, makes relatively conservative trend projections.



library(tseries)
library(urca)

# ADF test on unit root data, use the auto lag selection feature with the BIC
df.test <- ur.df(BOXRSA.ts,type="trend",selectlags="BIC")
print("trend ADF test")
print(summary(df.test))
sumStats <- summary(df.test)
#Third statistic since type = 'trend'
teststat <- sumStats@teststat[3]
# critical value at 5 percent
critical <- sumStats@cval[2]
teststat > critical


# We cannot reject the null, so Random walk plus drift (not AR process), which make sense given that there is a growth rate

# Also, I use the auto-arima function to find the best p and q in the ARIMA model. Also, I plot a linear trend for comparison. The RMSE is 0.482 for training set and 7.943 for validation set.


# ARIMA(d=1) model
BOXRSA.arima <- auto.arima(BOXRSA.train,d=1,ic="bic",seasonal=FALSE)
print(summary(BOXRSA.arima))

BOXRSA.arima.fcast <- forecast(BOXRSA.arima,h=nValid,level=95)
plot(BOXRSA.arima.fcast)
lines(BOXRSA.valid)

# plot linear trend for comparison
lines(BOXRSA.trend$fitted.values,col="red")
grid()

print(accuracy(BOXRSA.arima.fcast,BOXRSA.valid))

############################################################################################

### Part Four: Basic Holt filter method

# This method takes into account the trend of the dataset. Since our time series has an obvious trend, I tried Holt’s linear trend model. Empirical evidence indicates that these methods tend to over-forecast, especially for longer forecast horizons. Motivated by this observation, Gardner & McKenzie (1985) introduced a parameter that “dampens” the trend to a flat line sometime in the future. The RMSE is 0.540 for training set and 23.940 for validation set.

ses.pred <- holt(BOXRSA.train,damped = TRUE, h=nValid, level=95)
summary(ses.pred)
plot(ses.pred)
lines(BOXRSA.valid)
grid()
print(accuracy(ses.pred,BOXRSA.valid))


############################################################################################

### Part Five: Baseline 

# Finally, we use some baseline forecasting model to see how our models work and to find out the best model. 

#### Simple Linear Model

# Since this time series has an obvious trend and with no seasonality, it's reasonable to have a try on linear forecast.

# First, estimate addiive trend/seasonal filter
train.lm <- tslm(BOXRSA.train ~ trend )
#        Now, build forecasts for validation periods
train.lm.pred <- forecast(train.lm, h=nValid, level=95)

# plot all the results
plot(train.lm.pred)
lines(BOXRSA.valid)

grid()

print(accuracy(train.lm.pred,BOXRSA.valid))



#### Naive method & drift method


# Find multiple naive forecasts
BOXRSA.lm.naive <-  naive(BOXRSA.train, h = nValid, level = 0)
BOXRSA.lm.drift <-  rwf(BOXRSA.train, h = nValid, drift = TRUE, level = 0)

# plot the forecast in the validation period
plot(BOXRSA.ts)

# plot data in the validation period
lines(BOXRSA.lm.naive$mean,col="blue")
lines(BOXRSA.lm.drift$mean,col="red",lwd=2)
lines(BOXRSA.valid,lty="dashed")
legend("topleft",col=c("blue","red"),legend=c("naive","drift"),lty=1)
grid()

print(accuracy(BOXRSA.lm.naive,BOXRSA.valid))
print(accuracy(BOXRSA.lm.drift,BOXRSA.valid))


# After that, I created a table and compared all the validation RMSEs. So it's clear that ARIMA(d=1) is the best model to forecast MA-Boston Home Price Index with a RMSE only 7.943.


library(dplyr)
library(kableExtra)
RMSE <- data.frame('Model'=c("Naive","Drift","Holt filter","ARIMA (d=1)","Linear"),'Validation_RMSE'= c(accuracy(BOXRSA.lm.naive,BOXRSA.valid)[2,2],accuracy(BOXRSA.lm.drift,BOXRSA.valid)[2,2],accuracy(ses.pred,BOXRSA.valid)[2,2],accuracy(BOXRSA.arima.fcast,BOXRSA.valid)[2,2],accuracy(train.lm.pred,BOXRSA.valid)[2,2]))

RMSE <- RMSE %>% arrange(Validation_RMSE)

RMSE %>% 
  knitr::kable(digits=3,align='c')%>% 
  kable_styling(full_width = F)


# So it's clear that ARIMA(d=1) is the best model to forecast MA-Boston Home Price Index with a RMSE only 7.943.

############################################################################################

### Part Six: Compare ARIMA with Simple Linear model: Diebold/Mariano


# Here we use Diebold/Mariano test to verify whether ARIMA is the best model. I compared ARIMA with all other models, and Table 2 shows the p-value of these Diebold/Mariano tests.


# This section code suggests how you would probably use the Diebold/Mariano test in practice
# We want to know if the firm should bother to shift to ARIMA forecasting from other models
# A one sided test is most appropriate here, with the alternative = less
# A rejection (low p-value) means you should shift to the ARIMA forecast

print("Diebold/Mariano ARIMA versus naive test: alternative = less")
print(dm.test(residuals(BOXRSA.arima.fcast),residuals(ses.pred),alternative="less"))


DM <- data.frame('Model'=c("Naive","Drift","Holt filter","Linear"),'p.value'= c(dm.test(residuals(BOXRSA.arima.fcast),residuals(BOXRSA.lm.naive),alternative="less")$p.value,
  dm.test(residuals(BOXRSA.arima.fcast),residuals(BOXRSA.lm.drift),alternative="less")$p.value,
  dm.test(residuals(BOXRSA.arima.fcast),residuals(ses.pred),alternative="less")$p.value,
  dm.test(residuals(BOXRSA.arima.fcast),residuals(train.lm.pred),alternative="less")$p.value))

DM <- DM %>% arrange(desc(p.value))

DM %>% 
  knitr::kable(align='c')%>% 
  kable_styling(full_width = F)


# As shown in the table, p-values are all very small, so we can say that it is statistically significant that ARIMA model outperforms all other models.


