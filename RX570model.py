
#==============================================================================================
#--------------------------LINEAR REGRESSION---------------------------------------------------
#==============================================================================================

#  Now use the pandas module to perform linear regression to determine 
# a model for GPU price relation to coin price
def rx570model(lag=None):
  import statsmodels.formula.api as sm
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import datetime
  
  # Import Ethereum Price
  dfPrice=pd.read_csv('inputs/export-EtherPrice.csv')
  dfPrice['Date'] = pd.to_datetime(dfPrice['Date(UTC)']) 
  # Import Price of GPUs
  #dfRX570 = pd.read_csv('inputs/rx570Prices.csv')
  dfRX570 = pd.read_csv('inputs/June2018RX570.csv')
  dfRX570['Date'] = pd.to_datetime(dfRX570['date'])
  dfRX570['unix'] = dfRX570.Date.astype('int64')*10**-9 
  
  ##Import networkhashrate from etherscan (2018-02-13)
  #dfHashrate=pd.read_csv('inputs/export-NetworkHash.csv')
  ## add hashRate to Price df
  #dfPrice['hashrate'] = dfHashrate.Value
  ## pull in total market cap from coindance
  #dfCap = pd.read_csv('imputs/globalMarketCap.csv')
  #dfCap['Date'] = pd.to_datetime(dfCap['Label'])
  #dfCap['unix'] = dfCap.Date.astype('int64')
  ## interpolate Market cap date values to the rx570 dates
  #from scipy.interpolate import interp1d
  #f = interp1d(dfCap.unix, dfCap['Altcoin Market Cap'] )
  #dfRX570['MCap'] = f(dfRX570.unix)
  # interpolate eth price date values to the rx570 dates
  from scipy.interpolate import interp1d
  f = interp1d(dfPrice.UnixTimeStamp, dfPrice['Value'], fill_value="extrapolate" )
  dfRX570['eth'] = f(dfRX570.unix)
  plot =False
#  plot =True 
  if plot:
    # FIrst Compare GPU prices to ETH price
    plt.figure()
    plt.plot(dfRX570.Date, dfRX570.price, 'k-', label='RX570 Price [$]')
    plt.plot(dfRX570.Date, dfRX570.eth, 'b--', label='ETH Price [$]')
    plt.xlabel('Date')
    plt.ylabel('USD [$]')
    plt.grid()
    plt.legend(loc='best')

    # Second Create a forecast model for eth price
    import statsmodels.tsa.holtwinters as ems
    fit1=ems.ExponentialSmoothing(dfPrice.Value).fit()
    fit1.predict(start=1059,end=1324)
    dfForecast=pd.read_csv('inputs/ETHpriceForecastJune.txt',delim_whitespace=True)
    dfForecast['date']=pd.date_range('6/25/2018',periods=len(dfForecast))
    #Set lowest value to zero
    dfForecast.Flow.clip(lower=0.0)
    plt.figure()
    plt.plot(dfPrice['Date'][500::], dfPrice.Value[500::], '-k',label='Ethereum Price')
    plt.plot(dfForecast.date, dfForecast.F, 'k--', label='forecast')
    plt.plot(dfForecast.date, dfForecast.Flow.clip(lower=0.0), 'k.', label='90% Confidence Interval')
    plt.plot(dfForecast.date, dfForecast.Fhigh.values, 'k.', label=None)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('USD [$]')
    plt.savefig('plots/forecastPrice.png',dpi=600 )
    
    # Third create a forecast for the sigma model
    dfHR=pd.read_csv('inputs/sigmaModel.csv')
    dfHR['date']=pd.date_range('7/30/2015',periods=len(dfHR))

    plt.figure()
    plt.plot(dfHR.date,dfHR.TRUE,'k-',label='Network Hashrate')
    plt.plot(dfHR.date[:1750],dfHR.Model1[:1750],'b--', label='Model 1')
    plt.plot(dfHR.date[:1750],dfHR.Model2[:1750],'r-.', label='Model 2')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Year')
    plt.ylabel('Network Hashrate [GH/s]')
    plt.savefig('plots/forecastHashrate.png',dpi=600 )


    import ipdb; ipdb.set_trace()


  def lagarray(Lag, df=dfRX570):
    if Lag ==0:
      return df
    #import ipdb; ipdb.set_trace()
    dfLag=pd.DataFrame()
    price = np.array(df.price[Lag::]).reshape(-1,1)
    eth   = np.array(df.eth[:-Lag]).reshape(-1,1)
    #btc   = np.array(df.btc[:-Lag]).reshape(-1,1)
    #SnP   = np.array(df.SnP[:-Lag]).reshape(-1,1)
    #MC    = np.array(df.MCap[:-Lag]).reshape(-1,1)
    #import ipdb; ipdb.set_trace()
    #dfLag = pd.DataFrame(np.concatenate((price, eth, btc, SnP),axis=1), columns=['price', 'eth', 'btc', 'SnP'])
    dfLag = pd.DataFrame(np.concatenate((price, eth),axis=1), columns=['price', 'eth'])
    return dfLag
  
  # OLS regression function 
  def OLS(x, *kwargs):
    # Create lag array  
    lag = int(x[0])
    dfLag = lagarray(lag)
    # Create model
    model = sm.ols(formula=  ' price ~ eth' ,data=dfLag)
    # Get results
    results = model.fit()
    #print results.summary()
    return results

  # OLS regression function to optimize the lag
  def OLSphi(x, *kwargs):
    # Create lag array  
    lag = int(x[0])
    dfLag = lagarray(lag)
    # Create model
    model = sm.ols(formula=  ' price ~ eth' ,data=dfLag)
    # Get results
    results = model.fit()
    #print results.summary()
    # want max r2 so 1 minus results
    phi = 1. - results.rsquared
    return phi
  
  # function to optimize a lag array
  def optGPUlag(dfRX570):
    from pyswarm import pso
    lb = [0]
    ub = [100]
    args=dfRX570
    xopt, fopt = pso(OLSphi, lb, ub, swarmsize=20, minstep=1, minfunc=1e-5, debug=True)
    
    print ("-------------------------------")
    print ("   Particle Swarm Results      ")
    print ("-------------------------------")
    
    print ("Lag = " + str(xopt[0]))
    print ("-------------------------------")
    print ("phi = " + str(fopt) )
  
  
    # Run optimized model
    dfLag = lagarray(int(xopt[0]), dfRX570)
    # Create model
    model = sm.ols(formula=  ' price ~ eth' ,data=dfLag)
    # Get results
    results = model.fit()
    print(results.summary())
    f = open('TextResults/optimizedSummary.txt', 'w')
    f.write( str(results.summary())    )
    f.close()
    #print results.rsquare
    return results 
  
  # Find a linear regression model
  if lag==None:
    model = optGPUlag(dfRX570)
  else:
    model = OLS(lag)
  return model 
