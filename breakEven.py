import json
import requests
import numpy as np
from scipy.stats import binom
import pandas as pd
#%pylab inline
import matplotlib.pyplot as plt
from operator import itemgetter
from RX570model import rx570model
from miningFunctions import initializeVars, userBlockTime, buyRigs, fillPorts

GPU_0 = 18
blockReward = 3.
blockTime  = 15.
HR_GPU= 25.
HRnetwork = 285376.33 *10**3  # [MHs]
powerPerGPU = 0.160 #[kW]
powerCost = 0.14 / 3600. #[$/(kW *s)]
#===========================================================================
# Calculate lambda
def lam(exchangeRate=500, avgCostGPU=3000./6 ):
  revenue     =  (  (  blockReward * HR_GPU    * exchangeRate  )
	           / (   blockTime * HRnetwork    ) )
  expense =  (  powerCost * powerPerGPU              ) 
  lambd = (revenue - expense ) / avgCostGPU
  print('rev:     ' , revenue)
  print('expense: ' , expense)
  print('rev/exp  ' , revenue/expense)
  print lambd
  #import ipdb; ipdb.set_trace()
  return lambd, revenue, expense


# Assum price & HRnet constant; then seperable ode
def totalGPU(t):
  GPU =  GPU_0 * np.exp( lambd * t )
  return GPU


## zero growth model
def cZeroGrowth():
  #delta CoinPrice = CzeroGrowth* delta HRnet
  cZeroGrowth = ( blockTime * powerCost * powerPerGPU ) / ( blockReward * HR_GPU ) 
  return cZeroGrowth
#===========================================================================

# constant HR and Prixe model
t=np.linspace(0,365*24*3600)
lambd, rev, exp = lam()
#plt.plot(t/(3600.*24. ), totalGPU(t)); #plt.show()


# Zero Growth Model
coinPrice = 319.37
cZero = cZeroGrowth()
print('Zero Growth Constant: ', cZero)
deltaHRnet =  (np.linspace(-0.5,0.5,20)+1)*HRnetwork
plt.figure()
plt.plot(deltaHRnet/HRnetwork, cZero*deltaHRnet);
plt.xlabel('Ethereum Network Hashrate Growth')
plt.ylabel('Breakeven Coin Price [$/coin]')
plt.grid()
#plt.show()


print('Breakeven Coin Price: ', cZero*HRnetwork)
print('Breakeven Network Hashrate ', coinPrice/cZero)


# Look at historical Data, Import networkhashrate from etherscan (2018-04-29)
dfHashrate=pd.read_csv('inputs/export-NetworkHash.csv')
dfPrice=pd.read_csv('inputs/export-EtherPrice.csv')

dfPrice['Date'] = pd.to_datetime(dfPrice['Date(UTC)'])
dfHashrate['Date'] = pd.to_datetime(dfHashrate['Date(UTC)'])

dfHashrate['breakEvenPrice'] = dfHashrate.Value*10**3 * cZero # Convert Hasrate to [MHs]
dfPrice['breakEvenHash'] = dfPrice.Value / cZero


plt.figure()
plt.plot(dfPrice.Date, dfPrice.Value,'k-', label='Price')
plt.plot(dfHashrate.Date, dfHashrate.breakEvenPrice, 'b--', label='Break Even Price (BEP)')

delt = dfPrice.Value - dfHashrate.breakEvenPrice
plt.plot(dfPrice.Date, delt, 'r-.',label='Price - BEP')
plt.legend()
plt.grid()

plt.xlabel('Timestamp [s]')
plt.ylabel('USD [$]')
plt.title('ETH Price & B/E Price through 2018-06-28')
plt.savefig('BEanalysis.pdf')

#plt.figure()
#plt.plot(dfHashrate.UnixTimeStamp, dfHashrate.Value*10**3, 'k-', label='Hashrate')
#plt.plot(dfPrice.UnixTimeStamp, dfPrice.breakEvenHash, 'b--', label='Break-even Hashrate')
#plt.legend()

#plt.show()


import ipdb; ipdb.set_trace()

