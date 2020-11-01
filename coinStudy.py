from miningFunctions import initializeVars, userBlockTime, buyRigs, fillPorts
#from RX570model import rx570model
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.stats import binom
import pandas as pd
import numpy as np
import requests
import json

#Plotname

#name='dynamicPriceLow-ConstHR'
#name='dynamicPriceMid-ConstHR'
#name='dynamicPriceHigh-ConstHR'

#name='dynamicPriceMid-HRmin'
#descript='Predicted Price at 1.5% Weekly Hash Rate Growth '
#name='dynamicPriceMid-HRmean'
#descript='Predicted Price at 5.3% Weekly Hash Rate Growth '
#name='dynamicPriceHigh-HRmin'
#descript='Upper 90% CI Price at 1.5% Weekly Hash Rate Growth '
#name='dynamicPriceHigh-HRmean'
#descript='Upper 90% CI Price at 5.3% Weekly Hash Rate Growth '
name='none'
descript='none'

# First define a starting point of total GPUs
totalGPU0 = 24  # time zero GPU
#totalGPU0 = 500  # time zero GPU
totalGPU = totalGPU0

bank0 = 372 

# Use ETH price forcasting model?
dynamicPrice=False 
#dynamicPrice=True
if dynamicPrice:
  priceType = 'forecast'
  #priceType = 'montecarlo'
  if priceType == 'montecarlo':
    priceLow    = 175.0
    priceMed    = 500.0
    priceHigh   = 4000.0
    #mcKey set current mcSimulation to run
    mcKey='pMed'
    #mcKey='pLow'
    mcKey='pHigh'

dynamicGPU=False  
#dynamicGPU=True  

# current design 6 GPU rig
gpuPerRig = 6
nPSU  = 2
pMobo = 90.
pRAM  = 60.
pCPU  = 50.
pOS   = 40. # ethOS and SSD
pPSU  = 130.
pGPU  = 200. 
pCase = 20.
pRiser= 27.
# Flll GPU per Rig
pRigFull = ( pMobo + pRAM + pCPU + pOS + pPSU*nPSU + pGPU * gpuPerRig 
             + pCase + pRiser )
# only one GPU	   
pRigStart = ( pMobo + pRAM + pCPU + pOS + pPSU*nPSU + pGPU  + pCase + pRiser )

# Cost of Power (Residential)
pPowerRes =0.10 # [$/(kWH)]

# Calculate price of the rig by summing the vector
gpuPerRig  = 6

# Consider the Power pull of compnents and GPU
powerRig = 100  # [W]
powerGPU = 160  # [#]
# makes a total of 1060 W per 6 GPU rig

parameters  = { 'pMobo' : pMobo , 
                'pRAM'  : pRAM  , 
                'pCPU'  : pCPU  , 
                'pOS'   : pOS   , 
                'pPSU'  : pPSU  , 
                'pGPU'  : pGPU  ,
                'gpuPerRig' : gpuPerRig, 
                'pCase' : pCase ,
                'pRiser': pRiser,
                'pRigFull': pRigFull,
                'pRigStart':pRigStart,
                'pPowerRes': pPowerRes,
                'powerRig' : powerRig,
                'powerGPU' : powerGPU,
                'totalGPU0': totalGPU0,
                'totalGPU' : totalGPU,
	          }
print(f'Total Mining Hardware Cost = ${parameters["pRigFull"]}')   
print(f'Buy a new Rig Cost = ${parameters["pRigStart"]}')    


#=======================================================================
# Here we will predict eth mined with no reinvestment
#=======================================================================
weeks=52 # TimePeiod of intersest in weeks of interest (Assume weekly pay)
# Initialize variables
params = initializeVars(weeks, parameters)
coins0 = params['coins']
usd = params['usd']

#=======================================================================
# Start Hashing with no reinvestment and difficulty adjustment
#=======================================================================
for t in range(1,weeks):
    coinPerDay = params['coinPerBlock'] / userBlockTime(params['userHashrate'], 
                                              params['networkHashrateTime'][t],
                                              params['blockTime']	)
    coins0[t] = coins0[t-1] + params['coinPerDay'] * 7
    usd[t]  = coins0[t]  * params['usdPerCoin']

#=======================================================================
# Create DataFrame for Reinvestment results 
#=======================================================================
wks = np.array(range(weeks))
dfReinvest = pd.DataFrame(wks, columns=['week'])
columns = ['coinTotal', 'coin','coinPrice', 'coinPerDay', 
           'bank_0', 'bank_f', 'totalRigs', 'totalGPU','freePciePorts']
for col in columns:
   dfReinvest[col] = np.zeros(weeks)
#=======================================================================
# Consider the number of GPUs per rig and the current total number of GPUs
#=======================================================================
totalGPU = totalGPU0 
totalRigs = np.ceil(float(totalGPU0)/gpuPerRig)

# Calcule if there are currently free pcie ports given initial setup
freePciePorts = totalRigs * params['gpuPerRig'] - totalGPU

# Initialize create a profits bank
bank = bank0 
dfReinvest['bank_0'] = bank0 

# Initialize df
dfReinvest.loc[0, 'coinPerDay'] = ( params['coinPerBlock']
                                  / userBlockTime(params['userHashrate'], 
                                                  params['networkHashrateTime'][0],
                                                  params['blockTime']))
dfReinvest.loc[0,'totalGPU'] = totalGPU
dfReinvest.loc[0,'totalRigs'] = totalRigs
dfReinvest.loc[0,'freePciePorts'] = freePciePorts

if dynamicPrice:
  # Function to add a predicted GPU price based on model
  def GPUprice(dfETHPrice, GPUmodel):
    idx = range(len(dfETHprice))
    # Ethereum low, med, high prices 
    low = pd.DataFrame({'eth': dfETHprice.F }, index=idx ) 
    mid = pd.DataFrame({'eth': dfETHprice.Flow }, index=idx ) 
    high= pd.DataFrame({'eth': dfETHprice.Fhigh }, index=idx )
    # GPU low, med, high prices 
    dfETHPrice['GPUlow'] = GPUmodel.predict(low)
    dfETHPrice['GPUmid'] = GPUmodel.predict(mid)
    dfETHPrice['GPUhigh'] = GPUmodel.predict(high)
    return dfETHPrice

  if priceType == 'forecast':
    # Load Forcasted price from file	
    #dfETHprice = pd.read_csv('inputs/ETHpriceForecast.txt', delim_whitespace=True)
    dfETHprice = pd.read_csv('inputs/ETHpriceForecastJune.txt', delim_whitespace=True)
    # Convert ETHPRICE from days to weeks
    dfETHprice = dfETHprice[::7].reset_index()
    # To optimize the lag time and return model
    #GPUmodel = rx570model()
    # If lag time is known
    GPUmodel = rx570model(lag=[18.19600067])
    # Add GPU prices to dataFrame
    dfETHprice = GPUprice(dfETHprice, GPUmodel)
  if priceType == 'montecarlo':
    pLow  = np.linspace(priceMed,priceLow, weeks)
    dfETHprice =  pd.DataFrame({'pLow': pLow}, index=range(len(pLow)))
    dfETHprice['pMed']  = np.linspace(priceMed,priceMed, weeks)
    dfETHprice['pHigh'] = np.linspace(priceMed,priceHigh,weeks)

#===========================================================================
# Start hashing
#===========================================================================
for t in range(1,weeks):
    #print(f'Week: {t}')
    userHashrate = totalGPU * params['hashPerGPU'] # [H/s]
    coinPerDay = (params['coinPerBlock']
                  / userBlockTime(userHashrate, 
                                  params['networkHashrateTime'][t],
                                  params['blockTime']))
    dfReinvest.loc[t,'coinPerDay'] = coinPerDay
    dfReinvest.loc[t,'coin'] = coinPerDay * 7
    dfReinvest.loc[t,'coinTotal'] = dfReinvest.coinTotal[t-1] + coinPerDay * 7
    if dynamicPrice == False:
      usd[t] = dfReinvest.coin[t]  * params['usdPerCoin']
      # Save exchange rate
      dfReinvest.loc[t,'coinPrice'] = params['usdPerCoin']
    elif dynamicPrice == True:
      if priceType == 'forecast':	    
        # Save exchange rate
        dfReinvest.loc[t,'coinPrice'] = dfETHprice.F[t]
        #dfReinvest.loc[t,'coinPrice'] = dfETHprice.Flow[t]
        #dfReinvest.loc[t,'coinPrice'] = dfETHprice.Fhigh[t]
        # Calculate weekly earning
        usd[t] = dfReinvest.coin[t]  * dfReinvest.coinPrice[t] 
	#import ipdb;ipdb.set_trace()
	
      elif priceType == 'montecarlo':	    
        usd[t] = dfReinvest.coin[t]  * dfETHprice[mcKey][t]
        # Save exchange rate
        dfReinvest.coinPrice[t] = dfETHprice[mcKey][t]
        # if dynamic GPU price on update value
      if dynamicGPU==True:
        #name='highDynamicETHnGPU'
        parameters['pGPU'] = dfETHprice.GPUhigh[t]
      
    # Add USD to bank
    bank += usd[t]
            
    # subtract Power costs
    totalPower = ((totalRigs * params['powerRig']) 
                   + (totalGPU * params['powerGPU'])) # [W]
                   
    # Power cost per week ( kW/hr * W/1000  )
    powerCost = (parameters['pPowerRes'] * totalPower/1000 * 7 * 24)
    bank -= powerCost    
    # Save bank before buying new componets each week
    dfReinvest.loc[t,'bank_0'] = bank    
    # immediatly fill free GPU spots
    bank, freePciePorts, totalGPU = fillPorts(bank, freePciePorts, 
                                              totalGPU, parameters['pGPU'])
    # now buy as many rigs as possible
    bank, freePciePorts, totalGPU, totalRigs = buyRigs(bank, freePciePorts, 
                                                       totalGPU, totalRigs, 
                                                       params)
    
    # Save weekly GPU and Rig after reinvestment
    dfReinvest.loc[t,'totalGPU'] = totalGPU
    dfReinvest.loc[t,'totalRigs'] = totalRigs
    dfReinvest.loc[t,'freePciePorts'] = freePciePorts
    # Save bank after buying new componets each week
    dfReinvest.loc[t,'bank_f'] = bank
    #import ipdb; ipdb.set_trace()

# ----------------------------------
# Plot the Results
# ----------------------------------        
intval = 2 # Interval of weeks to plot
fig3,[[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2,figsize=(10,10))

ax1.plot(dfReinvest.week[::intval], dfReinvest.coinTotal[::intval], 'o--', color='k', label='Reinvestment')
ax1.plot(dfReinvest.week[::intval], coins0[::intval], 'x:', color='k', label='No Reinvestment')
ax1.set_xlabel('Week')
ax1.set_ylabel('Coin')
ax1.legend(loc='best')
ax1.grid()

# Add USD ticks
# Add some extra space for the second axis at the bottom
fig3.subplots_adjust(bottom=-0.2)
ax1b= ax1.twiny()
# Move twinned axis ticks and label from top to bottom
ax1b.xaxis.set_ticks_position("bottom")
ax1b.xaxis.set_label_position("bottom")
# Offset the twin axis below the host
ax1b.spines["bottom"].set_position(("axes", -0.15))
ax1b.plot(dfReinvest.coinPrice[1::intval],dfReinvest.coinTotal[1::intval],' ')
# Turn on the frame for the twin axis, but then hide all 
# but the bottom spine
ax1b.set_frame_on(True)
ax1b.patch.set_visible(False)
#for sp in ax1b.spines.itervalues():
#    sp.set_visible(False)
#ax1b.spines["bottom"].set_visible(True)
#ax1b.set_xlabel('USD [$]')
#ax1b.set_title('Total ' + str(parameters['coin'].upper()) + ' Generated')

#Second plot shows growth in CoinhPerDay
ax2.plot(dfReinvest.week[::intval], dfReinvest.coinPerDay[::intval]*7,'o--',color='k')
ax2.set_xlabel('Week')
ax2.set_ylabel('Coin / Week')
ax2.grid()

# Add USD ticks
ax2b= ax2.twiny()
# Move twinned axis ticks and label from top to bottom
ax2b.xaxis.set_ticks_position("bottom")
ax2b.xaxis.set_label_position("bottom")
# Offset the twin axis below the host
ax2b.spines["bottom"].set_position(("axes", -0.15))
ax2b.plot(dfReinvest.coinPrice[1::intval],dfReinvest.coinPerDay[1::intval]*7, ' ')
# Turn on the frame for the twin axis, but then hide all 
# but the bottom spine
ax2b.set_frame_on(True)
ax2b.patch.set_visible(False)
#for sp in ax2b.spines.itervalues():
#    sp.set_visible(False)
#ax2b.spines["bottom"].set_visible(True)
#ax2b.set_xlabel('USD [$]')

ax2b.set_title('Coin Generation Rate With Reinvestment')

#Third plot shows weekly bank USD before buying components
ax3.plot(dfReinvest.week[::intval], dfReinvest.bank_0[::intval],'o--',color='k')
ax3.set_xlabel('Week')
ax3.set_ylabel('Bank Before Reinvestment [USD]')
ax3.grid()
ax3.set_title('Bank')
# Add second Axis
#ax3b = ax3.twinx()
#ax3b.plot(dfReinvest.week[::intval], dfReinvest.bank_f[::intval],'x:',color='k')
#ax3b.set_ylabel('Bank After Reinvestment [USD]')
#ax3.legend(loc='best')
#ax3b.legend(loc='lower right')


#Fourth plot shows weeklygrowth of GPUs and Rigs
ax4.plot(dfReinvest.week[::intval], dfReinvest.totalGPU[::intval],'o--',color='k', label='GPU')
ax4.set_xlabel('Week')
ax4.set_ylabel('Total GPU')
ax4.grid()
# Add second y-axis ticks
ax4b = ax4.twinx()
ax4b.plot(dfReinvest.week[::intval], dfReinvest.totalRigs[::intval],'x--',color='k', label='Rig')
ax4b.plot(dfReinvest.week[::intval], dfReinvest.freePciePorts[::intval],'.',color='k', label='Free P-CIE')
ax4b.set_ylabel('Total Rigs/ Free P-CIE Ports')
ax4.set_title('Hardware Stats after Reinvestment')
#ax4.legend(loc='best')
ax4.legend(loc='center left')
ax4b.legend(loc='center right')
#ax4b.legend(loc='best')

plt.tight_layout()
plt.subplots_adjust(top=0.91)
plt.suptitle(descript)

#plt.savefig('plots/'+coin+'Reinvest.pdf')
#plt.savefig('plots/'+name+'.pdf')
plt.savefig('plots/'+name+'.png',dpi=600)
#plt.show()   

print( dfReinvest)
plt.show()

import ipdb; ipdb.set_trace()
#===========================================================================================
#===========================================================================================

# =============================================================================
# #calculate weekly hashrate increase
# deltaDiff = parameters['deltaDifficulty']
# weekDeltaDiff = deltaDiff**.25 
# # To maintain hasing power miner must incrase at equal rate
# equilibriumHash = np.zeros(weeks)
# equilibriumHash[0] = parameters['userHashrate']
# for i in range(1,weeks):
#   equilibriumHash[i] = equilibriumHash[i-1] * weekDeltaDiff
# 
# 
# # add network HashrateTime to df
# dfReinvest['networkHashrate'] = parameters['networkHashrateTime']
# 
# #Import networkhashrate from etherscan (2018-06-26)
# dfHashrate=pd.read_csv('inputs/export-NetworkHash.csv')
# dfPrice=pd.read_csv('inputs/export-EtherPrice.csv')
# 
# 
# # View network increase
# dfPrice['7dAvg']=dfPrice.rolling(7).mean().Value
# dfHashrate['7dAvg']=dfHashrate.rolling(7).mean().Value
# dfHashrate['50dAvg']=dfHashrate.rolling(50).mean().Value
# dfHashrate['200dAvg']=dfHashrate.rolling(200).mean().Value
# 
# # Convert UNix timestamp to date
# dfPrice['Date'] = pd.to_datetime(dfPrice['Date(UTC)'])
# dfHashrate['Date'] = pd.to_datetime(dfHashrate['Date(UTC)'])
# 
# plt.figure()
# plt.plot(dfHashrate.Date, dfHashrate.Value/ dfHashrate.Value.max(),'--', color='k', label='Network Hashrate'); 
# 
# plt.plot(dfPrice.Date, dfPrice.Value/dfPrice.Value.max(), ':', color='b', label='Price'); 
# plt.grid()
# plt.xlabel('Date')
# plt.ylabel('Normailized Value')
# plt.legend()
# plt.savefig('plots/normalizedPrice2HR.png',dpi=600)
# 
# 
# 
# 
# plt.figure()
# plt.plot(dfHashrate.UnixTimeStamp, dfHashrate['7dAvg'],'-', color='k', label='7 Day'); 
# plt.plot(dfHashrate.UnixTimeStamp, dfHashrate['50dAvg'],'--', color='b', label='50 Day'); 
# plt.plot(dfHashrate.UnixTimeStamp, dfHashrate['200dAvg'],':', color='c', label='200 Day'); 
# 
# plt.grid()
# plt.xlabel('Unix Timestamp [s]')
# plt.ylabel('Network Hashrate Moving Average')
# plt.legend()
# plt.tight_layout()
# 
# #np.diff(dfHashrate['200dAvg'].tail(20))
# 
# plt.figure()
# # Use 200 day moving avg to calc weekly %change in HR
# pcntChg =  dfHashrate['200dAvg'].pct_change(periods=7).dropna()[20:]*100
# plt.plot(dfHashrate.Date[226:], dfHashrate['200dAvg'].pct_change(periods=7).dropna()[20:]*100 , 'k-')
# 
# plt.grid()
# plt.xlabel('Date')
# plt.ylabel('Weekly Change in Network Hashrate [%]')
# #plt.legend()
# plt.tight_layout()
# 
# import ipdb; ipdb.set_trace()
# plt.show()
# 
# =============================================================================
