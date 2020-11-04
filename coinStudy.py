from miningFunctions import initializeVars, userBlockTime, buyRigs, fillPorts
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.stats import binom
import pandas as pd
import numpy as np
import requests
import json
import os

if not os.path.exists('plots'): os.makedirs('plots')


# TimePeiod of intersest in weeks of interest (Assume weekly pay)
weeks=52 
usdPerCoin = 385.0 
# Case 1: Fixed Price, Constant Network
case1 = { 'name': 'fixedPrice-ConstantHR',
          'descript': 'Fixed Price, Constant Network',
          'deltaDifficulty' : 1.000,
          'ethPrice' : [usdPerCoin]*weeks          
        }

# Load and process historical price and forecast model
ethPriceHistoricalAndForecast = pd.read_csv('data/forecastEthPrice.csv')
ethPriceHistoricalAndForecast['Date'] =pd.to_datetime(
                                         ethPriceHistoricalAndForecast.UnixTimeStamp, 
                                         origin='unix',unit='s')
ethPriceHistoricalAndForecast.set_index('Date', inplace=True)                                         
del ethPriceHistoricalAndForecast['UnixTimeStamp']
forecastFalseMask = ethPriceHistoricalAndForecast.forecast.isna()
ethPriceHistorical = ethPriceHistoricalAndForecast[forecastFalseMask].dropna(axis=1)
#ethPriceHistorical = ethPriceHistorical[::7]
ethPriceForecast = ethPriceHistoricalAndForecast[~forecastFalseMask].dropna(axis=1)
ethPriceForecast.forecastLow.clip(lower=0, inplace=True)
ethPriceForecast = ethPriceForecast[::7]
del ethPriceHistoricalAndForecast

# Case 2.a: Dynamic Price (moderate gain), constant network
case2a = {'name': 'dynamicPriceMid-ConstHR',
          'descript': 'Dynamic Price (moderate gain), constant network',
          'deltaDifficulty' : 1.000,
          'ethPrice': ethPriceForecast.forecast         
         }
# Case 2.b: Dynamic Price (Low), constant network
case2b = {'name': 'dynamicPriceLow-ConstHR',
          'descript' : 'Dynamic Price (Low), Constant Network',
          'deltaDifficulty' : 1.000,
          'ethPrice': ethPriceForecast.forecastLow,         
         }
# Case 2.c: Dynamic Price(Heroic Gains), constant Network
case2c = {'name' :'dynamicPriceHigh-ConstHR',
          'descript': 'Dynamic Price(Heroic Gains), Constant Network',
          'deltaDifficulty' : 1.000,
          'ethPrice': ethPriceForecast.forecastHigh
         }

networkHash = pd.read_csv('data/networkHash.csv')
networkHash['Date'] = pd.to_datetime(networkHash.UnixTimeStamp, 
                                     origin='unix',unit='s')
networkHash.set_index('Date', inplace=True) 
del networkHash['UnixTimeStamp']

networkHash['200DAvg'] = networkHash.rolling('200d').mean()
deltaDifficultyMin = 1+networkHash['200DAvg'].pct_change(periods=7).dropna()[50:].min()
deltaDifficultyMean = 1+networkHash['200DAvg'].pct_change(periods=7).dropna()[50:].mean()
#import ipdb; ipdb.set_trace()
# Overwrite/ hardcode to persios study % increace
deltaDifficultyMin = 1.015
deltaDifficultyMean =1.053

# Case 3.a: Dynamic Price(Moderate), Dynamic Netowork (minimum)
case3a = {'name':'dynamicPriceMid-HRmin',
          'descript':'Predicted Price at 1.5% Weekly Hash Rate Growth',
          'deltaDifficulty' : deltaDifficultyMin,
          'ethPrice': ethPriceForecast.forecast
         }
# Case 3.b: Dynamic Price(Moderate), Dynamic Netowork (mean)
case3b = {'name':'dynamicPriceMid-HRmean',
          'descript':'Predicted Price at 5.3% Weekly Hash Rate Growth',
          'deltaDifficulty' : deltaDifficultyMean,          
          'ethPrice': ethPriceForecast.forecast
         }
# Case 3.c: Dynamic Price(Heroic), Dynamic Netowork (min)
case3c = { 'name':'dynamicPriceHigh-HRmin',
           'descript':'Upper 90% CI Price at 1.5% Weekly Hash Rate Growth',
           'deltaDifficulty' : deltaDifficultyMin,     
           'ethPrice': ethPriceForecast.forecastHigh
         }
# Case 3.d: Dynamic Price(Heroic), Dynamic Netowork (mean) 
case3d = {'name' : 'dynamicPriceHigh-HRmean',
          'descript' :'Upper 90% CI Price at 5.3% Weekly Hash Rate Growth',
          'deltaDifficulty' : deltaDifficultyMean,    
          'ethPrice': ethPriceForecast.forecastHigh
          }

cases = [case1, 
         case2a, case2b, case2c, 
         case3a, case3b, case3c, case3d]
#import ipdb;ipdb.set_trace()
for case in cases:

    # Case specific variables
    name = case['name']
    descript = case['descript']
    deltaDifficulty = case['deltaDifficulty']
    ethPrice =case['ethPrice'] 
        
    # Define a starting point of total GPUs
    totalGPU0 = 24  # time zero GPU
    #totalGPU0 = 500  # time zero GPU
    totalGPU = totalGPU0

    bank0 = 700 

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
                    'deltaDifficulty': deltaDifficulty,
                  }
    print(f'Total Mining Hardware Cost = ${parameters["pRigFull"]}')   
    print(f'Buy a new Rig Cost = ${parameters["pRigStart"]}')    


    #=======================================================================
    # Here we will predict eth mined with no reinvestment
    #=======================================================================

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
        usd[t]  = coins0[t]  * ethPrice[t]

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
        
        # Save exchange rate
        dfReinvest.loc[t,'coinPrice'] = ethPrice[t]
        # Calculate weekly earning
        usd[t] = dfReinvest.coin[t]  * dfReinvest.coinPrice[t] 
        #import ipdb;ipdb.set_trace()
        
          
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


plt.figure(figsize=(10,4))
plt.plot(ethPriceHistorical/ethPriceHistorical.max(), 'b.', 
         label='Historical Price')
plt.plot(networkHash.Value/ networkHash.Value.max(), 'k--', label='Network Hashrate') 
plt.grid()
plt.legend()
plt.ylabel('Normalized Value')
plt.tight_layout()
plt.savefig('plots/networkHashAndPrice.png', dpi=600)     


plt.figure(figsize=(10,4))
# Use 200 day moving avg to calc weekly %change in HR
pcntChg =  networkHash['200DAvg'].pct_change(periods=7).dropna()*100
plt.plot(networkHash['200DAvg'].pct_change(periods=7).dropna()[50:]*100 , 'k-')
plt.grid()
plt.ylabel('Weekly Change in Network Hashrate [%]')
plt.tight_layout()
plt.savefig('plots/networkHashWeeklyChange.png', dpi=600)     


plt.figure(figsize=(10,4))
plt.plot(ethPriceHistorical, 'k-', label='Historical')
plt.plot(ethPriceForecast.forecast, 'k--', label='Forecast')
plt.plot(ethPriceForecast.forecastHigh, 'k.', label='90% Confidence Interval')
plt.plot(ethPriceForecast.forecastLow, 'k.', label= '__None')
plt.grid()
plt.legend()
plt.xlabel('Date')
plt.ylabel('USD [$]')
plt.savefig('plots/ethereumForecastModel.png', dpi=600)

plt.show()
#import ipdb; ipdb.set_trace()

#===========================================================================================
#===========================================================================================
# #calculate weekly hashrate increase
# deltaDiff = parameters['deltaDifficulty']
# weekDeltaDiff = deltaDiff**.25 
# # To maintain hasing power miner must incrase at equal rate
# equilibriumHash = np.zeros(weeks)
# equilibriumHash[0] = parameters['userHashrate']
# for i in range(1,weeks):
#   equilibriumHash[i] = equilibriumHash[i-1] * weekDeltaDiff
