import matplotlib.pyplot as plt
from operator import itemgetter
import json
import requests
import numpy as np
import pandas as pd


def initializeVars(weeks, parameters):
    '''
    Intitalized Vars for different studies
    
    Parameters
    ----------
    weeks: int
        number of weeks to intitalize
    parameters: Dictionary
        
    Returns
    -------
    Parameters: Dictionary
        Initialized parameters
    '''

    # Percent difficulty change in eth network per month
    parameters['blockTime']         = 15 
    parameters['networkDifficulty'] = 12.93*10**12
    parameters['networkHashrate']   = 288133782*10**6   #[Hs] 
    
    parameters['coinPerBlock']  = 3
    # Consider Difficulty increase
    parameters['hashPerGPU'] = 27.*10**6 #[H/s] ETHEREUM

    # Define values from the network and coin specific stats
    parameters['coins'] = np.zeros(weeks)
    parameters['usd']    = np.zeros(weeks)
    parameters['userHashrate'] = parameters['totalGPU0'] * parameters['hashPerGPU'] 
    parameters['daysToFindBlock'] = parameters['networkHashrate'] / parameters['userHashrate']                                     * parameters['blockTime'] / (3600. * 24.)
    parameters['coinPerDay'] = parameters['coinPerBlock'] / parameters['daysToFindBlock']

    # Describe how the network HR increases in time
    networkHashrateTime = np.zeros(weeks)
    #weekToIcreaseDifficulty = range(0,weeks,4) # increase every month    
    weekToIcreaseDifficulty = range(0,weeks,1) # increase every week    
    networkHashrateTime[0:4] = parameters['networkHashrate'] # Initialize

    # Set the networkHR to increase at % weekly 
    for i in weekToIcreaseDifficulty[1:]:
        networkHashrateTime[i:i+4] = ( networkHashrateTime[i-1] 
                                       * parameters['deltaDifficulty'] )
    parameters['networkHashrateTime'] = networkHashrateTime

    return parameters


def GPUprice(dfETHPrice, GPUmodel):
    '''
    Calculates GPU price based on price of ETH
    
    Parameters
    ----------
    dfETHPrice: DataFrame
        DataFrame of Prices
    GPUmodel: 
        linear regression model of GPU price wrt ETH price
    Returns
    -------
    dfGPUPrice: DataFrame
        DataFrame with GPU Price
    '''        
    idx = range(len(dfETHprice))
    # Ethereum low, med, high prices 
    low = pd.DataFrame({'eth': dfETHprice.F }, index=idx ) 
    mid = pd.DataFrame({'eth': dfETHprice.Flow }, index=idx ) 
    high= pd.DataFrame({'eth': dfETHprice.Fhigh }, index=idx )
    # GPU low, med, high prices 
    dfETHPrice['GPUlow'] = GPUmodel.predict(low)
    dfETHPrice['GPUmid'] = GPUmodel.predict(mid)
    dfETHPrice['GPUhigh'] = GPUmodel.predict(high)
    return dfGPUPrice


def userBlockTime(userHashrate, networkHashrate, blockTime):
    '''
    Update network hashrate
    
    Parameters
    ----------
    userHashrate: float
        Total Local hashrate
    networkHashrate: float
        Total network hashrate
    blockTime: float
        Time per block (average)

    Returns
    -------
    daysToFindBlock: Float
        Time in days to find a block   
    '''
    daysToFindBlock = networkHashrate / userHashrate * blockTime / (3600. * 24.)
    return daysToFindBlock


def fillPorts(bank, freePciePorts, totalGPU, priceGPU):
    '''
    Fill free Pcie Ports
    
    Parameters
    ----------
    bank: float
        Total available fund to buy components
    freePciePorts: int
        Total available GPU ports to fill
    totalGPU: int
        Total number of GPUs
        
    Returns
    -------
    bank: float
        Amount in bank after buying components
    freePciePorts: int
        Number of PCIE Ports Available
    totalGPU: int
        Total number of GPUs
    '''
        
    if freePciePorts == 0:
        return bank, freePciePorts, totalGPU
        
    nGPUs = int( bank /  priceGPU )
    if (priceGPU > bank):
        # Do nothing
        return bank, freePciePorts, totalGPU    
    # fill all ports on rig
    elif ( nGPUs >= freePciePorts ):
        bank -= freePciePorts * priceGPU
        totalGPU += freePciePorts
        freePciePorts = 0
    # fill as many PCIE ports as possible
    else:
        bank -= nGPUs * priceGPU
        totalGPU += nGPUs
        freePciePorts -= nGPUs
        
    return bank, freePciePorts, totalGPU


def buyRigs(bank, freePciePorts, totalGPU, totalRigs, params):
    '''
    Purchase a rigs
    
    Parameters
    ----------
    bank: float
        Available Funds
    freePciePorts: int
        Number of available PCIE ports on current rig
    totalGPU: int
        Total number of GPUs
    totalRigs: int
        Total number of Rigs
    params: Dict
        case parameters
    
    Returns
    -------
    bank: float
        Updated funds available
    freePciePorts: int
        Updated number of PCIER ports available
    totalGPU: int
        Updated Total number of GPUs
    totalRigs: int
        Total number of rigs    
    '''

    # price of rig with one GPU
    pGPU = params['pGPU']
    pRigFull   = params['pRigFull']
    nRigsFull = int( bank / pRigFull )   
    
    # update full rigs 
    if nRigsFull > 1.0:
        bank -=  ( nRigsFull * pRigFull ) 
        totalGPU  += ( nRigsFull * params['gpuPerRig']) 
        totalRigs += nRigsFull        
    # Starter rig is rig with only one GPU
    pRigStart = params['pRigStart']
    nRigsStart = int( bank / pRigStart )    
    if nRigsStart >= 1.0:
        bank -= pRigStart
        totalGPU  += 1
        totalRigs += 1
        freePciePorts = params['gpuPerRig'] - 1
        # Now fill remaning ports
        bank, freePciePorts, totalGPU = fillPorts(bank, freePciePorts, 
                                                  totalGPU, params['pGPU'])
    # return updated quantities
    return bank, freePciePorts, totalGPU, totalRigs    
    
    
def getLiveEth():
    '''
    Not Implemented but should work
    '''
    # Grab current ethereum stats
    r = requests.get('https://etherchain.org/api/basic_stats').json()
    # https://etherscan.io/ether-mining-calculator
    parameters['usdPerCoin'] = r['currentStats']['price_usd']
    print( 'Ehtereum Exchange Rate: ', parameters['usdPerCoin'])
    # Percent difficulty change in eth network per month
    parameters['blockTime']         = r['currentStats']['block_time'] # [s]
    parameters['networkDifficulty'] = r['currentStats']['difficulty']
    parameters['networkHashrate']   = r['currentStats']['hashrate']
    return None
    
