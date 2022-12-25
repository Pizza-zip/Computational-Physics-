#Final Exam 300116212
#IF YOU RUN THE CODE RUN BY CELL AND READ CELL TITLES, CODE IS VERY SLOW
# Code contains 4 parts
# Random Walks and Fitting
# Importance Sampling
# DLA
# PSEUDO RANDOM NUMBER GENERATORS (PNRGs)

import numpy as np
import random 
import matplotlib.pyplot as plt
import statistics 

from scipy.optimize import curve_fit
from numpy import inf


#%% All defined functions EXCEPT 2D ising model those are in seperate cells

def randomWalk2D(M,N): #THis function returns data for a random walk in 2D
    randomWalkX = [] #Define empty lists
    randomWalkY = []
    m = 1 #Define factor m 
    while m<= M: #while loop for all M values
        xStart = np.zeros(N) #Creates Array of length N all 0.0s, this allows the array to keep up with the updating numbers in the up coming if statements
        yStart = np.zeros(N) #Could be done with ones or 8s or whatever, just some sort stand in is needed
        for z in range (0,N,1): #Simlar for loop for all 0-N steps
            randomProb = random.random() #Creates random number (works as Probability)
            if randomProb <= 0.25: #IF Values appear in quarters (Equal probability) values in array change , function STEPS
                xStart[z] = xStart[z-1]+1 #Step +1 in x
                yStart[z] = yStart[z-1] #Set y to previous value (No step)
            elif 0.25 < randomProb <= 0.50:
                xStart[z] = xStart[z-1]-1 #Step -1 in x 
                yStart[z] = yStart[z-1]
            elif 0.5 < randomProb <= 0.75:
                yStart[z] = yStart[z-1]+1 #Step +1 in y
                xStart[z] = xStart[z-1] #Set x to previous value (No step)
            else:
                yStart[z] = yStart[z-1]-1
                xStart[z] = xStart[z-1]
        randomWalkX.append(xStart) #Random Walks in x for Mth walker apppened to larger List
        randomWalkY.append(yStart) #Random Walks in y for Mth walker appended to larger List
        m += 1 
    return randomWalkX, randomWalkY #Return Values
    

def randomWalkDVectorSquare(M,N): #This function calcualtes the square mean displacment 
    nList = np.arange(0,N,1) #Create nList for 0-N
    displacementList = [] #empty List
    randomWalk2Dx , randWalk2Dy = randomWalk2D(M,N) #calcualtes M random walks of varring steps N
    tranRandX , tranRandY = np.transpose(randomWalk2Dx), np.transpose(randWalk2Dy)
    for i in range(0,N,1): #creates list for increasing N
        rDisplacement = np.sqrt((tranRandX[i])**2 + (tranRandY[i])**2)
        diffDisplacemnt = np.diff(rDisplacement) #this calculates r(x + dx) - r(x)
        diffDisplacemntsq = diffDisplacemnt**2 #calcualtes the square
        meanDisplacement = np.mean(diffDisplacemntsq) #Calcualtes the mena
        displacementList.append(meanDisplacement) #appends to list
    return displacementList , nList #Returns values


def LinearRegressionWORKING(x,y,sigma): #This function calcualtes the linear regression of data x,y and sigma 
    
    xmean = np.mean(x)
    ymean = np.mean(y)
    
    # Calculate the terms needed for the numator and denominator of beta
    Covariance = ((x - xmean) * (y - ymean))
    xVariance = ((x - xmean)**2)
    
    # Calculate beta and alpha
    beta = sum(Covariance) / sum(xVariance)
    alpha = ymean - (beta * xmean)
    
    x = np.array(x)
    yRegression = (alpha + beta * x)

     
    goodnessFit(y,yRegression,sigma) #Calls function for KI Goodness fit 
      
    return yRegression #Return value 


def LinearRegression(x,y,sigma): #This function calcualtes the linear regression of data x,y and sigma 
  
    f0,f1,f2 = 4 , 2*x,'0' #Calls getFunctions and returns up to 3 functions 
    if f0 == '0': #Checks if any functions are 0 
        F = [f1,f2]
    elif f1 == '0':
        F = [f0,f2]
    elif f2 == '0':
        F = [f0,f1]
    else:
        F = [f0,f1,f2]
    row = [] #Empty lists
    alpha = []
    beta = []
    print(F)
    
    for i in F:  #Calculates vector beta 
        betaTemp = sum((y*(i))/(sigma**2)) 
        beta.append(betaTemp) #appends each value to a larger list
    for z in F: #Calculates matrix alpha 
        row = [] #Defines empty row
        for v in F:
            alpha_mm = sum((z*(v))/(sigma**2)) #calcualtes each alpha value
            row.append(alpha_mm) #appends to list row
        alpha.append(row) #appends row to larger List
    print(alpha)
        
    inverseAlpha = np.linalg.inv(np.array(alpha)) #Calls matrix inverse function
      
    print('The Covariance Matrix is: ') #Prints covariance matrix 
    print(inverseAlpha)
    
    if len(F) == 2: #Checks if matrix is 2x2
        multiply = np.multiply(inverseAlpha,beta) #Multplies matracies 
        Am = [multiply[0][0]+multiply[1][0],multiply[0][1]+multiply[1][1]] #Calcualtes coeficients am
        yRecursion = Am[0]*f0 + Am[1]*f1 #Returns linear recursion 
    else:
        multiply = np.multiply(inverseAlpha,beta) #Multplies matracies
        Am = [multiply[0][0]+multiply[1][0]+multiply[2][0],multiply[0][1]+multiply[1][1]+multiply[2][1],multiply[0][2]+multiply[1][2]+multiply[2][2]]#Calcualtes coeficients am
        yRecursion = Am[0]*f0 + Am[1]*f1 + Am[2]*f2#Returns linear recursion 
    goodnessFit(y,yRecursion,sigma[0]) #Calls function for KI Goodness fit 
     
    return x,yRecursion #Return values 


def goodnessFit(y,yRecursion,sigma): #This functions calcualtes the goodness of Fit ki**2
    ki = ((y-yRecursion)**2)/yRecursion
    ki = np.nan_to_num(ki)
    kiFit = abs(sum(ki))
    print('The goodness of fit value is: ', kiFit) #Prints goodness Fit


def monteCarloMethod(N): #This function calcualtes the monteCarlo Integral of multidemension function xHat.
    SumP1 = [] #Define empty list and startiong variables
    SumP2 = []
    Lam = 0.81511
    
    p_1 = 1/(np.sqrt(np.pi))
    p_2 = lambda x : (np.sqrt((2*Lam)/(1-np.exp(-2*Lam*np.pi))))*np.exp(-Lam*x)
    fx = lambda x : 1/((x**2)+(np.cos(x)**2))
    
    
    for i in range(0,int(N)): #Run the integral for N counts, as specificed by the user
        xHat = random.random()*np.pi
        sumTempP1 = ((fx(xHat))/p_1)
        sumTempP2 = ((fx(xHat))/p_2(xHat))
        SumP1.append(sumTempP1)
        SumP2.append(sumTempP2)
    MonteCarloP1 = (1/N)*sum(SumP1) #Calcualte the value for the integral 
    MonteCarloP2 = (1/N)*sum(SumP2) #Calcualte the value for the integral 
    return MonteCarloP1,MonteCarloP2

def randomNumberLinear(N): #This function calcualtes a list of length N random numbers using congruent method PRNG
    a ,c ,M, r = 57, 1, 256, 10 #Define inital variables
    randomList = [r] #Start list
    
    for i in range(0,N,1): #for range N 
        rplus1 = ((a*r+c) % M) #Calcualte random vlaue
        randomList.append(rplus1) #append to list
        r = rplus1 # old random value replaced with new
        
        randomList = randomList
    randomList = np.array(randomList)/256.0 #Gives random number between 0-1
    return randomList #Return list
def calcPeriod(): #This function calcualtes the period of the PRNG previously deined
    depends = True #While boolean requirment 
    a ,c ,M, r = 57, 1, 256, 10 #Pre defined variables
    counter = 0 #Counter
    while depends: #While loop until period is found
        rplus1 = ((a*r+c) % M) #Same PRNG
        r = rplus1
        if rplus1 == 10: #Starting value, end if found
            depends = False
            return counter
        else: #else increaste the period counter
            counter += 1

def orderMomnets(N,k,M): #This function calcualtes the order moments as defined in LAB 3
    moment = []     #define empty lists
    momentList = []
    tempRandList = []
    nList = []
    for i in np.logspace(1,N,num = 9-2+1,base = 10.0): #have a loop that will increase x10 each time
        for x in range(1,M): #loop that will create list of M arrays of random numbers
            tempRand = np.array(randomNumberLinear(100)) #create array of random variables legnth 100
            tempRandList.append(tempRand) #append to larger list
        tempRandArray = np.array(tempRandList) #convert larger list to Array
        moment = (tempRandArray)**k #calcualte the moments of Larger List
        momentSum = (1/N)*sum(moment) #sum all moments and take average for each N
        
        momentList.append(np.mean(momentSum)) #append values to larger list
        nList.append(i) #append N values increasing by 10x each time
    return momentList,nList #return lists

def randomCorrelation(N,j): #Calcualtes near neighbour correlation for 2 random sets of data
    correlationList = [] #define empty lists and variables
    tempRandList1 = []
    tempRandList2 = []
    nList = []
    M = 100    
    for i in np.logspace(1,(N-j),num = 9-2+1,base = 10.0): #have list that loops for 1 -> N-j increases by x10
            for x in range(1,M): #loop for M arrays of random variables
                tempRand1 = np.array(randomNumberLinear(100)) #createss 2 lists of random numbers
                tempRand2 = np.array(randomNumberLinear(100))
                tempRandList1.append(tempRand1) #appends random number lists to larger list
                tempRandList2.append(tempRand2)
                
            tempRandArray1 = np.array(tempRandList1) #Turns larger lists into arrays
            tempRandArray2 = np.array(tempRandList2)
            corelation = (tempRandArray1)*(tempRandArray2) #calcualtes the product of bot harrays
            correlationSum = (1/(N-j))*(sum(corelation)) #calculates the near neaigbhour corelation
            
            correlationList.append(np.mean(correlationSum)) #creates larger list of correlation
            nList.append(i)#appends N values
    return correlationList, nList #returns N values and Correlation list

def avgDevation(dataList,k):  #This gunciton calcualtes the deiviation as set by Lab 3
    x = np.array(dataList)
    return abs(x-(1/(1+k)))

def lamProof(): #This function calcualtes and returns data that show sthe ideal value of lambda is 0.81511
    proofList = [] #Define some empty lists
    subList = []
    variance = []
    x = [0,(1/4)*np.pi,np.pi/2,(3/4)*np.pi,np.pi] #Define some test x
    lamTest = [0.20152,0.40312,0.81511,1.62300,3.0] #define some test lambda vlaues
    f = lambda x:  1/((x**2)+(np.cos(x)**2)) #Define 2 functions 
    p_2 = lambda Lam,x : (np.sqrt((2*Lam)/(1-np.exp(-2*Lam*np.pi))))*np.exp(-Lam*x)
    
    for i in lamTest: #for each test lambda, calcualte all values of each x and append to list
        for u in x:    
            prooftemp = f(u)/p_2(i,u)
            subList.append(prooftemp)
        proofList.append(subList)
        subList = []
    for i in proofList: #Calcualte the varriacne of each list previouslly calcualted and append to varriance list
        var = statistics.pvariance(i)
        variance.append(var)
    return variance #Return varriance list.

def inLimit(cor): #This function calculates if a given point is inside a circle of radius  and returns a boolean
    if (cor[0] ** 2 + cor[1] ** 2) > endRadius ** 2: 
        return False
    else:
        return True
    
def isStuck(cor,mass): #This function calcualtes if the current coords (cor) is within the greater radius
    global stuck, startRadius, endRadius, maxradius,newRad,massList,radiusList,densityDone
    posCheck = (cor[0] ** 2 + cor[1] ** 2) #Check if current position is inside the current max radius
    if posCheck > maxradius ** 2:
        maxradius = int(np.sqrt(posCheck))
        if maxradius != newRad: #Condition for new max radius
            newRad = maxradius #new radius is assigned
            radiusList.append(newRad) #append radius value to radius list
            massList.append(mass) #append mass value to mass list
        startRadius = maxradius + startRadius_step if (maxradius + startRadius_step) < N else N   # runs into edges
        endRadius = maxradius + endRadius_step if (maxradius + endRadius_step) < N else N
    stuck += 1

    lattice[cor[0] + N, cor[1] + N] = 255   # making this = hits  has the effect of color coding the time of arrival
def nearNeighbour(cor): #This funciton calcualtes the nearest neighbours 

    latticepos = (cor[0] + N, cor[1] + N)# convert from lattice coords to array coords
    nearNeighbourSteps = ((0, 1), (0, -1), (1, 0), (-1, 0)) #Possible neigherst neighbours 

    for step in nearNeighbourSteps: #Checks if any of these nearest neghbours are accounted for 
        if lattice[latticepos[0] + step[0], latticepos[1] + step[1]] != 0:
            return True
    else:
        return False #Returns boolean


def DLA(numParticles): #This function plots a Diffusion limited aggregation 
    nnsteps = ((0, 1), (0, -1), (1, 0), (-1, 0)) #Possible neigherst neighbours
    massTemp = 0
    for particle in range(numParticles): #Does loop for N selected particles
        
        angle = random.random() * 2 *np.pi # find a random angle  from 0 -> 2pi
        
        cor = [int(np.sin(angle) * startRadius), int(np.cos(angle) * startRadius)] #Defines some starting positon 
    
        isDead = False      # walker starts off alive
        while not isDead: #While loop for random walks 
            massTemp += 1 #increase mass by 1
            moveDir = random.choice(nnsteps) # pick one of the nearest neighbour sites to explore
            cor[0] += moveDir[0] # and apply the selected move to position coordinate, pos
            cor[1] += moveDir[1]
            
            if not inLimit(cor): #Break if the position leaves the circle
                isDead = True
                break
            elif nearNeighbour(cor): #break if poition is one of the nearest neighbours
                isStuck(cor,massTemp) #If so update birthradius and death radius and maxRadius aswell as hits 
                isDead = True
                break
    M = maxradius #Update M
    grph = N - M, N + M #Collect plotting data 
    
    
    #Plot DLA
    plt.figure(dpi = 100)
    plotrange = np.arange(-M, M + 1)
    plt.pcolormesh(plotrange, plotrange, lattice[grph[0]:grph[1], grph[0]:grph[1]], cmap='summer_r')
    plt.xlabel('Step In x')
    plt.ylabel('Step In y') 
    plt.show()
    
    return massList,radiusList

def linear(x,D,B): #this function returns a linear function 
    return D*x + B
        
#%% JUST 2D ISING SPIN FUNCTIONS FOR Better PRNG 
#If running Part Ising spin for random.random() PRNG Load this cell than run

def initalMicroState(N): #This function gives an inital random array of integers either 1,-1
    microState = 2*np.random.randint(2,size=(N,N))-1
    return microState

def metropolisMethod(microState, beta,B): #This function Calcualtes the Metroplois method of some Inital micro state 
    J = 1 #Define fixed values for equation for calculting the ising model with metroplois method
    mew = 1
    for i in range(N): #2 nested loops
        for j in range(N):
                a = np.random.randint(0, N) #Some random intger between 0,N
                b = np.random.randint(0, N)
                s =  microState[a, b] #Get some random value  from the NxN microstate
                nearNeighbours = microState[(a+1)%N,b] + microState[a,(b+1)%N] + microState[(a-1)%N,b] + microState[a,(b-1)%N] #Calcualte the nearest Neighbours 
                engDiference = 2*J*s*nearNeighbours +2*mew*B*s #Calculate the energy diffrence between s'-s
                #Conditions for next microstate
                if engDiference < 0: #First condition 
                    s *= -1
                elif np.random.rand() < np.exp(-engDiference*beta): #Second condition 
                    s *= -1
                microState[a, b] = s #otherwise new state equals previous 
    return microState #Return new microState

def magnetization(microstates): #This function Calcualtes the Magnetization 
    mag = np.sum(microstates)
    return mag

def getData(numOfTempPoints,N,equilSteps,moteCarloSteps,B): #This function collects all the data for the ising Model
    Temperature = np.linspace(0, 5, numOfTempPoints) #defines temperature array based on input temp length
    Magnetization = np.zeros(numOfTempPoints) #Define empty arrays 
    
    for x in range(numOfTempPoints): #List to collect 1 point of each requried data set
        microstates = initalMicroState(N)         # SEt intial microstates

        M1 = 0 #define 0 variables
        iT=1.0/Temperature[x]  #define inverse temperature (1/kT)
        z = 0 #stoping param
        while z < equilSteps: # To avoid more correlation from Markov
            metropolisMethod(microstates, iT,B)
            z += 1
        z = 0
        while z < moteCarloSteps: #Actual mesaurments
            metropolisMethod(microstates, iT,B) #metroplois method     
            Mag = magnetization(microstates) # calculate the magnetisation
            
            M1 += Mag
            z += 1 #end param
        #Divide by inverse steps inorder to get correct values
        Magnetization[x] = (1.0/(moteCarloSteps*(N**2)))*M1
        
    return Magnetization,Temperature #Return all values 




def magnetizationExact(N,T): #This function calcualtes the exact magneitzation as defined by the exam. T crictical was calcualted as seen
    #NOTE, function actually does not exist after Temperature critical, and therefore no limiting variables needed, goes to inifitiy.
    tempCritical = 1.64*10**23 #calcualted temperature critical
    
    j = 1 #magnetic field constant
    beta = 1/((T)) #calcculates 1/kT, however my temperature is alread kT
    mExact = N*(1-np.sinh(2*beta*j)**(-4))**(1.0/8) #Calcualtes mExact as defined in the exam
    mExact = np.nan_to_num(0)
    return mExact

#%%PART 4 WITH BAD PRNG
#If running Part ising spin for garabge PRNG load this cell and then run

def initalMicroState(N): #This function gives an inital random array of integers either 1,-1
    rnd = np.array(randomNumberLinear(N))
    rnd[rnd%2 ==1] = -1
    rnd[rnd%2 == 0] = 1
    
    microState = [rnd,rnd[::-1]]
    return microState

def metropolisMethod(microState, beta,B): #This function Calcualtes the Metroplois method of some Inital micro state 
    J = 1 #Define fixed values for equation for calculting the ising model with metroplois method
    mew = 1
    rnd = randomNumberLinear(N)
    for i in range(N): #2 nested loops
        for j in range(N):
                a = round(rnd[j]*N) #Some random intger between 0,N
                b = round(rnd[j]*N)
                s =  microState[a][b] #Get some random value  from the NxN microstate
                nearNeighbours = microState[(a+1)%N][b] + microState[a][(b+1)%N] + microState[(a-1)%N][b] + microState[a][(b-1)%N] #Calcualte the nearest Neighbours 
                engDiference = 2*J*s*nearNeighbours +2*mew*B*s #Calculate the energy diffrence between s'-s
                #Conditions for next microstate
                if engDiference < 0: #First condition 
                    s *= -1
                elif np.random.rand() < np.exp(-engDiference*beta): #Second condition 
                    s *= -1
                microState[a, b] = s #otherwise new state equals previous 
    return microState #Return new microState

def magnetization(microstates): #This function Calcualtes the Magnetization 
    mag = np.sum(microstates)
    return mag

def getData(numOfTempPoints,N,equilSteps,moteCarloSteps,B): #This function collects all the data for the ising Model
    Temperature = np.linspace(0, 5, numOfTempPoints) #defines temperature array based on input temp length
    Magnetization = np.zeros(numOfTempPoints) #Define empty arrays 
    
    for x in range(numOfTempPoints): #List to collect 1 point of each requried data set
        microstates = initalMicroState(N)         # SEt intial microstates

        M1 = 0 #define 0 variables
        iT=1.0/Temperature[x]  #define inverse temperature (1/kT)
        z = 0 #stoping param
        while z < equilSteps: # To avoid more correlation from Markov
            metropolisMethod(microstates, iT,B)
            z += 1
        z = 0
        while z < moteCarloSteps: #Actual mesaurments
            metropolisMethod(microstates, iT,B) #metroplois method     
            Mag = magnetization(microstates) # calculate the magnetisation
            
            M1 += Mag
            z += 1 #end param
        #Divide by inverse steps inorder to get correct values
        Magnetization[x] = (1.0/(moteCarloSteps*(N**2)))*M1
        
    return Magnetization,Temperature #Return all values 



def magnetizationExact(N,T): #This function calcualtes the exact magneitzation as defined by the exam. T crictical was calcualted as seen
    #NOTE, function actually does not exist after Temperature critical, and therefore no limiting variables needed, goes to inifitiy.
    tempCritical = 1.64*10**23 #calcualted temperature critical
    
    j = 1 #magnetic field constant
    beta = 1/((T)) #calcculates 1/kT, however my temperature is alread kT
    mExact = N*(1-np.sinh(2*beta*j)**(-4))**(1.0/8) #Calcualtes mExact as defined in the exam
    return mExact
#%% Part 1 RandomWalk and Fitting

M = 10000 #Define some variables
N = 10000
sigma = np.full((1,N),1) #A list of sigmas for my attempted linear regression
disPlace , Nlist = randomWalkDVectorSquare(M,M) #Calcualte random walk data

disLog = np.abs(np.log(disPlace)) #Turn into Log data
nLog = np.log(Nlist)
nLog[nLog == -inf] = 0 #remove infinities 
disLog[disLog == inf] = 0

yCursion = LinearRegressionWORKING(nLog, disLog, sigma) #Use the working linear recursion function
goodnessFit(disLog, yCursion, sigma) #Call goodness of fit function 

#Plot DATA
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('N steps')
plt.ylabel('Mean Square Displacement (R_n**2)')
plt.title('Mean Square Displacment for random Walk in 2D')
plt.plot(nLog,disLog,'o',label = 'Actual Value')
plt.plot(nLog,yCursion, label = 'expected')
plt.legend()
plt.show()




#%%  Part 2 Importance sampling 
N = [1e3,1e4,1e5,1e6] #Define magnitudes of N for Part 1
sigma = np.full((1,4),1) #Define list of sigmas (NOT REALLY USED)
Y = [] #Define some empty lists
X = [] 

lamTest = [0,0.40312,0.81511,1.62300,3.0] #List of some lambda values

for x in N: #calls function 100 times for each set of N (NOTE this is super slow, so be warned if running, to speed up reduce 100 to 10 or make N list smaller)
    ytemp = [] #Define some empty temp lists
    xtemp = []
    for i in range(0,100): 
        _,y = monteCarloMethod(x) #Calls moneCarlo integration N times, NOTE if you want to run, first slot is for P1 weighted function and second one is for P2, just interchange y and _
        ytemp.append(y) #append data
        xtemp.append(x)
    print('Done') #print done just to give you an idea the code is still running and not broken
    Y.append(ytemp) #append to greater list
    X.append(xtemp)

lambdaProof = lamProof() #Gives data to prove that 0.81511 is ideal value

#Plot distirbutions on histograms  
fig, axs = plt.subplots(2, 2,dpi=200) #Plot 4 diffrent random Walks
plt.xticks(rotation  = 45)
axs[0, 0].hist(Y[0])
axs[0, 1].hist(Y[1])
axs[1, 0].hist(Y[2])
axs[1, 1].hist(Y[3])
#Sets axis for the plots
for ax in axs.flat:
    ax.set(xlabel='Integral Value')

#Prints tabular data
print(np.mean(Y[0]),np.mean(Y[1]),np.mean(Y[2]),np.mean(Y[3]))
print(np.std(Y[0]),np.std(Y[1]),np.std(Y[2]),np.std(Y[3]))

#example tabular daata
sigIP1 = [0.016634894401266847,0.0062428996241374385,0.001809602547757796,0.0007031396716663709]
sigIP2 = [0.005703019942752179,0.0018092053195655784,0.0005879005878562754,0.00015849175891861642]

sigfP1 = (np.sqrt(N)*sigIP1) #Calculates the ratio for standard deviation of f/p
sigfP2 = (np.sqrt(N)*sigIP2)

recP1 = LinearRegressionWORKING(N, sigfP1, sigma) #calcualtes the linear regression of each
recP2 = LinearRegressionWORKING(N, sigfP2, sigma)

#Plots ratio of standard deviation and there respective linear regression
plt.figure()
plt.xlabel('N')
plt.xscale('log')
plt.ylabel('Mean of f/p')
plt.plot(N,sigfP1,'o',label = 'P1')
plt.plot(N,sigfP2,'o', label = 'P2')
plt.plot(N,recP1,label = 'P1 fit')
plt.plot(N,recP2,label = 'P2 fit')
plt.legend()
plt.show()
#%% PART 3 DLA circle 
#Starting and stoping radius'as well as there step values
startRadius,startRadius_step = 5,5               
endRadius,endRadius_step = 50,50

massList = []
radiusList = []
densityTemp = 0
newRad = -1
maxradius = -1 # -1 will be updated on first iteration
N = 200   # for NxN lattice
size = (2 * N) + 1 # so total lattice width is 2L + 1
stuck = 0  #initlize hits

lattice = np.zeros((size, size), dtype=np.int32) #initalize seed
lattice[N, N] =   255  #Defines pixel value for each stuck particle

numParticles = 10000 #Defines number of particles
density,radius = DLA(numParticles) #Calls DLA function

#Density Plot
P0 = [12,-18900000.0] #inital Guess'
popt , _ = curve_fit(linear,radius[60:135],density[60:135],P0) #scipy curve fit for linear fit


radius = np.array(radius) #radius to array
#PLOTTING
plt.figure()
plt.plot((radius[60:135]),(density)[60:135], 'o')
plt.plot((radius[60:135]),(linear(radius[60:135],popt[0], popt[1])),label='fit: D = %5.3f, b = %0f' % (tuple(np.log(popt))))
plt.xlabel('Radius (r)')
plt.ylabel('Mass(M)')
plt.title('Density of DLA fractal')
plt.legend()
plt.show()

#%% Part 4 Random Number Generator    
N = 200  #Deinfe starting variable N  
rndx = randomNumberLinear(N) #Call bad PRNG
rndy = randomNumberLinear(N)
period = calcPeriod()  #Calcualtes period  

#Plot data for graphical test of randomness
plt.figure()
plt.title('Graphical Test of Randomness')
plt.xlabel('Random x')
plt.ylabel('Random y')
plt.plot(rndx,rndy,'o')
plt.show()

#Calcualte the test of uniformity 
x7rand,N7vals = orderMomnets(100, 7,10)
x3rand,N3vals = orderMomnets(100, 3,10)
x1rand,N1vals = orderMomnets(100, 1,10)

cor1, n = randomCorrelation(100, 1)
cor2, n = randomCorrelation(100, 1)
cor3, n = randomCorrelation(100, 1)

#Plots Mean moments
plt.figure()
plt.title('Mean moments')
plt.xscale('log')
plt.ylabel('Kth order Moments')
plt.xlabel('Nth value')
plt.axhline(y=1/2,c='y',label = 'k=1 limit')
plt.axhline(y=1/4,c='g',label = 'k=3 limit')
plt.axhline(y=1/8,c='r',label = 'k=7 limit')
plt.legend()
plt.plot(N7vals,x7rand,'r.')
plt.plot(N3vals,x3rand,'g.')
plt.plot(N1vals,x1rand,'y.')
plt.show()
#Plots average standard devation
plt.figure()
plt.title('Average deviation')
plt.xscale('log')
plt.ylabel('Standard Deviation')
plt.xlabel('Nth value')
plt.plot(N7vals,avgDevation(x7rand,7),'r.',label = 'k=7')
plt.plot(N3vals,avgDevation(x3rand,3),'g.',label = 'k=3')
plt.plot(N1vals,avgDevation(x1rand,1),'y.',label = 'k=1')
plt.legend()
plt.show()

#Plots near neighbour correlations
plt.figure()
plt.title('Near Neighbour Correlations')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('correlation')
plt.xlabel('Nth value')
plt.plot(n,cor1,'.')
plt.plot(n,cor2,'o')
plt.plot(n,cor2,'.')
plt.show()

#THIS IS FOR RUNNING ISING SPIN IN 2D
numOfTempPoints = 40  #number of Temperature points
N = 5 #N spins for NxN matrix
equilSteps = 256       #Steps to avoid correlation
moteCarloSteps = 512       #Mote carlo Steps
B = 1 #Magnetic value

Magnetization,Temperature = getData(numOfTempPoints, N, equilSteps, moteCarloSteps,B) #Returns magnetization values and Temperature values 
MagnetizationExact = magnetizationExact(N,Temperature)

magTot = Magnetization - MagnetizationExact
#Plots data
plt.figure()
plt.plot(Temperature/(1.380649*10**-23), abs(magTot),'go')
plt.xlabel("Temperature (T)") 
plt.ylabel("Magnetization - Magnetization_Exact")
plt.title('2D ising Magnetization with mersenne twister')
plt.vlines(1.64*10**23, min(abs(magTot)), max(abs(magTot)),label = 'Critical Temperature')  
plt.axis('tight')
plt.legend() 
plt.show()