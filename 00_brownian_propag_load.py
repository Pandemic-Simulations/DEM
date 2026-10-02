###############################################################################
# Model description : virus concentration evolution in brownian motion
# 05/2020
###############################################################################
# Virus need close contact between people. No propagation is possible unless 
# people get closer than a certain distance (d_contagion). When two people are 
# close enough durint a small time dt and one of them is sick (his virus 
# concentration C is above a threshold V_contagious), the non sick receive a small 
# virus charge dC_in which is proportional to (1-d/d_contagion) with d being 
# the current distance between the two people, to the time of contact dt and to 
# the virus charge rejected by the sick person. This person reject a virus 
# concenration dC_out proportional to its virus charge C and to the frequency 
# at which he cough (coughrate). In addition to all of that, the sick and non
# sick can limit the risk of virus exchange by using individual protections.
# This is taken into account by additional factors (1-protec_in) and 
# (1-protec_out). All in all, the exchange of virus id given by
# dC_in=(1-protec_in)*(1-d/d_contagion)*dC_out
# dC_out=(1-protec_out)*coughRate*C*dt if V>V_contagious
#
# The amount of virus exchange depends on the virus charge C in the sick
# person body. The evolution of this virus charge is given by the two equations
# dC/dt=(virusGrowth-virusDeath*L)*C
# dL/dt=immuneSpeed*C(1-L)-qL
# where L is the activation degree of the immune system. 
# The parameter virusGrowth characterize the virus growth, virusDeath the 
# efficiency of the immune system, immuneSpeed the speed of activation of the 
# immune system
#
# We enforce social distancing through the use of different repulsive modulus 
# between particles. A person respecting social distancing will have a "young 
# modulus" (E_careful) stiffer than people not respecting it (E_reckless). 
# We assume that only a fraction of the population (careFrac) respect social 
# distancing. When people get really sick (C>Cc) we assume that they will now
# respect social distancing.
#
# The simulation considers a given number (Nparts) of people moving in a box of
# size Lx x Ly. The position and the displacement direction on the people are 
# set randomly with a given velocity (Vref). During the simulation people speed
# may vary due to elastic collision, but the total energy of the system remains
# constant. The people density together with the initial velocity drives the 
# frequency of contacts.
# 
# People move around in a Brownian motion and their contacts are checked
# regularly (every checkTime). Virus charge is then updated. Initially we 
# assume that a fraction (initInfectedFract) of the people are infected. We 
# give them randomly a small virus charge while their immune system is not 
# yet activated.
###############################################################################
import numpy as np
import random
import os
import pickle

###############################################################################
# List of parameters social parameters
###############################################################################
# Command to be used to start the parametric study 
# with 10 cpu (-j10) and 2 cpus per job (--job-threads=2)
# yadedaily-batch -j2 --job-threads=2 params.table brownian_movement.py
readParamsFromTable(ide = 11, simuNameA='A_20231212T174547p1902200', setBehavior='1', 
                    setInfection = '1',setPop = '1',setConf = '1', initInfectedFract = 0.05,
                    careFrac=0.0,dmin_care=1.5,dmin_reck=0.5,carefulIfSick=0,
                    protec_in=0,protec_out=0,confStartTimeD=10,confEndTimeD=20,
                    confFrac=0.8,d_contagion=1.,m_target=50.,coughRate=10.,
                    coughVirusFrac=10.**-4,V_contagious=0.05,V_recovery=1.5,
                    V_infected=1.5*10**-3,incubationTimeD=3.4,recoveryTimeD=6.,
                    immuneActTimeD=52.,p_symptomatic=0.49)
from yade.params import table


# time parameters
dayTime = 24              # day time in h
nCheckPerContact=10       # number of contact checks during a typical contact 
                          # duration

#loading state
simuNameA = table.simuNameA                    # name of the initial particles disposition state
O.load('statesA' + '/' + simuNameA + '.yade')  # loaging of the initial particles disposition state
setBehavior = table.setBehavior
setInfection = table.setInfection
setPop = table.setPop
setConf = table.setConf
ide = table.ide


# general parameters and social interaction
# read in list_StateA the parameters used in the considered stateA
Nparts = 0 # number of people
phi = 0.
D_interaction = 0.
Ncd = 0
with open('statesA/list_StateA.txt','r') as f:
    statesA = f.read().split('\n')
statesA = [row.split('\t') for row in statesA]
for i in range (0,len(statesA)):
        if statesA[i][0] == simuNameA:
            Nparts = int(statesA[i][2])        # number of people
            phi = float(statesA[i][3])           # particle density (between 0 and pi/4)
            D_interaction = float(statesA[i][4]) # (m) social interaction distance
            Ncd = float(statesA[i][5])           # estimated number of daily contacts 
initInfectedFract = table.initInfectedFract  # infected initial fraction
careFrac = table.careFrac              # fraction of people respecting social distancing
dmin_care = table.dmin_care            # for a typical minimal distance of interaction 1m (needs back fitting)
dmin_reck = table.dmin_reck            # for a typical minimal distance of interaction 0.4m (needs back fitting)
carefulIfSick = table.carefulIfSick    # change of attitude of symptomatic people
m_target = table.m_target

print(Nparts)
# individual protection such as masks, hygiene,...
protec_in = table.protec_in           # to avoid beeing contaminated
                                       # 0 : no protection, 1 : total protection
protec_out = table.protec_out         # to avoid contaminate
                                       # 0 : no protection, 1 : total protection

# Confinement parameters. 
# To prevent confinement set startTime>endTime or confFrac=0
confStartTime = table.confStartTimeD*dayTime # (h) begining of the confinement
confEndTime = table.confEndTimeD*dayTime     # (h) end of the confinement
confFrac = table.confFrac                    # fraction of the population concerned

# demographic characteristics 
popStatsDict = {'ageCat':        ['young','adult','old'],
               'popFrac':       [0.1539  ,0.3605   ,0.1345],
               'skinnyFrac':    [0.04    ,0.014    ,0.018],
               'normalFrac':    [0.612   ,0.419    ,0.301],
               'overweightFrac':[0.247   ,0.385    ,0.466],
               'obeseFrac':     [0.101   ,0.182    ,0.215]}
#Recuperation of the pickle lists: 
picklePopName = 'pickles/pop_set' + setPop + str(Nparts) + '.pickle'
pickleBehavName = 'pickles/behavior_set' + setBehavior + str(Nparts) + '.pickle'
pickleInfectionName = 'pickles/infection_set' + setInfection + str(Nparts) + '.pickle'
pickleConfName = 'pickles/conf_set' + setConf + str(Nparts) + '.pickle'

# Travelling parameters
longRangeDailyFrac = 0.01                        # fraction of people travelling far away each day
N_neighbours = {'young':15,'adult':30,'old':10}  # accessible neighbours within a day

###############################################################################
R_interaction = D_interaction/2.
Lx = R_interaction*np.sqrt(Nparts*np.pi/phi)          # (m x m) size of the box
Ly = Lx                             
Vref=(np.sqrt(Lx*Ly/Nparts)-2*R_interaction)*Ncd/dayTime     # (m/h) mean speed
Vinit=np.sqrt(2)*Vref                                 # (m/h) initial speed. 
                                                      # The square root has a meaning but I do not remember

# stiffness of the reckless and careful people
E_reckless = m_target*Vref**2/(2*R_interaction*(R_interaction-dmin_reck/2)**2)
E_careful = m_target*Vref**2/(2*R_interaction*(R_interaction-dmin_care/2)**2)

# check contacts sufficiently frequently 
w_care = (2*E_careful*R_interaction/m_target)**0.5
checkTime = 1./nCheckPerContact*np.pi/w_care # the typical time of interaction 
                                             # is half of the harmonic 
                                             # oscillator period which is 
                                             # pi*(m/2kn)**0.5. 

# distance from which a coming back force is applied
farFromHomeDist={k:np.sqrt(N_neighbours[k]/phi)*R_interaction \
                 for k in N_neighbours.keys()}

###############################################################################
# List of parameters virus transmission parameters
###############################################################################
# Contagion law parameters
d_contagion = table.d_contagion     # (m) 
# NB: what is indeed important is the ratio d_interaction/d_contagion, the 
#     travel distances does not correspond to physical space
coughRate = table.coughRate         # the number of cough per hour
coughVirusFrac=table.coughVirusFrac # fraction of virus expelled at each cough

V_contagious = 0.05                 # virus charge above which the person is 
                                    # sick and start expelling virus
V_recovery = 1.5
V_infected = 1.5*10**-3             # given for the reference case phi=0.5 Ncd=40

# Characteristic times
incubationTime = table.incubationTimeD*dayTime#*8/7.3
recoveryTime = table.recoveryTimeD*dayTime#*8/7.3
immuneActTime = table.immuneActTimeD*dayTime#*8/7.3

# Characteristic times
incubationTime = table.incubationTimeD*dayTime#*8/7.3
recoveryTime = table.recoveryTimeD*dayTime#*8/7.3
immuneActTime = table.immuneActTimeD*dayTime#*8/7.3

# Probability to develop symptoms
p_symptomatic = table.p_symptomatic
###############################################################################
# Automatic definition of parameters
# parameters of the prey/predator model for the defense system against virus
immuneActSpeed = 4./immuneActTime# speed at which the lymphocites are activated 
                                 # Characteristic time 1/immuneActSpeed for C=1
                                 # (can depend on people characteristics)
virusGrowth = 4./incubationTime  # speed of reproduction of the virus 
                                 # Characteristic time 1/virusGrowth
                                 # (intrinsic to the virus)
virusDeath = virusGrowth+\
             4./recoveryTime     # efficacity of the lymphocites in killing viruses 
                                 # Characteristic time 1/virusDeath for L=1
                                 # if virusDeath>virusGrowth the immune system win
                                 # (can depend on people characteristics))
###############################################################################
# Recording parameters
###############################################################################
record = True                  # whether vtk files are saved
###############################################################################
# Une periodic boundary conditions
###############################################################################
periodicBC=True
###############################################################################
# give an automatic name to the simu
simuName=simuNameA+str(ide).zfill(4)+'_'+str(int(random.random()*1000))
print(simuName)
if record:
    if not os.path.exists('visus'):
        os.makedirs('visus')
    os.mkdir('visus/'+simuName)
if not os.path.exists('results'):
    os.makedirs('results') 
os.mkdir('results/'+simuName)
###############################################################################
# Define an agent class
class Agent:
    """
    Define a class of agent:
    - the id of the particle (id)
    - infection status (isInfected)
    - infection time (contagiousTime)
    - futur recovery or death if infected (willDie)
    - daily viral charge (dailyCharge)
    - cautiousness between 0 and 1 (cautiousness)
    """
    def __init__(self):
        # agent characteristics
        self.id=-1
        self.socialDistancing=0
        self.protecIn=0
        self.protecOut=0
        self.ageCategory=''
        self.BMICategory=''
        self.homePos=[]
        self.farFromHomeDist=0
        self.immuneEfficiency=1
        self.immuneActSpeed=0
        # current state parameters
        self.virusCharge=0
        self.immuneActivation=0
        self.infectionStatus=''
        # history recorded parameters
        self.maxVirusCharge=0
        self.infectedTime=-1
        self.contagiousTime=-1
        self.peakContagiousTime=-1
        self.recoverTime=-1
        self.infectedIds=set()
        self.symptomatic=0
        # for daily statistics
        self.contactTags=[]
        self.contactTimes=[]
        self.contactMinDists=[]
        self.contactMaxVels=[]
        self.contactdCin=[]
        self.dayInitPos=[]

    def dict(self):
        print('Agent instance: ',self.id)
        print('  virusCharge = ',self.virusCharge)
        print('  immuneActivation = ',self.immuneActivation)
        print('  contagiousTime = ',self.contagiousTime)
        print('  recoverTime = ',self.recoverTime)
        print('  infectedIds = ',self.infectedIds)
        print('     immuneEfficiency = ',self.immuneEfficiency)
        print('     immuneActSpeed = ',self.immuneActSpeed )
        print('     ageCategory = ',self.ageCategory)
        print('     BMICategory = ',self.BMICategory)
        print('         socialDistancing = ',self.socialDistancing)
        print('         protecIn = ',self.protecIn)
        print('         protecOut = ',self.protecOut)

infectionTable={} # to record all data on agents virus charge
###############################################################################
# define two materials, one for reckless, one for careful people
materialNames=['reckless','careful']
rho=(3*m_target)/(4*np.pi*R_interaction**3)
walls = O.materials.append(FrictMat(density=rho,young=E_careful,poisson=0.25,
                                    frictionAngle=radians(0),label='walls'))
careful = O.materials.append(FrictMat(density=rho,young=E_careful,poisson=0.25,
                                      frictionAngle=radians(0),label='careful'))
reckless = O.materials.append(FrictMat(density=rho,young=E_reckless,poisson=0.25,
                                       frictionAngle=radians(0),label='reckless'))
###############################################################################
# Define the bounding box
if periodicBC==True:
    NSetOfWalls=0
    O.periodic = True
    O.cell.setBox(Lx,Ly,4*D_interaction)
else:
    NSetOfWalls=10 #to increase contact forces with wall to avoid loss of agents
    for i in range(NSetOfWalls):
        O.bodies.append(utils.wall(-i*R_interaction/NSetOfWalls,axis=0,sense=1,
                                   material='walls',color=(0,0,0)))
        O.bodies.append(utils.wall(Lx+i*R_interaction/NSetOfWalls,axis=0,sense=-1,
                                   material='walls',color=(0,0,0)))
        O.bodies.append(utils.wall(-i*R_interaction/NSetOfWalls,axis=1,sense=1,
                                   material='walls',color=(0,0,0)))
        O.bodies.append(utils.wall(Ly+i*R_interaction/NSetOfWalls,axis=1,sense=-1,
                                   material='walls',color=(0,0,0)))
###############################################################################                               
# Define the age and BMI of all agents
popStatus=[]
for i in range(len(popStatsDict['ageCat'])):
    cat=popStatsDict['ageCat'][i]
    Ncat=int(Nparts*popStatsDict['popFrac'][i])
    Nskinny=int(Ncat*popStatsDict['skinnyFrac'][i])
    Nnormal=int(Ncat*popStatsDict['normalFrac'][i])
    Noverweight=int(Ncat*popStatsDict['overweightFrac'][i])
    Nobese=int(Ncat*popStatsDict['obeseFrac'][i])
    popStatus+=[(cat,'skinny') for j in range(Nskinny)]
    popStatus+=[(cat,'normal') for j in range(Nnormal)]
    popStatus+=[(cat,'overweight') for j in range(Noverweight)]
    popStatus+=[(cat,'obese') for j in range(Nobese)]
# add a few standart agents
popStatus+=[('adult','normal') for j in range(len(popStatus),Nparts)]

orderPop = pickle.load(open(picklePopName, 'rb'))
behaviorStatus = pickle.load(open(pickleBehavName,'rb'))
confStatus = pickle.load(open(pickleConfName,'rb'))
for i in range(Nparts):
    agent=Agent()
    
    # add the agent to the simulation
    agent.id=orderPop[i]
    
    x = O.bodies[agent.id].state.refPos[0]
    y = O.bodies[agent.id].state.refPos[1]
    # set individual caracteristics of self defense against virus 
    agent.immuneEfficiency=virusDeath
    agent.immuneActSpeed=immuneActSpeed

    # assume whether the agent respect social distancing or not
    if (float(behaviorStatus[i])/float(Nparts))<careFrac:
        agent.socialDistancing=1
        O.bodies[agent.id].material=O.materials[materialNames[1]]
    else:
        agent.socialDistancing=0
        O.bodies[agent.id].material=O.materials[materialNames[0]]

    # enforce individual protections
    agent.protecIn=protec_in
    agent.protecOut=protec_out

    # select age and BIM status
    agent.ageCategory=popStatus[agent.id][0]
    agent.BMICategory=popStatus[agent.id][1]
    
    # set home adress and range of travel
    agent.farFromHomeDist=farFromHomeDist[agent.ageCategory]
    agent.homePos=[x,y]

    # set up the initially infected people
    if i<initInfectedFract*Nparts:
        agent.virusCharge=2*V_infected#+random.random()*(V_contagious-V_infected)
        agent.infectionStatus='infected'
        agent.socialDistancing=0 # initial infected people are reckless
        agent.infectedTime=0
        col=[1,1,0] # yellow particles are infected but not contagious 
    else:
        agent.virusCharge=0
        agent.infectionStatus='healthy'
        col=[0,1,0] # green particles are healthy people
        
    # record the initial position
    agent.dayInitPos=[[x,y]]
    O.bodies[agent.id].shape.color=col
    #O.bodies[agent.id].material=materialNames[agent.socialDistancing]
    # record the agent in infectionTable
    infectionTable[agent.id]=agent

# Initialize the confinement status variable
isConfActivated=False

# record the energies (just to check whether no dissipation occur).
O.trackEnergy=True
###############################################################################
# Define the simulation loop
O.engines=[
   ForceResetter(),
   InsertionSortCollider([Bo1_Sphere_Aabb(),Bo1_Wall_Aabb()]),
   InteractionLoop(
      [Ig2_Sphere_Sphere_ScGeom(),Ig2_Wall_Sphere_ScGeom()],
      [Ip2_FrictMat_FrictMat_FrictPhys()],
      [Law2_ScGeom_FrictPhys_CundallStrack()]
   ),
   # to avoid agent travelling to far from home
   PyRunner(command='goHome()',label='goHome',dead=False),
   # Newton's law
   NewtonIntegrator(gravity=(0,0,0),damping=0),
   # to estimate some daily social properties
   PyRunner(command='contactStatistics()',label='contactStats',dead=False), 
   # virus transmission step
   PyRunner(command='contactCheck()',label='cCheck',dead=False),
   # daily check  
   PyRunner(command='dailyStatistics()',label='dCheck',dead=False), 
   # record some vtk files for visualization purposes
   VTKRecorder(label='vtkRecorder',recorders=['spheres','colors'],
               fileName='./visus/'+simuName+'/',dead=not record) 
]
###############################################################################
# define the time step
O.dt=checkTime/10. # in h
if O.dt>0.9*PWaveTimeStep():
   print('Too large time step !')
###############################################################################
# define the frequency of checks and recording
dCheck.iterPeriod=int(1/O.dt*dayTime) 
cCheck.iterPeriod=int(1/O.dt*checkTime)
contactStats.iterPeriod=int(1/O.dt*checkTime)
goHome.iterPeriod=10*int(1/O.dt*checkTime)

# estimate the typical time of free flight if we assume a uniform spatial 
# repartition of the spheres. Then each sphere has a typical surface around it
# dS=LxLy/Nparts and will get out of it at speed Vref in time
# (LxLy/Npart)**0.5/2Vref 
recFreq=(Lx*Ly/Nparts)**0.5/(2*Vref)
vtkRecorder.iterPeriod=int(1/O.dt*recFreq)

###############################################################################
# record the virus charges and immune activation of a few agents
virusCharges={i:[] for i in random.sample(infectionTable.keys(), k=40)}
# follow the initially infected agents
for i in infectionTable.keys():
    if infectionTable[i].virusCharge>0:
        virusCharges[i]=[]
virusCharges['time']=[]
immuneActivations={i:[] for i in virusCharges.keys()}
immuneActivations['time']=[]
###############################################################################
# Record some daily statistics
# nC,nC_positive : mean number of daily contacts and infectious contacts
# meanCTime, stdCTime : mean contact time and standard deviation around it
# meanMaxVel: mean maximum relative velocity at contact
# meanDailyDists:  mean travelled distances each day
# meanMinDist_ReckReck,stdMinDist_ReckReck: mean minimal distance and standard 
#                                           deviation for contacts involving 
#                                           two agents not respecting social 
#                                           distancing
# meanMinDist_CareCare, stdMinDist_CareCare: mean minimal distance and standard 
#                                            deviation for contacts involving 
#                                            two agents respecting social 
#                                            distancing
# meanMinDist_ReckCare, stdMinDist_ReckCare: mean minimal distance and standard 
#                                            deviation for hybrid contacts
# meandCin, stddCin: mean virus charge and standard deviation for positive contacts
# meandCin_ReckReck, stddCin_ReckReck: same as above but restricted to contacts 
#                                      involving two agents not respecting  
#                                      social distancing
# meandCin_CareCare, stddCin_CareCare: same as above but restricted to contacts 
#                                      involving two agents respecting social 
#                                      distancing
# meandCin_ReckCare, stddCin_ReckCare: same as above but restricted to hybrid
#                                      contacts
# 'totEnergy': energy of the simulation (check for non dissipation)
dailyStats={'nC':[],'nC_positive':[],
            'meanCTime':[],'stdCTime':[],
            'meanMaxVel':[],'meanDailyDists':[],
            'meanMinDist_ReckReck':[],'stdMinDist_ReckReck':[],
            'meanMinDist_CareCare':[],'stdMinDist_CareCare':[],
            'meanMinDist_ReckCare':[],'stdMinDist_ReckCare':[],
            'meandCin':[],'stddCin':[],
            'meandCin_ReckReck':[],'stddCin_ReckReck':[],
            'meandCin_CareCare':[],'stddCin_CareCare':[],
            'meandCin_ReckCare':[],'stddCin_ReckCare':[],
            'totEnergy':[]
            }
###############################################################################
# record the virus doses transmitted together with contact type and time from 
# infection
virusDoses={'dC':[],'tFromInfected':[],'ctactType':[]}

###############################################################################
# Functions called in the simulation loop
###############################################################################
def contactStatistics():
    """
    keep track of contact statistics
    """
    global contactDict,infectionTable,virusCharges
    
    for intr in O.interactions:
        # get the ordered body ids
        id1=min([intr.id1,intr.id2])
        id2=max([intr.id1,intr.id2])
        if isinstance(O.bodies[id1].shape,Sphere) and \
           isinstance(O.bodies[id1].shape,Sphere):
            # get the signature for the current contact: bodies in contact and 
            # moment of creation
            tag=(id1,id2,intr.iterMadeReal)
            relVel=O.bodies[id1].state.vel-O.bodies[id2].state.vel
            relVel=relVel.norm()
            dist=O.bodies[id1].shape.radius+O.bodies[id2].shape.radius-\
                 intr.geom.penetrationDepth
            agent1=infectionTable[id1]
            agent2=infectionTable[id2]
            if tag in agent1.contactTags:
                # update contact attributes for each agent if the contact exists
                i=agent1.contactTags.index(tag)
                agent1.contactTimes[i]=(O.iter-tag[2])*O.dt
                agent1.contactMinDists[i]=min(dist,agent1.contactMinDists[i])
                agent1.contactMaxVels[i]=max(relVel,agent1.contactMaxVels[i])
                
                i=agent2.contactTags.index(tag)
                agent2.contactTimes[i]=(O.iter-tag[2])*O.dt
                agent2.contactMinDists[i]=min(dist,agent2.contactMinDists[i])
                agent2.contactMaxVels[i]=max(relVel,agent2.contactMaxVels[i])
            else:
                # add the contact to the contact list of both agents
                agent1.contactTags+=[tag]
                agent1.contactTimes+=[(O.iter-tag[2])*O.dt]
                agent1.contactMinDists+=[dist]
                agent1.contactMaxVels+=[relVel]
                agent1.contactdCin+=[0]
                
                agent2.contactTags+=[tag]
                agent2.contactTimes+=[(O.iter-tag[2])*O.dt]
                agent2.contactMinDists+=[dist]
                agent2.contactMaxVels+=[relVel]
                agent2.contactdCin+=[0]
    # record the virus charges of a few agents
    for i in virusCharges.keys():
        if i!='time':
            virusCharges[i]+=[infectionTable[i].virusCharge]
            immuneActivations[i]+=[infectionTable[i].immuneActivation]
    virusCharges['time']+=[O.time]
    immuneActivations['time']+=[O.time]

def dailyStatistics():
    """
    get daily statistics 
    enforce confinement if needed
    check whether the epidemy has finished
    """
    global infectionTable,contactDict,dailyStats,dayTime
    global contactStats
    global simuName
    global periodicBC
    global isConfActivated
    global virusDoses
    # count the infected, sick and recovered and display it
    nSick,nRecovered,nInfected=0,0,0
    for i in infectionTable.keys():
        if infectionTable[i].infectionStatus=='infected':
            nInfected+=1
        elif infectionTable[i].infectionStatus=='contagious' or\
             infectionTable[i].infectionStatus=='symptomatic' or\
             infectionTable[i].infectionStatus=='asymptomatic':
            nSick+=1
        elif infectionTable[i].infectionStatus=='recovered':
            nRecovered+=1
    print('')
    s='At day %0.0f, %i people are infected, '%(O.time/dayTime,nInfected)
    s+='%i people are contagious and %i people have recovered'%(nSick,nRecovered)
    print(s)
    
    #with open('results/'+simuName+'/'+simuName+'time_stat.txt','a+') as f:
    #    f.write('%0.0f\t%i\t%i\t%i\n'%(O.time/dayTime,nInfected,nSick,nRecovered))
    #    print('ok')
    # update the daily statistics if contact stats are recorded
    if contactStats.dead==False:
        # get the contact statistics
        # contact durations
        cTimes=[]
        # maximum relative velocities at contact
        cMaxVels=[]
        # daily travel distances (in straight line)
        dailyDists=[]
        # minimal distances at contact
        cMinDists_ReckReck,cMinDists_CareCare,cMinDists_ReckCare=[],[],[]
        # viral charge exchanged (only if non zero)
        dCin,dCin_ReckReck,dCin_CareCare,dCin_ReckCare=[],[],[],[]
        for i in infectionTable.keys():
            agent=infectionTable[i]
            cTimes+=agent.contactTimes
            cMaxVels+=agent.contactMaxVels
            pos=O.bodies[i].state.pos
            dayInitPos=agent.dayInitPos[-1]
            agent.dayInitPos+=[[pos[0],pos[1]]]
            dailyDists+=[sum([(pos[j]-dayInitPos[j])**2 for j in range(2)])**0.5]
            dCin+=[dC for dC in agent.contactdCin if dC>0]
            for j in range(len(agent.contactTags)):
                id1=agent.contactTags[j][0]
                id2=agent.contactTags[j][1]
                # get whether the agent respect social distancing 
                # (symptomatic agents may respect social distancing even if 
                # they were initially reckless)
                soc1=O.bodies[id1].material.label
                soc2=O.bodies[id2].material.label
                if agent.contactMinDists[j]<=0:
                    print('Problem here, interactions between agent '+str(agent.id)+ 'and '+str(j) +'gives negative distance')
                if soc1=='careful' and soc2=='careful':
                    cMinDists_CareCare+=[agent.contactMinDists[j]]
                    if agent.contactdCin[j]>0:
                        dCin_CareCare+=[agent.contactdCin[j]]
                        virusDoses['dC']+=[agent.contactdCin[j]]
                        virusDoses['ctactType']+=['CareCare']
                        if id1==i:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id2].infectedTime]
                        else:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id1].infectedTime]
                elif soc1=='reckless' and soc2=='reckless':
                    cMinDists_ReckReck+=[agent.contactMinDists[j]]
                    if agent.contactdCin[j]>0:
                        dCin_ReckReck+=[agent.contactdCin[j]]
                        virusDoses['dC']+=[agent.contactdCin[j]]
                        virusDoses['ctactType']+=['ReckReck']
                        if id1==i:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id2].infectedTime]
                        else:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id1].infectedTime]
                else:
                    cMinDists_ReckCare+=[agent.contactMinDists[j]]
                    if agent.contactdCin[j]>0:
                        dCin_ReckCare+=[agent.contactdCin[j]]
                        virusDoses['dC']+=[agent.contactdCin[j]]
                        virusDoses['ctactType']+=['ReckCare']
                        if id1==i:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id2].infectedTime]
                        else:
                            virusDoses['tFromInfected']+=[O.time-infectionTable[id1].infectedTime]
        # compute the daily stats 
        # daily mean contacts per agent
        dailyStats['nC']+=[float(len(cTimes))/len(infectionTable.keys())]
        # daily mean positive contacts per agent contagious
        if nSick>0:
            dailyStats['nC_positive']+=[float(len(dCin))/nSick]
        else:
            dailyStats['nC_positive']+=[0]
        # daily mean contact duration and standard deviation
        dailyStats['meanCTime']+=[np.mean(cTimes)]
        dailyStats['stdCTime']+=[np.std(cTimes)]
        # daily mean minimal distance at contact and standard deviations
        dailyStats['meanMinDist_ReckReck']+=[np.mean(cMinDists_ReckReck)]
        dailyStats['meanMinDist_CareCare']+=[np.mean(cMinDists_CareCare)]
        dailyStats['meanMinDist_ReckCare']+=[np.mean(cMinDists_ReckCare)]
        dailyStats['stdMinDist_ReckReck']+=[np.std(cMinDists_ReckReck)]
        dailyStats['stdMinDist_CareCare']+=[np.std(cMinDists_CareCare)]
        dailyStats['stdMinDist_ReckCare']+=[np.std(cMinDists_ReckCare)]
        # daily mean maximal relative velocity at contact
        dailyStats['meanMaxVel']+=[np.mean(cMaxVels)]
        # daily mean travelled distance
        dailyStats['meanDailyDists']+=[np.mean(dailyDists)]
        # daily mean virus charge exchange and standard deviation 
        # for positive contacts
        if dCin!=[]:
            dailyStats['meandCin']+=[np.mean(dCin)]
            dailyStats['stddCin']+=[np.std(dCin)]
        else:
            dailyStats['meandCin']+=[0]
            dailyStats['stddCin']+=[0]
        if dCin_ReckReck!=[]:
            dailyStats['meandCin_ReckReck']+=[np.mean(dCin_ReckReck)]
            dailyStats['stddCin_ReckReck']+=[np.std(dCin_ReckReck)]
        else:
            dailyStats['meandCin_ReckReck']+=[0]
            dailyStats['stddCin_ReckReck']+=[0]
        if dCin_CareCare!=[]:
            dailyStats['meandCin_CareCare']+=[np.mean(dCin_CareCare)]
            dailyStats['stddCin_CareCare']+=[np.std(dCin_CareCare)]
        else:
            dailyStats['meandCin_CareCare']+=[0]
            dailyStats['stddCin_CareCare']+=[0]
        if dCin_ReckCare!=[]:
            dailyStats['meandCin_ReckCare']+=[np.mean(dCin_ReckCare)]
            dailyStats['stddCin_ReckCare']+=[np.std(dCin_ReckCare)]
        else:
            dailyStats['meandCin_ReckCare']+=[0]
            dailyStats['stddCin_ReckCare']+=[0]
        dailyStats['totEnergy']+=[O.energy['elastPotential']+O.energy['kinetic']]
        
        # reset the contact stats
        for i in infectionTable.keys():
            infectionTable[i].contactTags=[]
            infectionTable[i].contactTimes=[]
            infectionTable[i].contactMinDists=[]
            infectionTable[i].contactMaxVels=[]
            infectionTable[i].contactdCin=[]

    if contactStats.dead==False:
        print('Daily contacts is on average : %0.2f'%(dailyStats['nC'][-1]))
        print('Daily contacts time is on average : %0.2f h'%(dailyStats['meanCTime'][-1]))
        # estimate an effective min distance at contact 
        # (for a normal distribution, 97.5% of the minimal distances are 
        # above mean-2 std)
        dmin_ReckReck=dailyStats['meanMinDist_ReckReck'][-1]-\
                      2*dailyStats['stdMinDist_ReckReck'][-1]
        dmin_CareCare=dailyStats['meanMinDist_CareCare'][-1]-\
                      2*dailyStats['stdMinDist_CareCare'][-1]
        dmin_ReckCare=dailyStats['meanMinDist_ReckCare'][-1]-\
                      2*dailyStats['stdMinDist_ReckCare'][-1]
        s='Daily min dists are for reck/reck, care/care and reck/care: '
        s+='%0.2f m,%0.2f m,%0.2f m'%(dmin_ReckReck,dmin_CareCare,dmin_ReckCare)
        print(s)
        s='Daily mean virus exchange for risky contacts are for reck/reck,'+\
          ' care/care and reck/care : '
        s+='%0.2e, %0.2e, %0.2e'%(dailyStats['meandCin_ReckReck'][-1],\
                                  dailyStats['meandCin_CareCare'][-1],\
                                  dailyStats['meandCin_ReckCare'][-1])
        print(s)
        print(dailyStats['meanMinDist_ReckReck'][-1])
        print(dailyStats['stdMinDist_ReckReck'][-1])
        with open('picklecMinDists_ReckReck', 'wb') as handle:
            pickle.dump(cMinDists_ReckReck, handle, protocol=2)
    # try to enforce confinement
    enforceConfinement()
    
    # try to make people travel if confinement is not on
    if isConfActivated==False:
        makeTravelPeople()
    
    # check whether the epidemic is over
    if (nSick==0 and nInfected==0): # or (O.iter == 100):
        O.pause()
        #saveData('results/'+simuName+'/'+simuName+'.txt')
                
def enforceConfinement():
    """
    enforce confinement/deconfinement
    """
    global infectionTable
    # general parameters
    global Nparts,R_interaction,totEnergy
    # confinement parameters
    global confStartTime,confEndTime,confFrac,isConfActivated
    if confStartTime<=O.time<confEndTime and isConfActivated==False:
        nFixed=int(confFrac*Nparts)
        pickedIds = []
        for i in range (0,Nparts):
            if confStatus[i]<= nFixed:
                pickedIds+=[infectionTable[i].id]
        #pickedIds=random.sample(infectionTable.keys(), k=nFixed)
        #print('nFixed',nFixed,pickedIds)
        print(len(pickedIds),pickedIds[1])
        for i in pickedIds:
            O.bodies[i].state.vel=[0,0,0]
            O.bodies[i].state.blockedDOFs='xyzXYZ'
        isConfActivated=True
        
        print('Confinement started')
    if O.time>=confEndTime and isConfActivated==True:
        currentEnergy=O.energy['elastPotential']+O.energy['kinetic']
        deltaE=totEnergy-currentEnergy
        m=4./3*np.pi*R_interaction**3*O.materials['reckless'].density
        fixedIds=[i for i in infectionTable.keys() 
                  if O.bodies[i].state.blockedDOFs=='xyzXYZ']
        #print(m,len(fixedIds),O.materials['reckless'].density)
        # set the velocity value of deconfined particles to restore the total
        # energy to its value before confinement
        v=(2*deltaE/(m*len(fixedIds)))**0.5
        for i in infectionTable.keys():
            if O.bodies[i].state.blockedDOFs=='xyzXYZ':
                theta=random.random()*2*np.pi
                O.bodies[i].state.vel[0]=v*np.cos(theta)
                O.bodies[i].state.vel[1]=v*np.sin(theta)
                O.bodies[i].state.blockedDOFs=''
        isConfActivated=False
        print('Confinement stopped')

def makeTravelPeople():
    """
    chose randomly some people (without any contacts) and make them travel 
    far away (find a spot where they can be put with no initial contacts)    
    """
    global infectionTable
    global Nparts,longRangeDailyFrac,Lx,Ly,R_interaction
    global periodicBC
    Ntravellers=int(Nparts*longRangeDailyFrac)
    # chose the travellers among rattlers
    excludeIds=set()
    for intr in O.interactions:
        excludeIds.add(intr.id1)
        excludeIds.add(intr.id2)
    ids=set(infectionTable.keys()).difference(excludeIds)
    if len(ids)>Ntravellers:
        ids=random.sample(ids,Ntravellers)
    # try to make move travellers
    effectiveTravellers=0
    for i in ids:
        ntry=0
        test=False
        while ntry<30 and test==False:
            test=True
            ntry+=1
            x=R_interaction+random.random()*(Lx-2*R_interaction)
            y=R_interaction+random.random()*(Ly-2*R_interaction)
            for j in infectionTable.keys():
                pos=[O.bodies[j].state.pos[0],O.bodies[j].state.pos[1]]
                # account for periodic boundary conditions 
                # (pos is not given modulo Lx and Ly)
                if periodicBC==True:
                    Lx=O.cell.hSize[0,0]
                    Ly=O.cell.hSize[1,1]
                    if pos[0]>0:
                        pos[0]=pos[0]-int(pos[0]/Lx)*Lx
                    else:
                        pos[0]=pos[0]+(1+int(-pos[0]/Lx))*Lx
                    if pos[1]>0:
                        pos[1]=pos[1]-int(pos[1]/Ly)*Ly
                    else:
                        pos[1]=pos[1]+(1+int(-pos[1]/Ly))*Ly
                d2=(x-pos[0])**2+(y-pos[1])**2
                if d2<(2*R_interaction)**2:
                    test=False
        if test==True:
            pos=O.bodies[i].state.pos
            O.bodies[i].state.se3=(Vector3(x,y,pos[2]),O.bodies[i].state.se3[1])
            infectionTable[i].homePos=[x,y]
            effectiveTravellers+=1
    print('%i people have travelled far away today'%(effectiveTravellers))

def goHome():
    """
    modify the velocity direction of the agents to avoid they go to far from 
    their starting position. This modification avoid modifying the energy of 
    the system
    """
    global m_target,adjustVTime,infectionTable
    global periodicBC
    for i in infectionTable.keys():
        # find the initial position and current distance from it
        homePos=infectionTable[i].homePos
        farFromHomeDist=infectionTable[i].farFromHomeDist
        pos=O.bodies[i].state.pos
        dx=pos[0]-homePos[0]
        dy=pos[1]-homePos[1]
        d2=dx**2+dy**2
        # if the agent is to far from its initial position and not in confinement
        if d2>farFromHomeDist**2 and O.bodies[i].state.blockedDOFs!='xyzXYZ':
            # get its velocity 
            velDir=1/O.bodies[i].state.vel.norm()*O.bodies[i].state.vel
            # get the direction from its starting position to its current position
            rappelDir=Vector3(-dx/d2**0.5,-dy/d2**0.5,0)
            # get the perpendicule direction to it (trigonometric positive rot)
            perpRappelDir=Vector3(rappelDir[1],-rappelDir[0],0)
            # get the angle between velocity and "goHome" direction
            cosAngle=max(-1,min(1,velDir.dot(rappelDir))) # to avoid pathological cases where velDir.dot(rappelDir)>1 or <-1 due to numerical error
            angle=np.arccos(cosAngle)
            # set the fraction of the correction imposed 
            # (velocity is set in the "goHome" direction when the current 
            # distance is twice the threshold distance)
            adjustIntensity=min(1,(d2**0.5-farFromHomeDist)/farFromHomeDist)
            angle=(1-adjustIntensity)*angle
            # make the velocity vector turn towards the "goHome" direction
            if velDir.dot(perpRappelDir)>0:
                newVel=O.bodies[i].state.vel.norm()*(np.cos(angle)*rappelDir+\
                                                     np.sin(angle)*perpRappelDir)
            else:
                newVel=O.bodies[i].state.vel.norm()*(np.cos(angle)*rappelDir-\
                                                     np.sin(angle)*perpRappelDir)
            O.bodies[i].state.vel=newVel

def contactCheck():
    """
    exchange viral charge
    """
    # Objects to update
    global infectionTable
    # contagious law parameters
    global d_contagion,coughRate,coughVirusFrac,virusGrowth
    global V_infected,V_contagious,V_recovery
    global p_symptomatic
    # to easily recover particle ids in case non periodic BC are used
    global NSetOfWalls
    # social behavior
    global carefulIfSick
    
    # if a contact is detected, assume it existed since last call to cCheck
    dtCheck=cCheck.iterPeriod*O.dt 
    # initialize daily increments of virus
    dC_self={} # self production of virus
    dC_out={} # expelled virus quantities
    dC_in={}  # received virus quantities
    # update self virus production
    for i in infectionTable.keys():
        agent=infectionTable[i]
        # virus production according to the immune activation degree of the agent
        # dC_self/dtCheck=(virusGrowth-immuneEfficiency*L)*C-virusGrowth*V_infected
        dC_self[i]=(virusGrowth-agent.immuneEfficiency*agent.immuneActivation)*\
                   agent.virusCharge*dtCheck-virusGrowth*V_infected*dtCheck
        # update immune activation
        # dL/dtCheck=immuneActSpeed*C*(1-L)
        dL=agent.immuneActSpeed*agent.virusCharge*\
           (1-agent.immuneActivation)*dtCheck
        agent.immuneActivation=min(1,agent.immuneActivation+dL)
        # virus excretion
        if agent.infectionStatus=='contagious' or\
           agent.infectionStatus=='symptomatic' or\
           agent.infectionStatus=='asymptomatic':
            dC_out[i]=(1-agent.protecOut)*coughVirusFrac*coughRate*\
                      agent.virusCharge*dtCheck
        else:
            dC_out[i]=0
        # initialize dC_in
        dC_in[i]=0
    # check all interactions
    for i in O.interactions:
        tag=(i.id1,i.id2,i.iterMadeReal)
        # interagent distance
        d=O.bodies[i.id1].shape.radius+O.bodies[i.id2].shape.radius-i.geom.penetrationDepth
        # if the two bodies are agents and close enough exchange some virus
        if i.id1>=4*NSetOfWalls and i.id2>=4*NSetOfWalls and d<d_contagion:
            agent1=infectionTable[i.id1]
            agent2=infectionTable[i.id2]
            dC_in_id1=(1-agent1.protecIn)*(1-d/d_contagion)*dC_out[i.id2]
            dC_in_id2=(1-agent2.protecIn)*(1-d/d_contagion)*dC_out[i.id1]
            dC_in[i.id1]+=dC_in_id1
            dC_in[i.id2]+=dC_in_id2
            # update contact statistics if recorded
            if contactStats.dead==False:
                if tag in agent1.contactTags:
                    j=agent1.contactTags.index(tag)
                    agent1.contactdCin[j]+=dC_in_id1
                if tag in agent2.contactTags:
                    j=agent2.contactTags.index(tag)
                    agent2.contactdCin[j]+=dC_in_id2
            # check whether the contact proved to be contagious
            if V_infected-dC_in_id1<agent1.virusCharge<V_infected:
                agent2.infectedIds.add(i.id1)
            elif V_infected-dC_in_id2<agent2.virusCharge<V_infected:
                agent1.infectedIds.add(i.id2)
    
    # update virus charges and infection status
    for i in infectionTable.keys():
        agent=infectionTable[i]
        agent.virusCharge=max(0,agent.virusCharge+dC_in[i]+dC_self[i]-dC_out[i])
        
        infectionTest = agent.infectionStatus=='healthy' and\
                        agent.virusCharge>V_infected
        contagiousTest = agent.infectionStatus=='infected' and\
                         agent.virusCharge>=V_contagious
        symptomaticTest = agent.infectionStatus=='contagious' and \
                          O.time-agent.contagiousTime>2*dayTime
        recoveredTest = ((agent.infectionStatus=='asymptomatic' or\
                         agent.infectionStatus=='symptomatic') and\
                        agent.virusCharge<=V_recovery and \
                        agent.virusCharge<agent.maxVirusCharge) or \
                        ((agent.infectionStatus=='infected') and \
                        (agent.virusCharge<=0.))   
        
        if infectionTest:
            O.bodies[i].shape.color[0]=1 # from green to yellow if initially healthy
            agent.infectedTime=O.time
            agent.infectionStatus='infected'
        elif contagiousTest:
            agent.contagiousTime=O.time
            agent.infectionStatus='contagious'
            O.bodies[i].shape.color=(1,0,0) # red
        elif symptomaticTest:
            if random.random()<p_symptomatic:
                agent.infectionStatus='symptomatic'
                agent.symptomatic=1
                O.bodies[i].shape.color=(1,0,1) # magenta
                # enforce social distancing for symptomatic contagious people
                if carefulIfSick==1:
                    O.bodies[i].material=O.materials['careful']
                    # + wear mask
                    agent.protecOut=0.8
            else:
                agent.infectionStatus='asymptomatic'
        elif recoveredTest:
            agent.recoverTime=O.time
            agent.infectionStatus='recovered'
            O.bodies[i].shape.color=(0,1,1) # cyan

        # look for the peak of contagiousness
        if agent.virusCharge>agent.maxVirusCharge:
            agent.maxVirusCharge=agent.virusCharge
            if agent.infectionStatus!='healthy':
                agent.peakContagiousTime=O.time    
       
def saveData(filename):
    """
    save the simulation data in txt files
    """
    # Objects to save
    global infectionTable,dailyStats
    # time parameters
    global dayTime,nCheckPerContact
    # Initial set up parameters
    global Nparts,phi,m_target,initInfectedFract
    # Social distancing parameters
    global D_interaction,careFrac,Ncd,dmin_care,dmin_reck
    # sociological parameters
    global N_neighbours,longRangeDailyFrac,popStatsDict
    # Individual protection
    global protec_in,protec_out
    # confinement parameters
    global confStartTime,confEndTime,confFrac
    # Contagion law parameters
    global d_contagion,coughRate,V_infected,V_contagious,V_recovery,coughVirusFrac
    # Virus concentration evolution law
    global incubationTime,immuneActTime,recoveryTime
    # Probabilities for complications
    global p_symptomatic
    # for daily contact statistics 
    global contactStats
    
    with open(filename,'w') as f:
        f.write('# Simulation run with the following parameters:\n')
        f.write('# Time parameters:\n')
        f.write('# dayTime,nCheckPerContact = '+str(dayTime)+','+str(nCheckPerContact)+'\n') 
        f.write('# Initial set up parameters:\n')
        f.write('# Nparts,phi,m_target,initInfectedFract = '+\
                str(Nparts)+','+str(phi)+','+str(m_target)+','+\
                str(initInfectedFract)+'\n')
        f.write('# Social distancing parameters:\n')
        f.write('# D_interaction,careFrac,Ncd,dmin_care,dmin_reck,carefulIfSick = '+\
                str(D_interaction)+','+str(careFrac)+','+str(Ncd)+','+\
                str(dmin_care)+','+str(dmin_reck)+','+str(carefulIfSick)+'\n')
        f.write('# Individual protection parameters:\n')
        f.write('# protec_in,protec_out = '+\
                str(protec_in)+','+str(protec_out)+'\n')
        f.write('# Confinement parameters:\n')
        f.write('# confStartTime,confEndTime,confFrac = '+str(confStartTime)+\
                ','+str(confEndTime)+','+str(confFrac)+'\n')
        f.write('# Contagion law parameters:\n')
        f.write('# d_contagion,coughRate,V_infected,V_contagious,V_recovery,coughVirusFrac = '+\
                str(d_contagion)+','+str(coughRate)+','+str(V_infected)+','+\
                str(V_contagious)+','+str(V_recovery)+','+str(coughVirusFrac)+'\n') 
        f.write('#  Virus concentration evolution law:\n')
        f.write('# incubationTime,immuneActTime,recoveryTime = '+\
                str(incubationTime)+','+str(immuneActTime)+','+\
                str(recoveryTime)+'\n') 
        f.write('#  Probabilities for complications:\n')
        f.write('# p_symptomatic = '+str(p_symptomatic)+'\n') 
        f.write('#  Sociological parameters:\n')
        f.write('# longRangeDailyFrac = '+str(longRangeDailyFrac)+'\n')
        f.write('# N_neighbours, popStatsDict : '+repr(popStatsDict)+','+\
                repr(popStatsDict)+'\n')
        f.write('id\tinfectedTime\tcontagiousTime\tpeakContagiousTime\trecoverTime\t'+\
                'symptomatic\tinfectedIds\timmuneActivation\tsocialDistancing\t'+\
                'protecIn\tprotecOut\tageCat\tBIMCat\n')      
        for i in infectionTable.keys():
            #string='%i '%(infectionTable[i].id)'\t'
            string='%i '%(infectionTable[i].id)+'\t'
            string+=str(infectionTable[i].infectedTime)+'\t'
            string+=str(infectionTable[i].contagiousTime)+'\t'
            string+=str(infectionTable[i].peakContagiousTime)+'\t'
            string+=str(infectionTable[i].recoverTime)+'\t'
            string+=str(infectionTable[i].symptomatic)+'\t'
            string+=repr([i for i in infectionTable[i].infectedIds])+'\t'
            string+=str(infectionTable[i].immuneActivation)+'\t'
            string+=str(infectionTable[i].socialDistancing)+'\t'
            string+=str(infectionTable[i].protecIn)+'\t'
            string+=str(infectionTable[i].protecOut)+'\t'
            string+=str(infectionTable[i].ageCategory)+'\t'
            string+=str(infectionTable[i].BMICategory)
            f.write(string+'\n')
            #print(str(i)+' ' + str(infectionTable[i].id))
        f.close()   
    # if daily statistics are recorded
    if contactStats.dead==False:
        with open(filename.replace('.txt','_contactStats.txt'),'w') as f:
            print('contact stats ok')
            keys=dailyStats.keys()
            s=''
            for k in keys:
                s+=k+' '
            f.write(s[:-1]+'\n')
            for i in range(len(dailyStats[list(keys)[0]])):
                s=''
                for k in keys:
                    s+=str(dailyStats[k][i])+' '
                f.write(s[:-1]+'\n')
    print('file '+filename+' saved')
    
###############################################################################
# Print some estimated parameters of the simulation
###############################################################################
O.run(1,1)
totEnergy=O.energy['elastPotential']+O.energy['kinetic']
# the typical time of interaction between two particles is half of the harmonic
# oscillator period which is pi/w with w=(2kn/m)**0.5. 
m=4./3*np.pi*R_interaction**3*O.materials['reckless'].density
k_reck_reck=E_reckless*R_interaction
k_reck_care=2*E_reckless*E_careful/(E_careful+E_reckless)*R_interaction
k_care_care=E_careful*R_interaction
w_reck_reck=(2*k_reck_reck/m)**0.5
w_reck_care=(2*k_reck_care/m)**0.5
w_care_care=(2*k_care_care/m)**0.5
print('-----------------------------------------------------------------------')
print('Typical contact times for reck/reck, reck/care and care/care people are:')
print('%0.3f h, %0.3f h and %0.3f h'\
      %(np.pi/w_reck_reck,np.pi/w_reck_care,np.pi/w_care_care))
print('While contact is checked every %0.3f h'%(checkTime))
# For two particles with a typical kinetic energy (Ecpart) equal to the total 
# energy (Ecpart) divided by the number of particles, a frontal collision 
# will correspond to a maximal depth of 2vp/w where vp=(2*Ecpart/m)**0.5 and 
# w=(2kn/m)**0.5
Ecpart=totEnergy/Nparts
vp=1/np.sqrt(2)*(2*Ecpart/m)**0.5
dmin_reck_reck=2*(R_interaction-vp/w_reck_reck)
dmin_reck_care=2*(R_interaction-vp/w_reck_care)
dmin_care_care=2*(R_interaction-vp/w_care_care)
print('w_reck_reck='+str(w_reck_reck))
print('-----------------------------------------------------------------------')
print('Typical minimum distances reck/reck, reck/care and care/care people are:')
print('%0.2f m, %0.2f m and %0.2f m'\
      %(dmin_reck_reck,dmin_reck_care,dmin_care_care))
print('While virus propagation starts for d<%0.2f m'%(d_contagion))
# For a contact for which the penetration depth is sufficient to allow for 
# contagion, the period during which the two people are close enough is 
# estimated as 2(pi/(2w)-t0) with t0=arcsin(w(2R_interaction-d_contagion)/2vp)/w
# This comes from the solution of an harmonic oscillator with initial speed 2vp
if dmin_reck_reck<d_contagion:
    t0_reck_reck=np.arcsin(w_reck_reck*(2*R_interaction-d_contagion)/(2*vp))/w_reck_reck
    tcontagion_reck_reck=2*(np.pi/(2*w_reck_reck)-t0_reck_reck)
else:
    tcontagion_reck_reck=0
if dmin_reck_care<d_contagion:
    t0_reck_care=np.arcsin(w_reck_care*(2*R_interaction-d_contagion)/(2*vp))/w_reck_care
    tcontagion_reck_care=2*(np.pi/(2*w_reck_care)-t0_reck_care)
else:
    tcontagion_reck_care=0
if dmin_care_care<d_contagion:
    t0_care_care=np.arcsin(w_care_care*(2*R_interaction-d_contagion)/(2*vp))/w_care_care
    tcontagion_care_care=2*(np.pi/(2*w_care_care)-t0_care_care)
else:
    tcontagion_care_care=0
print('-----------------------------------------------------------------------')
print('Typical transmission times for reck/reck, reck/care and care/care are:')
print('%0.3f h, %0.3f h and %0.3f h'\
      %(tcontagion_reck_reck,tcontagion_reck_care,tcontagion_care_care))
print('While contact is checked every %0.3f h'%(checkTime))
# The virus charge variations during a contact are roughly estimated for a
# unit virus charge and an active immune system
dC_out_reck_reck=(1-protec_out)*coughVirusFrac*coughRate*tcontagion_reck_reck
dC_out_reck_care=(1-protec_out)*coughVirusFrac*coughRate*tcontagion_reck_care
dC_out_care_care=(1-protec_out)*coughVirusFrac*coughRate*tcontagion_care_care

dC_in_reck_reck=(1-protec_in)*dC_out_reck_reck*(1-dmin_reck_reck/d_contagion)
dC_in_reck_care=(1-protec_in)*dC_out_reck_care*(1-dmin_reck_care/d_contagion)
dC_in_care_care=(1-protec_in)*dC_out_care_care*(1-dmin_care_care/d_contagion)

dC_self_reck_reck=(virusGrowth-virusDeath)*tcontagion_reck_reck
dC_self_reck_care=(virusGrowth-virusDeath)*tcontagion_reck_care
dC_self_care_care=(virusGrowth-virusDeath)*tcontagion_care_care

print('-----------------------------------------------------------------------')
print('Typical virus charge variations during reck/reck, reck/care and ')
print('care/care contacts for a unit virus charge and an operational immune ')
print('system:')
print('dC_self = %0.2e, %0.2e, %0.2e'\
      %(dC_self_reck_reck,dC_self_reck_care,dC_self_care_care))
print('dC_in = %0.2e, %0.2e, %0.2e'\
      %(dC_in_reck_reck,dC_in_reck_care,dC_in_care_care))
print('dC_out = %0.2e, %0.2e, %0.2e'\
      %(dC_out_reck_reck,dC_out_reck_care,dC_out_care_care))
print('-----------------------------------------------------------------------')
print('If these values suit you, you may want to start running the simulation')
print('but do not forget to check contact statistics a posteriori !')
print('-----------------------------------------------------------------------')
print('contactStats.dead'+str(contactStats.dead))
with open('results/'+simuName +'/'+simuName + 'time_stat.txt','w') as f:
        f.write('Day    nInfected    nSick    nRecovered')
if initInfectedFract == 0.:
    O.run(int(100*24/O.dt,wait=1))
else:
    O.run(-1,True)
print('wesh wesh yo la kiffance')

saveData('results/'+simuName+'/'+simuName+'.txt')
utils.waitIfBatch()