###############################################################################
# Model description : virus concentration evolution in brownian motion
# basic version: 05/2020 by Antoine
# modification (separation): 11-12/2023 by Marie
###############################################################################
# General
# Virus need close contact between people. No propagation is possible unless 
# people get closer than a certain distance (d_contagion). When two people are 
# close enough durint a small time dt and one of them is sick (his virus 
# concentration C is above a threshold C_sick), the non sick receive a small 
# virus charge dC_in which is proportional to (1-d/d_contagion) with d being 
# the current distance between the two people, to the time of contact dt and to 
# the virus charge rejected by the sick person. This person reject a virus 
# concenration dC_out proportional to its virus charge C and to the frequency 
# at which he cough (coughrate). In addition to all of that, the sick and non
# sick can limit the risk of virus exchange by using individual protections.
# This is taken into account by additional factors (1-protec_in) and 
# (1-protec_out). All in all, the exchange of virus id given by
# dC_in=(1-protec_in)*(1-d/d_contagion)*dC_out
# dC_out=(1-protec_out)*coughRate*C*dt if C>C_sick
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
# File separation - Marie (11/2023)
# To be able to to oberve the effect of each parameters independantly, each 
# simulation is run in several steps:
#  A- the position of the particles and their initial velocity direction is 
#     defined and saved (A)
#  B- the demography of the population is defined, and  agents are associated
#     yade's particles. The state of the simulation is saved (B)   
#  C- depending on the initial number of infected cases, agents' health status
#     are randomly attributed. The state of the simulation is saved (C)
#  D- the general infectious parameters are defined and applied in the
#     simulation. The simulation is run and results are saved.   
# It enables to test different parameters, populations and initial status, with
# the same initial conditions.  
###############################################################################
# File A:
# In this file, the initial disposition (positions of particles and initial 
# directions) is defined and saved.
###############################################################################
import numpy as np
import random
import os
import pickle

readParamsFromTable(setPop = '1',m_target=50.,Nparts=10000,phi=0.5,Ncd=40)
from yade.params import table


###############################################################################
# List of parameters - initial disposition parameters 
###############################################################################
# general parameters
dayTime = 24              # day time in h
Nparts = table.Nparts            # number of people
NpartsInit = table.Nparts
phi = table.phi                 # particle density (between 0 and pi/4)
m_target = table.m_target
setPop = table.setPop

# social interaction
D_interaction = 2.        # m social interaction distance
Ncd = table.Ncd            # estimated number of daily contacts (needs back fitting)
###############################################################################
# Automatic definition of parameters
###############################################################################
R_interaction = D_interaction/2
Lx = R_interaction*np.sqrt(Nparts*np.pi/phi)          # m x m size of the box
Ly = Lx                             
Vref=(np.sqrt(Lx*Ly/Nparts)-2*R_interaction)*Ncd/dayTime     # m/h mean speed
Vinit=np.sqrt(2)*Vref                                 # m/h initial speed. 
###############################################################################
# Une periodic boundary conditions
###############################################################################
periodicBC=True
###############################################################################
#pseudo material (to get the good mass)
rho=(3*m_target)/(4*np.pi*R_interaction**3)
O.materials.append(FrictMat(density=rho,label='neutre'))
###############################################################################
# give an automatic name to the simu
simuName='A_' + O.tags['id']
print(simuName)
os.mkdir(simuName)                                                      # The square root has a meaning but I do not remember
###############################################################################
# set up the agents without initial contacts on a grid cell of size 
# 2*R_interaction x 2*R_interaction
if Nparts>int(Lx/(2*R_interaction))*int(Ly/(2*R_interaction)):
    Nparts=int(Lx/(2*R_interaction))*int(Ly/(2*R_interaction))
    print('Too many agents, number has been reduced to %i'%(Nparts))
# set up the list of possible (x,y) coordinates
coordsIds=[]
for i in range(int(Lx/(2*R_interaction))):
    coordsIds+=[(i,j) for j in range(int(Ly/(2*R_interaction)))]
    
for i in range(Nparts):
    # place the agent randomly in the box (no overlapping)
    xyId=random.randrange(len(coordsIds))
    xy=coordsIds.pop(xyId)
    x=2*R_interaction*xy[0]+1
    y=2*R_interaction*xy[1]+1

    
    # add the agent to the simulation
    O.bodies.append(sphere(center=(x,y,2*D_interaction),radius=R_interaction,
                           color=[0,1,0],material='neutre'))

    # give the agents an initial random velocity direction
    theta=random.random()*2*np.pi
    O.bodies[-1].state.vel[0]=Vinit*np.cos(theta)
    O.bodies[-1].state.vel[1]=Vinit*np.sin(theta)

# record the energies (just to check whether no dissipation occur).
O.trackEnergy=True

fileName = 'list_StateA.txt' 

if not os.path.isfile(fileName):
    with open(fileName,'a') as f:
        f.write('simulation\tNpartsInit\tNpartsReal\tphi\tD_interact'
                +'\tNcd\n')
        f.close()

with open(fileName,'a') as f:
    f.write(simuName+'\t'+str(NpartsInit)+'\t'+str(Nparts)+'\t'+str(phi)+'\t'
            +str(D_interaction)+'\t'+str(Ncd)+'\n')
    f.close()

O.save(simuName + '.yade')

nCheckPerContact=10       # number of contact checks during a typical contact 
                          # duration
                          
## Popuilation
picklePopName = 'pickles/pop_set' + setPop + str(Nparts) + '.pickle'
# demographic characteristics 
popStatsDict = {'ageCat':        ['young','adult','old'],
               'popFrac':       [0.1539  ,0.3605   ,0.1345]}
# Travelling parameters
longRangeDailyFrac = 0.01                        # fraction of people travelling far away each day
N_neighbours = {'young':15,'adult':30,'old':10}  # accessible neighbours within a day
dmin_reck=0.5
E = m_target*Vref**2/(2*R_interaction*(R_interaction-dmin_reck/2)**2)

# check contacts sufficiently frequently 
w_care = (2*E*R_interaction/m_target)**0.5
checkTime = 1./nCheckPerContact*np.pi/w_care # the typical time of interaction 
                                             # is half of the harmonic 
                                             # oscillator period which is 
                                             # pi*(m/2kn)**0.5. 

# distance from which a coming back force is applied
farFromHomeDist={k:np.sqrt(N_neighbours[k]/phi)*R_interaction \
                 for k in N_neighbours.keys()}
                 
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
        self.ageCategory=''
        self.homePos=[]
        self.farFromHomeDist=0)
        # for daily statistics
        self.contactTags=[]
        self.contactTimes=[]
        self.contactMinDists=[]
        self.contactMaxVels=[]
        self.contactdCin=[]
        self.dayInitPos=[]

    def dict(self):
        print('     ageCategory = ',self.ageCategory)

infectionTable={} # to record all data on agents virus charge

###############################################################################
# define two materials, one for reckless, one for careful people
materialNames=['people']
rho=(3*m_target)/(4*np.pi*R_interaction**3)
walls = O.materials.append(FrictMat(density=rho,young=E,poisson=0.25,
                                    frictionAngle=radians(0),label='walls'))
people = O.materials.append(FrictMat(density=rho,young=E,poisson=0.25,
                                      frictionAngle=radians(0),label='peoplel'))

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
                                   
# Define the age and BMI of all agents
popStatus=[]
for i in range(len(popStatsDict['ageCat'])):
    cat=popStatsDict['ageCat'][i]
    Ncat=int(Nparts*popStatsDict['popFrac'][i])
    popStatus+=[(cat) for j in range(Nparts)]
# add a few standart agents
popStatus+=[('adult','normal') for j in range(len(popStatus),Nparts)]
orderPop = pickle.load(open(picklePopName, 'rb'))

for i in range(Nparts):
    agent=Agent()
    
    # add the agent to the simulation
    agent.id=orderPop[i]
    
    x = O.bodies[agent.id].state.refPos[0]
    y = O.bodies[agent.id].state.refPos[1]
    
    O.bodies[agent.id].material=O.materials['people']

    # select age and BIM status
    agent.ageCategory=popStatus[agent.id][0]
    
    # set home adress and range of travel
    agent.farFromHomeDist=farFromHomeDist[agent.ageCategory]
    agent.homePos=[x,y]
        
    # record the initial position
    agent.dayInitPos=[[x,y]]

    infectionTable[agent.id]=agent
    
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

dailyStats={'nC':[],'nC_positive':[],
            'meanCTime':[],'stdCTime':[],
            'meanMaxVel':[],'meanDailyDists':[],
            'meanMinDist':[],'stdMinDist':[],
            'meandCin':[],'stddCin':[],
            'totEnergy':[]
            }
            
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
        cMinDists=[],[],[]
        # viral charge exchanged (only if non zero)
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
                if agent.contactMinDists[j]<=0:
                    print('Problem here, interactions between agent '+str(agent.id)+ 'and '+str(j) +'gives negative distance')
                cMinDists+=[agent.contactMinDists[j]]
                    if agent.contactdCin[j]>0:
                        dCin+=[agent.contactdCin[j]]
                        
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
        dailyStats['meanMinDist]+=[np.mean(cMinDists)]
        dailyStats['stdMinDist']+=[np.std(cMinDists)
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
        dmin=dailyStats['meanMinDist'][-1]-\
                      2*dailyStats['stdMinDist'][-1]
        s='Daily min dists is: '
        s+='%0.2f m,%0.2f m,%0.2f m'%(dmin)
        print(s)

        print(dailyStats['meanMinDist'][-1])
        print(dailyStats['stdMinDist'][-1])
        with open('picklecMinDists_ReckReck', 'wb') as handle:
            pickle.dump(cMinDists_ReckReck, handle, protocol=2)
    # try to enforce confinement
    enforceConfinement()
    
    # try to make people travel if confinement is not on

    makeTravelPeople()
    
    #if (nSick==0 and nInfected==0): # or (O.iter == 100):
    #    O.pause()
    

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

