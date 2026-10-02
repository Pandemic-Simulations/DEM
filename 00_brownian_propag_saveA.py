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
###############################################################################
# List of parameters - initial disposition parameters 
###############################################################################
readParamsFromTable(phi = 0.5,Nparts=10000,Ncd=40)
from yade.params import table



# general parameters
dayTime = 24
Nparts = table.Nparts             # number of people
NpartsInit = Nparts
phi = table.phi                 # particle density (between 0 and pi/4)
m_target = 50.

# social interaction
D_interaction = 2.        # m social interaction distance
Ncd = table.Ncd #40#25            # estimated number of daily contacts (needs back fitting)
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
utils.waitIfBatch()
   