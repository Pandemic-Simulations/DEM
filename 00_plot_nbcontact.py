import numpy as np
import random
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression


chemin = 'C:/Users/mmari/OneDrive - University of Calgary/simulation/00_calibration/00_density_and_V0/results'
#noms_dossiers = [nom for nom in os.listdir(chemin) if os.path.isdir(os.path.join(chemin, nom))]

noms_dossiers = ['A_20250211T180737p2895400000_929','A_20250206T160115p31258790000_506', 'A_20250206T160115p31258790000_619', 'A_20250206T160116p31258880000_198', 'A_20250206T160116p31258880000_387', 'A_20250206T160116p31258940000_111', 'A_20250206T160116p31258940000_619', 'A_20250206T160117p31259050000_320', 'A_20250206T160117p31259050000_418', 'A_20250206T160118p31259130000_144', 'A_20250206T160118p31259130000_332', 'A_20250206T160118p31259200000_450', 'A_20250206T160118p31259200000_596', 'A_20250206T160119p31259280000_406', 'A_20250206T160119p31259280000_733', 'A_20250206T160119p31259310000_383', 'A_20250206T160119p31259310000_388', 'A_20250206T160120p31259380000_265', 'A_20250206T160120p31259380000_399', 'A_20250206T160121p31259490000_107', 'A_20250206T160121p31259490000_180', 'A_20250206T160121p31259550000_499', 'A_20250206T160331p10190370000_134', 'A_20250206T160331p10190400000_955', 'A_20250206T160332p10190660000_369', 'A_20250206T160332p10190740000_600', 'A_20250206T160333p10190820000_378', 'A_20250206T160334p10190910000_339', 'A_20250206T160334p10190940000_660', 'A_20250206T160335p10191020000_159', 'A_20250206T160335p10191100000_763', 'A_20250206T160336p10191180000_906', 'A_20250206T160336p10191260000_967', 'A_20250211T180736p2895370000_150', 'A_20250211T180736p2895370000_813', 'A_20250211T180737p2895400000_358', 'A_20250211T180737p2895470000_530', 'A_20250211T180737p2895470000_539', 'A_20250211T180738p2895580000_138', 'A_20250211T180738p2895580000_611', 'A_20250211T180738p2895640000_917', 'A_20250211T180738p2895640000_951', 'A_20250211T180739p2895670000_813', 'A_20250211T180739p2895670000_942', 'A_20250211T180739p2895800000_262', 'A_20250211T180739p2895800000_667', 'A_20250211T180740p2895880000_171', 'A_20250211T180740p2895880000_866', 'A_20250211T180740p2895950000_264', 'A_20250211T180740p2895950000_824', 'A_20250211T180741p2896040000_249', 'A_20250211T180741p2896040000_611']


['A_20250211T180739p2895670000_942','A_20250211T180739p2895670000_813']

n=len(noms_dossiers)


with open('list_StateA.txt','r') as f:
    listStateA = f.read().split('\n')
listStatesA = [row.split('\t') for row in listStateA]
nStatesA = len(listStateA)
Npart = []
phi = []
Ncd = []
contactmoy = []
contactMin = []
contactMax = []
vinit = []

for i in range (0,n):
    #print(noms_dossiers[i])
    stateA = noms_dossiers[i][0:-8]
    chercheState = True 
    iState=1
    while chercheState and iState<nStatesA:
        stateAi = listStatesA[iState][0]
        if stateAi == stateA:
            chercheState = False
            Nparti = float(listStatesA[iState][2])
            phii = float(listStatesA[iState][3])
            Ncdi = float(listStatesA[iState][5])
            D_interi = float(listStatesA[iState][4])
            lx = D_interi/2 * np.sqrt(Nparti*np.pi/phii) 
            vinit +=[np.sqrt(2)*(lx/np.sqrt(Nparti)-D_interi)*Ncdi/24]
            Npart += [Nparti]
            phi += [phii]
            Ncd += [Ncdi]
        else:
            iState = iState+1
    if chercheState:
        print('error in list of state A name with simulation ',noms_dossiers[i][0:25])
    file = chemin + '/' + noms_dossiers[i] + '/' + noms_dossiers[i] + '_contactStats.txt'
    with open(file,'r') as f:
        evo_contact_mean = f.read().split('\n')
    evo_contact = [row.split() for row in evo_contact_mean]
    #print(evo_contact)
    n_evo_contact = len(evo_contact_mean)
    mean = 0
    nb = 0
    mini = 1000000
    maxi = 0
    for iEvo in range (1, n_evo_contact-10):
        #print(evo_contact[iEvo][0])
        mean += float(evo_contact[iEvo][0])
        nb += 1
        if float(evo_contact[iEvo][0]) < mini:
            mini = float(evo_contact[iEvo][0])
        if float(evo_contact[iEvo][0]) > maxi:
            maxi = float(evo_contact[iEvo][0])
        
    contactmoy += [mean / nb]
    contactMax += [maxi]
    contactMin += [mini]
    print('simu: ',noms_dossiers[i],'  phi: ',listStatesA[iState][3],'  Ncd: ',listStatesA[iState][5],'  contact mean: ',mean / nb)
    
list_phi = [phi[0]]
vel_phi = [[vinit[0]]]
mean_phi = [[contactmoy[0]]]
max_phi = [[contactMax[0]]]
min_phi = [[contactMin[0]]]
Ncd_phi = [[Ncd[0]]]

for i in range (1,len(phi)):
    nj = len(list_phi)
    find = False
    j = 0
    phi_i = phi[i]
    while (j<nj) and (find == False):
        phi_j = list_phi[j]
        if phi_i == phi_j:
            mean_phi[j]+=[contactmoy[i]]
            max_phi[j]+=[contactMax[i]]
            min_phi[j]+=[contactMin[i]]
            Ncd_phi[j]+=[Ncd[i]]
            vel_phi[j]+=[vinit[i]]
            
            find = True
        else:
            j += 1
    if (find==False):
        list_phi += [phi_i]
        mean_phi += [[contactmoy[i]]]
        max_phi += [[contactMax[i]]]
        min_phi += [[contactMin[i]]]
        Ncd_phi += [[Ncd[i]]]
        vel_phi += [[vinit[i]]]
        
        
colors = ['b','g','r','m','c','b','g','r','m','c']
        
fig, (ax) = plt.subplots(figsize=(9,6),ncols=1)
plt.rcParams['text.usetex'] = True    
ax.set_xlabel('Ncd')
ax.set_ylabel('Number of contacts')

fig2, (ax2) = plt.subplots(figsize=(9,6),ncols=1)   
ax2.set_xlabel('Initial velocity ($m.h^{-1}$)')
ax2.set_ylabel('Number of contacts')
for i in range (0,len(list_phi)):
    x1 = np.array(Ncd_phi[i]).reshape((-1, 1))
    x2 = np.array(vel_phi[i]).reshape((-1, 1))
    y = np.array(mean_phi[i])
    model1 = LinearRegression().fit(x1, y)
    model2 = LinearRegression().fit(x2, y)
    r_sq1 = model1.score(x1, y)
    r_sq2 = model2.score(x2, y)
    y1_pred = model1.predict(x1)
    y2_pred = model2.predict(x2)
    x00 =np.array([0]).reshape((-1,1))
    x01 =np.array([1]).reshape((-1,1))
    y10=model1.predict(x00)
    y11=model1.predict(x01)
    y20=model2.predict(x00)
    y21=model2.predict(x01)
    eq1 = ' y = '+str(y11[0]-y10[0])+'x+'+str(y10[0])
    eq2 = ' y = '+str(y21[0]-y20[0])+'x+'+str(y20[0])
    ax.plot(Ncd_phi[i],mean_phi[i],label=r'$\phi = '+str(list_phi[i])+'$',color=colors[i],linestyle='None',marker='x')
    ax.plot(x1,y1_pred,label=r'linear regression $\phi = '+str(list_phi[i])+'\ \ r='+str(r_sq1)+'\ \ '+ eq1+'$',color=colors[i],linestyle='-')
    ax2.plot(vel_phi[i],mean_phi[i],label=r'$\phi = '+str(list_phi[i])+'$',color=colors[i],linestyle='None',marker='x')
    ax2.plot(x2,y2_pred,label=r'$\phi = '+str(list_phi[i])+'\ \ r='+str(r_sq2)+'\ \ '+ eq2+'$',color=colors[i], linestyle = '-')

plt.grid()        
plt.legend()
plt.show()
    
    
    