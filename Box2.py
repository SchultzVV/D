import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn as nn
import pickle
import random as rd
import torch.optim as optim
import numpy as np;import sys as s
import matplotlib.pyplot as plt
import math

#           GERANDO O BANCO DE DADOS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def Box_2_dataset(n_batch,batch_size,interval_size):
    T=[i for i in range(0,interval_size)]
    O=[];    Q=[];    A=[];    J=[]
    v_min,v_max=5,40
##    v_min,v_max=0,3
    velocidades=np.linspace(v_min,v_max,num=100)
    for i in range(n_batch*batch_size):
        v_rand1=rd.randint(len(velocidades)/2,len(velocidades)-1)
        v_rand2=rd.randint(0,len(velocidades)-1-len(velocidades)/2)
        v_free_i=velocidades[v_rand1]
        v_rot_i=velocidades[v_rand2]
        J1=velocidades[v_rand1]-velocidades[v_rand2]
        J.append(J1)
        random_list=np.linspace(0,J1,num=20)
        aux_rand=rd.randint(0,len(random_list)-1)
        v_free_f=random_list[aux_rand]
        v_rot_f=J1-v_free_f
        #----------------------------------------------------------
        q_rot_i   = [velocidades[v_rand2]*(len(T)-i-1) for i in T]
        q_free_i  = [-velocidades[v_rand1]*(len(T)-i-1) for i in T]
        q_free_f  = [v_free_f*(i) for i in T]
        q_rot_f   = [v_rot_f*(i) for i in T]
##        print('v_free_i = ',v_free_i)
##        print('v_rot_i = ',v_rot_i)
##        print('J1 = ',J1)
##        print('v_free_f = ',v_free_f)
##        print('v_rot_f = ',v_rot_f)
##        print('q_free_i = ',q_free_i)
##        print('q_rot_i = ',q_rot_i)
##        print('q_free_f = ',q_free_f)
##        print('q_rot_f = ',q_rot_f)
        tpred=rd.randint(0,interval_size-1)
        q=[tpred]
        a=0
        for i in q_free_f:
            q.append(T[a])
            a+=1
            q.append(i)
        Q.append(q)
        a=[q_rot_f[tpred]]
        A.append(a)
        o=[];    a=0
        for i in q_rot_i:
            o.append(T[a])
            a+=1
            o.append(i)
        a=0
        for i in q_free_i:
            o.append(T[a])
            a+=1
            o.append(i)
        O.append(o)
    O=np.array(O).reshape(n_batch,batch_size,interval_size*4)
    Q=np.array(Q).reshape(n_batch,batch_size,interval_size*2+1)
    A=np.array(A).reshape(n_batch,batch_size,1)
    J=np.array(J).reshape(n_batch,batch_size,1)
    O=torch.as_tensor(O);    Q=torch.as_tensor(Q);    A=torch.as_tensor(A)
    print('np.shape(J)',np.shape(J));    print('np.shape(O)',np.shape(O))
    print('np.shape(Q)',np.shape(Q));    print('np.shape(A)',np.shape(A))
##    s.exit()
    address = open("O_train","wb");    pickle.dump(O, address);    address.close()
    address = open("Q_train","wb");    pickle.dump(Q, address);    address.close()
    address = open("A_train","wb");    pickle.dump(A, address);    address.close()
    address = open("J_train","wb");    pickle.dump(J, address);    address.close()
#Box_2_dataset(5,500,5)
#s.exit()
#-------------------------------------------------------------------------------
#-----------------LOAD DATA-----------------------------------------------------
#-------------------------------------------------------------------------------
inp     = pickle.load(open("O_train","rb"))
question= pickle.load(open("Q_train","rb"))
out     = pickle.load(open("A_train","rb"))
J       = pickle.load(open("J_train","rb"))
n_batch=np.shape(inp)[0]
batch_size=np.shape(inp)[1]
n_examples=np.shape(inp)[2]
Q_shape=np.shape(question)
#-------------------------------------------------------------------------------
#------------------DEFINE O MODELO----------------------------------------------
#-------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        # N, 50
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_examples,400),
            nn.ELU(),
            nn.Linear(400,300),
            nn.ELU(),
            nn.Linear(300,200),
            nn.ELU(),
            nn.Linear(200,1), # -> latent
            nn.ELU(),
        )
        self.project=nn.Linear(Q_shape[2],150)
        self.decoder=nn.Sequential(
            nn.Linear(151,200),
            nn.ELU(),
            nn.Linear(200,300),
            nn.ELU(),
            nn.Linear(300,450),
            nn.ELU(),
            nn.Linear(450,50),
            nn.ELU(),
            nn.Linear(50,1),
        )
    def forward(self, x, t):
        encoded = self.encoder(x)
        t=self.project(t)
        aux=torch.cat((encoded,t),1)
##        print(np.shape(t))
##        print(np.shape(encoded))
##        print(np.shape(aux))
        decoded = self.decoder(aux)
        return decoded,encoded
#-------------------------------------------------------------------------------
#------------------CHAMA O MODELO E INICIA CAMADAS DE PESOS ORTOGONAIS----------
#-------------------------------------------------------------------------------
model = Autoencoder()
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())#,lr=1e-4,weight_decay = 1e-5)
#optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,weight_decay = 1e-5)#,momentum=0.5)
#-------------------------------------------------------------------------------
#------TREINO DO DECODER MODIFICADO--O[5]+Q[11] >> OUTPUT[1]--------------------
#-------------------------------------------------------------------------------
def treine(epochs):
    inp = pickle.load( open( "O", "rb" ) )
    question= pickle.load( open( "Q", "rb" ) )
    out =  pickle.load( open( "A", "rb" ) )
    n_batch=np.shape(inp)[0]
    batch_size=np.shape(inp)[1]
    n_examples=np.shape(inp)[2]
    for epoch in range(epochs):
        for batch_idx in range(n_batch):
            O=inp[batch_idx]
            Q=question[batch_idx]
            A=out[batch_idx]
            O=O.float()
            Q=Q.float()
            A=A.float()
            recon,latent = model(O,Q)
            loss=torch.mean((recon-A)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch+1},Loss:{loss.item():.4f}')
##treine(1000)
'''
#    O treinamento que foi efetivo, teve mudança         #
#        entre os exemplos de treino conforme abaixo:    #

treine(1000)
Box_2_dataset(5,500,5)
treine(250)
Box_2_dataset(5,500,5)
treine(250)

'''
#-------------------------------------------------------------------------------
#--------------------- SALVANDO-------------------------------------------------
#-------------------------------------------------------------------------------
PATH_save='Estado_Box_2_Elu().pt'
PATH_load='Estado_Box_2_Elu().pt'
#torch.save(model.state_dict(), PATH_save)
#s.exit()
model.load_state_dict(torch.load(PATH_load))
#torch.save(model.state_dict(), PATH_save)
#-------------------------------------------------------------------------------
#---------------------GRÁFICOS--------------------------------------------------
#-------------------------------------------------------------------------------
def Latent_values_Scynet():
    for aux in range(n_batch):
        O=inp[aux].float();        Q=question[aux].float()
        A=out[aux].float();        j=J[aux]
        x=np.zeros(np.shape(j)[0]);        y=np.zeros(np.shape(j)[0])
        recon,latent=model(O,Q)
        for i in range(0,499):
            x[i]=j[i]
            y[i]=latent[i]
        plt.scatter(x,y)
        plt.xlabel('Momento angular total')
        plt.ylabel('Latent Activation value')
        plt.legend();        plt.pause(1.8);        plt.close()
    plt.show()
Latent_values_Scynet()
