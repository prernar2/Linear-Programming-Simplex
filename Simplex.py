#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from numpy.random import randn
import random
from numpy.linalg import norm,cond


# In[2]:


def simplex_step(A, b, c, iB, iN, xB, irule):
    
    m=len(A[:,1])
    origA=np.matrix(A.shape)
    origA=A.copy()
    CA=np.matrix(A.shape)
    CA=A.copy()
    c1=np.matrix(c.shape)
    c1=c.copy()
    B=np.eye(m)
    B=np.asmatrix(B)
    Binv=np.eye(m)
    Binv=np.asmatrix(Binv)
    cb=np.matrix([[0]*len(A[:,0])],dtype = np.float64)
    ratio=np.matrix([[0]*len(A[:,0])])
    cn=np.matrix([[0]*(A.shape[1]-len(b))],dtype = np.float64)
    istatus=0
    xb=b      
#Reducing index by 1 - Basic Variables & Non-basic Variables

    iB=np.array(iB)-1
    iN = np.array(iN)-1
       
#Computing B inverse        
    j=0
    for i in iB: 
        B[:,j] = A[:,i]
        j+=1    
    Binv =np.linalg.inv(B)  

    j=0
    for i in iB:
        cb[:,j] = c[:,i]
        j+=1
    j=0
    for i in iN:
        cn[:,j] = c[:,i]-cb*Binv*A[:,i]
        j+=1

        
#Computing reduced cost
    for i in iN:
        c1[:,i] = c[:,i] - cb*Binv*A[:,i]   
    
#Bland's rule

    if irule==1:   
        for i in iN:
            if c1[:,i]<0:
                CA[:,i]=Binv*A[:,i]
                if max(CA[:,i])>0:
                    mini = i
                    break
                else:
                    istatus=16  #Unbounded
                    return(istatus,iB+1,iN+1,xB,Binv)
            else:
                istatus=-1
                return(istatus,iB+1,iN+1,xB,Binv)
            
        mini = np.where(iN==mini)
        
#Dantzig's

    if irule==0:
        m=np.array(cn[0,:])
        if np.min(m)<0:
            index=iN[np.argmin(m)]
            CA[:,index]=Binv*A[:,index]
            if max(CA[:,index])>0:
                mini = np.argmin(m)
            else:
                istatus=16        #Unbounded
                return(istatus,iB+1,iN+1,xB,Binv)
        else:
            istatus=-1
            return(istatus,iB+1,iN+1,xB,Binv)
               
    #calculating b dash   
    bd=Binv*b

    #minimum ratio test 
    
    minrat=math.inf
    for i in range(len(bd)):
        if CA[i,iN[mini]]>0:

            ratio=(bd[i,0]/CA[i,iN[mini]])
            if (minrat>ratio):
                minrat=ratio
                j=i
                e=iB[i]

    iB[j]=iN[mini]
    iB=iB+1

    k=0
    for i in iN:
        if i==iN[mini]:
            break
        k=k+1
    iN[k]=e
    iN=iN+1

    j=0   
    for i in (iB-1): 
        B[:,j] = origA[:,i]
        j=j+1      

    Binv =np.linalg.inv(B)  
    xb=Binv*b
    return(istatus,iB,iN,xb,Binv)
    


# In[3]:


def simplex_init(A,b,c):
    
    for i in range(len(b)):
        if b[i,:]<0:
            A[i,:]=-1*A[i,:]
            b[i,:]=-1*b[i,:]
            
    
    origA=np.matrix(A.shape)
    origA=A.copy()
    I = np.eye(len(b))
    I = np.asmatrix(I)
    D = np.hstack((A,I))
    
    c_a = [[0]*c.shape[1]]
    c_a = np.hstack((c_a,[[1]*len(b)]))
     
    lis = []
    for i in range(D.shape[1]-len(b)+1,D.shape[1]+1):
        lis.append(i)

    iB=lis
    lis=[]
    lis=[x for x in range(1,D.shape[1]-len(b)+1)]

    iN=lis
    
    xB = b
    irule = 0
    istatus,iB,iN,xb,Binv=simplex_step(D, b, c_a, iB, iN, xB, irule)
    
    while(istatus==0):
        istatus,iB,iN,xB,Binv=simplex_step(D, b, c_a, iB, iN, xB, irule)
    newiN=[]
    status=0
    for i in (iB):
        for j in range(D.shape[1]-len(b)+1,D.shape[1]+1):
            if(i==j):
                status=1
                art=np.where(iB==i)
                istatus=4  
                if(xB[art]>0):
                    istatus=16
                    return (istatus, iB,iN,xB)
                return (istatus, iB,iN,xB)               
    
    if(status==0):
        istatus=0
        for k in (iN):
            status=0
            for l in range(D.shape[1]-len(b)+1,D.shape[1]+1):
                if(k==l):
                    status=1
                    break
            if(status==0):
                newiN.append(k)
        iN=newiN
        return istatus,iB,iN,xB


# In[4]:


def simplex_method(A,b,c,irule):
    
    istatus2,iB,iN,xB=simplex_init(A,b,c)
    istatus1=80
    if istatus2==16:
        istatus = 4

    elif istatus2==4:
        istatus = 16
   
    while (istatus2==0):
        istatus1,iB,iN,xB,Binv= simplex_step(A, b, c, iB, iN, xB, irule)
        if istatus1==16:
            istatus = 32

        istatus2=istatus1
    iB=np.array(iB)-1
    eta=0  
    X=np.zeros((1,len(iB)+len(iN)))
  
    if istatus1==-1:
        cB=np.zeros((1,len(iB)))
        j=0
        for i in iB:
            cB[:,j]=c[:,i]
            j+=1
            
        eta=cB*xB
        
        istatus=0
        j=0
        for i in iB:
            X[:,i]=xB[j,:]
            j+=1

    return (istatus,X,eta,iB+1,iN,xB)

