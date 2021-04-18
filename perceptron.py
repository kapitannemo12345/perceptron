import numpy as np
import pandas as pd
from sklearn import preprocessing as prep
import sys
import math
from math import e
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import random
import matplotlib.pyplot as plt

def F(W,x,y,punkty):
    
    #print('punkty:',punkty)
    #print('x:',x)
    #E=np.zeros((punkty.shape[0],punkty.shape[1]-1), )
    E=[]
    
    #print('E:',E)
    y_pred=np.empty((x.shape[0],1), float)
    for i in range(x.shape[0]):
       calculate=W[0]*1+W[1]*x[i,1]+W[2]*x[i,2]
       
       print('obliczone',calculate)
       #np.dot(i,x)
       
       
       if(calculate>0):#funkcja przyporzadkwywyujaca klasyfikacje
           y_pred[i,0]=1
       elif(calculate<=0):
           y_pred[i,0]=-1
           
       if(y_pred[i,0]!=y[i]):
           #print(punkty[i,:])
           #print(E)
           E.append(punkty[i,:])
           #E[i,:]=x[i,:]
           
           
    
    
    print('waga teraz',y_pred)
    #print('porpawnioee e:',E)
    
    return E

def alg_delta(punkty):
    W= np.zeros((3), dtype=float)#x0=1 bo tak bylo na wykladzie dlatego 3 pola w wektorze
    sizex=punkty.shape[1]-1#ostatnia kolumna  to wagi dlatego bez ostatniej klolumny
    print(sizex)
    x=punkty[:,0:sizex]
    y=punkty[:,3]
    wsp=0.5
    kroki=0#zlicza kroki
    W= np.zeros((x.shape[1]), dtype=float)#x0=1 bo tak bylo na wykladzie dlatego 3 pola w wektorze
    print(x)
    print(y)
    print(W)
   
    #E=np.zeros((4), )
    
    #E=F(W,x,y,punkty)
    #print(E)
    
    i=0
    while(i!=10000):
        #print('test')
        #E=[]
        
        E=F(W,x,y,punkty)
        print('e ma teraz:',E)
        print()
       
        if(E==[]):#warunek stopu            
            break
        H=random.choice(E)#losowanie Z E
        #print(H)
               
        W[0]=W[0]+wsp*H[0]*H[3]#koretka wag
        W[1]=W[1]+wsp*H[1]*H[3]
        W[2]=W[2]+wsp*H[2]*H[3]

        i=i+1
          
    return W,i
def generate_nums(m,range_1, range_2):
    tempList = []
    for i in range(0,m):
        x = round(random.uniform(range_1,range_2),2)
        tempList.append(x)
    return np.transpose(tempList).reshape(m,1)

def generate_label(m, X):
    tempList = []
    for i in range(m):
        if X[i][1] > X[i][2]:
            tempList.append(-1)
        else:
            tempList.append(1)
    return np.transpose(tempList).reshape(m, 1)
def slide(data, distance):
    for i in range(data.shape[0]):
        if data[i,-1] == -1:
            data[i, 1] = data[i, 1] + distance
            #data[i, 2] = data[i, 2] + distance
        else:
            data[i, 1] = data[i, 1] - distance
            #data[i, 2] = data[i, 2] - distance

    return data
def main():

    punkty=np.array([[1, -3, 1, -1]
    , [1, -1, -3, -1]
    , [1, 0, 0, -1]
    , [1, 2, -2, -1]
    , [1, 0, 2, -1]
    , [1, 1, 4, 1]
    , [1, 2, 5, 1]
    , [1, 3, 3, 1]
    , [1, 5, 5, 1]
    , [1, 7, 3, 1]
    , [1, -5, 5, -1]
    , [1, 1, 4, 1]],dtype='f')

    m = 500
    data = np.transpose([1 for _ in range(m)]).reshape(m, 1)
    data = np.hstack([data, generate_nums(m, -5, 5), generate_nums(m, -5, 5)])
    Y = generate_label(m, data)
    data = np.hstack([data, Y])
    data = slide(data,2)

    W,i=alg_delta(data)
    
    print(" W:",W)
    print(" i:",i)

    if(W[0]==0):
        W[0]=0.001
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.scatter(data[:,1],data[:,2],c=data[:,3])
    slope = -(W[0]/W[2])/(W[0]/W[1])  
    intercept = -W[0]/W[2]
    i=np.linspace(np.min(data[:,1]),np.max(data[:,1]))
    y = (slope*i) + intercept
    plt.plot(i,(slope*i) + intercept)
    
if __name__ == "__main__":
    main() 
    