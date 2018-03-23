



import scipy.io 
import pandas 
import random
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def distance(centroids , p , k , a  ):
    d = np.empty([a,k])
    t = np.empty([a,1])
    
    for i in range( 0, a) :
        d[i][:] = [np.linalg.norm(p[i][:] - centroids[c][:]) for c in range (0,k)]
        t[i] = np.argmin(d[i][:])
        
        
    return t 
    
    
def centroid (centroids , p , k , a,b , labels  ):
    sum= np.empty([k,b])
    for i in range(0,k):
        count =0 
        for j in range(0,a):
            if labels[j]== i:
                sum[i][:] = sum[i][:] + p[j][:]
                count = count +1 
        if count == 0 :
            sum[i][:] = centroids[i][:]
        else :           
            sum[i][:] = (sum[i][:] / count)
    
        
    return sum
    
    
data = scipy.io.loadmat('kmeans_data.mat')
p = data['data']
p =np.array(p)
q = []
[a , b ] = p.shape 
K = 11
cost_final =[]
fin_cost=np.empty([K,1])
for k in range(2 ,K ):
    idx = np.random.randint(a , size = k)


    centroids  = p[idx, :]
    new_cent = np.zeros([ k, b ])
    prev = np.zeros([k,b])
    dist_cent= np.empty([ k, b ])
    labels = [] 
    count = 1 



    dist_cent = np.linalg.norm(centroids - new_cent)
    
    count = 0 


    while dist_cent > 0.0001: 

        labels = distance(centroids , p , k , a)
    #print count
        count = count +1 
    
        new_cent = centroid(centroids , p , k  , a ,b, labels )
    
 
        dist_cent = np.linalg.norm(centroids - new_cent)
    
    #prev =  centroids
        centroids = new_cent
    #print centroids
        
    #count = count + 1 
        #print count
    
    #print dist_cent 
    print labels
    print "k = " 
    print k 
    

    cost =np.empty([k])   
    
    print "start" , fin_cost
    for j in range(0,k):
    
        for i in range(0,a):
            if labels[i] == j :
                #print "$$$$" , p.shape
                #print "******" , centroids.shape
                
                cost[j] =cost[j] + np.sum(np.square(p[i,:] - centroids[j,:]),0)
    print (cost)
    #print "blank",fin_cost 
    fin_cost = np.sum(cost)
    
    #print "HEHE",fin_cost
        #v= np.sum(cost , axis =1 )     
        #q.append( np.sum(a , axis =0 ))
      
    #print fin_cost 
    cost_final.append(( k , fin_cost))   
#print q  
#print fin_cost
#print  fin_cost.shape
print cost_final
t = []
#print fin_cost
#t = list((range(2,4)))

#plt.plot(t,q)

