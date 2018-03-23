import scipy.io 
import pandas 
import random
import numpy as np
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
    
def initialisation(p, k ):
    #count = np.array([i1 for i1 in range (p.shape[0])])
    #count = count.reshape(p.shape[0] , 1)
        
    idx = np.random.randint(p.shape[0] , size = 1)
    cent = np.empty([ k-1, p.shape[1]])
    cent= np.concatenate((cent  ,p[idx]),axis =0)
    print cent.shape
    while cent.shape[0]< k :
        count = 1
        dist = np.empty([a,cent.shape[0]])
        for i in range(0,p.shape[0]):
            dist[i][:] = [np.linalg.norm(p[i][:] - cent[c][:]) for c in range (0,cent.shape[0])]
            
        #dist = distance(cent , p , len(cent) , p.shape[0] ) 
        print dist.shape
        print dist
        sum_dist = np.sum(dist , axis =0)
        print sum_dist
        print sum_dist.shape
        prob = np.divide(dist,sum_dist)
        print "---------->",prob        
        cum_prob = np.cumsum(prob)
        r = random.random()
        #ind = np.where(cum_prob >= r)[0][0]
        print cum_prob.shape
        cent= np.concatenate( cent , p[np.where(cum_prob >= r)] , axis =0)        
        #cent.append(p[ind])
        #count = count +1 
    print "********",cent   
    #print len(cent)
    return cent 
    
    #temp = np.array([p[idx],]*p.shape[0])
   
    # list =[]
    # list.append(p[idx])
    # dist = np.empty([p.shape[0],1])
    #dist =np.array([]).reshape(p.shape[0] , 0)
    # while(True):
        # for i in range(len(list)):
            # list[i] = list[i].reshape(21)
            # temp = np.array([list[i],]*p.shape[0])
            # print p.shape 
            # print temp.shape
            # dist  = dist + np.sqrt(np.sum(np.square(np.int64(p - temp)),1)) 
        # t =np.concatenate((dist , count),axis =1)
            
        # sort_arr = sorted(t,key =lambda x: (x.__getitem__(0)))
        # sel_arr = sort_arr[len(sort_arr)-1]
        # if len(list) >=k :
            # return list
        # else :
            # list.append(sel_arr)
            
            
    
data = scipy.io.loadmat('kmeans_data.mat')
p = data['data']
p =np.array(p)
q = []
[a , b ] = p.shape 
K = 11
cost_final =[]
fin_cost=np.empty([K-1,1])
for k in range(2 ,K ):
    #idx = np.random.randint(a , size = k)
    
    centroids  = initialisation( p , k )
    print centroids
    
    
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
    
    where_are_NaNs = np.isnan(cost)
    cost[where_are_NaNs] = 0
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