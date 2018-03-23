import scipy.io 
import random
import numpy as np
import csv
file = open('finaltrain.csv' , 'r')
data = csv.reader(file , delimiter=',')
data_train = np.array([row for  row in data])
data_train = data_train.astype('float')
 
data = scipy.io.loadmat('knn_data.mat')
#data_train= data['train_data']
label_train = data['train_label']
#data_test = data['test_data']
#label_test = data['test_label']
#label_train = np.concatenate((label_train,label_test),axis=0)
combined = np.concatenate((data_train , label_train ) , axis = 1)
#total_p = np.concatenate((data_test , label_test ) , axis = 1)
#combined= np.concatenate((total_d , total_p ) , axis = 0)
random.shuffle(combined)
[rowTotal , colTotal] = combined.shape
arrays = np.split(combined,5)
 
final_accuracy = []
l=0
while (l < 9 ):
    k = (2 *l)+1
     
    for i in range(len(arrays)):
        test_t = arrays[i] 
        train_t = np.array([]).reshape( 0, 51)
        data_test = test_t[:,0:50]
        Label_test = test_t [:, 50]
        for j in range(len(arrays)):
            if i == j:
                continue
            else :
                train_t = np.concatenate(( train_t, arrays[i]), axis = 0)
            
        data_train = train_t[:,0:50]
        
        label_train = train_t[:,50]
        
        
    
        row_test, col_test =  data_test.shape
        row_train   , col_train =  data_train.shape
        
        
        
        count = np.array([i1 for i1 in range (row_train)])
        count = count.reshape(row_train , 1)
        accuracy =0 
        for y in range(row_test):
            tp = np.array([data_test[y],]*row_train)
        
            neigh_dist  = np.sqrt(np.sum(np.square(np.int64(data_train - tp)),1))
            neigh_dist = neigh_dist.reshape(row_train, 1)
           
            t =np.concatenate((neigh_dist , count),axis =1)
            
            array_sorted = sorted(t,key =lambda x: (x.__getitem__(0)))
            array_selected = array_sorted[0:k]
            class_label_zero=0 
            class_label_one=1
            for j in array_selected:
                if (label_train[int(j[1])] == 0):
                
                    class_label_zero = class_label_zero+1
                
                
                else:
                    class_label_one = class_label_one+1
              
            ans=1
            if class_label_zero > class_label_one:
                ans = 0
            if Label_test[y] == ans:
                accuracy = accuracy +1 
        accuracy = (float(accuracy) /float(row_test) )*100   
        print accuracy
        final_accuracy.append((k , i, accuracy))
        l = l + 1
      
    
    
    