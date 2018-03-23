
import scipy.io 
import random
import numpy as np

data = scipy.io.loadmat('knn_data.mat')

data_train = data['train_data']
label_train = data['train_label']
#data_test = data['test_data']
#Label_test = data['test_label']
combined = np.concatenate((data_train , label_train ) , axis = 1)

random.shuffle(combined)


[rowTotal , colTotal] = combined.shape
if (rowTotal % 5 != 0):
    tp = rowTotal % 5 
    a = combined[0:(rowTotal-tp) , :]
    a = np.split(a , 5 )
    a[4] = np.concatenate((a[4] ,combined[(rowTotal-tp):rowTotal , :] ),axis =0)
else :
    a = np.split(combined,5)


 

final_accuracy = []
l=0
while (l < 9 ):
    k = (2 *l)+1
     
    for i in range(len(a)):
        test_t = a[i] 
        train_t = np.array([]).reshape( 0, 167)
        data_test = test_t[:,0:166]
        Label_test = test_t [:, 166]
        for j in range(len(a)):
            if i == j:
                continue
            else :
                train_t = np.concatenate(( train_t, a[i]), axis = 0)
            

        data_train = train_t[:,0:166]
        
        label_train = train_t[:,166]
        
        
    
        row_test, col_test =  data_test.shape
        row_train   , col_train =  data_train.shape
        
        
        
        count = np.array([i1 for i1 in range (row_train)])
        count = count.reshape(row_train , 1)
        accuracy =0 
        for i2 in range(row_test):
            tp = np.array([data_test[i2],]*row_train)
        
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
            if Label_test[i2] == ans:
                accuracy = accuracy +1 
        accuracy = (float(accuracy) /float(row_test) )*100   
        print accuracy
        final_accuracy.append((k , i, accuracy))
        l = l + 1
      

    
    
    
