from operator import itemgetter
from random import shuffle
import numpy as np
from sklearn.metrics import confusion_matrix
from math import *
from scipy.spatial import distance
import matplotlib.pyplot as plt

def readData(file):
    rfile = open(file, "r")

    l= []
    for line in rfile:
        stripped_line = line.strip()
        item = stripped_line.split(';')
        l.append(item)

    rfile.close()
    return l

def formatData(lst):
    for item in lst:
        for i in range(len(item)-1):
            item[i] = float(item[i])
    return lst

def formatTestData(lst):
    for item in lst:
        for i in range(len(item)):
            item[i] = float(item[i])
    return lst

def norm(data_1, data_2, data_len):
    dist = 0
    for i in range(data_len):
        dist = dist + np.square(data_1[i] - data_2[i])
    return np.sqrt(dist)

def divideData(lst, prop):
    shuffle(lst)
    pop, test = lst[:int(prop*len(lst))], lst[int(prop*len(lst)):]
    pop_class, test_class = [],[]
    for i in range(len(pop)):
        pop_class.append(pop[i][-1::])
        pop[i] = pop[i]
        
    for i in range(len(test)):
        test_class.append(test[i][-1::])
        test[i] = test[i]
    return pop, test, pop_class, test_class

def knn(dataset, testInstance, k): 
    distances = [None] * len(dataset)
    length = len(testInstance)
    i = 0 
    for x in range(len(dataset)):
        #print("test") 
        distances[x] = [x, distance.euclidean(testInstance, dataset[x][:-1])]
        
    
    sort_distances = sorted(distances, key = itemgetter(1))
    #print(sort_distances)
    neighbors = []
    
    for x in range(k):
        neighbors.append(sort_distances[x])
    
    counts = {0 : 0, 1 : 0, 2 : 0, 3 : 0}
    
    for x in range(len(neighbors)):
        response = dataset[neighbors[x][0]][-1]
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
    
    sort_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    #print(sort_counts)
    return(list(sort_counts.keys())[0])

def confusionMatrix(actu, pred):
    return confusion_matrix(actu, pred)

def outputFile(data):
    file = open("prediction_Demare_Delgado_groupeD.txt", "w+")
    for i in range(len(data)):
        file.write(data[i] + "\n")
        
    file.close()
    

if __name__ == '__main__' :
    dataset = formatData(readData("data_knn/data.txt"))
    preTest = formatData(readData("data_knn/preTest.txt"))
    finalTest = formatTestData(readData("data_knn/finalTest.txt"))
    

    #training
    k_n = np.arange(1, 50)
    '''k_n = [i for i in range(50,60)]
    dev, _, dev_class, _ = divideData(preTest, 1)
    dev_set_k = {}
    
    for k in k_n:
        dev_set = [None] * len(dev)
        for i in range(len(dev)):
            dev_set[i] = knn(dataset, dev[i][0:7], k)
            
        dev_set_k[k] = dev_set
    
    ks = {k:0 for k in k_n} 
    for k in k_n:
        count = 0
        for i in range(len(dev)):
            if(dev_set_k[k][i] == dev_class[i][0]):
                count += 1
        
        ks[k] = count/len(dev)
    
    plt.plot(ks.keys(), ks.values(), label = 'preTest dataset Accuracy')
    print("test")'''
    '''
    dev, _, dev_class, _ = divideData(dataset, 1)
    dev_set_k = {}
    
    for k in k_n:
        print(k)
        dev_set = [None] * len(dev)
        for i in range(len(dev)):
            dev_set[i] = knn(dataset, dev[i][0:7], k)
            
        dev_set_k[k] = dev_set
    
    ks = {k:0 for k in k_n} 
    for k in k_n:
        count = 0
    for i in range(len(dev)):
        if(dev_set_k[k][i] == dev_class[i][0]):
            count += 1
        
        ks[k] = count/len(dev)
    
    print(ks)
    
    
    plt.plot(ks.keys(), ks.values(), label = 'data dataset Accuracy')'''
    '''
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()'''
    
    # Conclusion : best k = 6 ou k = 4
    
    ###############################################
    
    #testing
    '''test, _, test_class, _ = divideData(dataset, 0.7)
    k = 4
    test_set = [None] * len(test)
    for i in range(len(test)):
        test_set[i] = knn(dataset, test[i][0:7], k)
    
        count = 0
        for i,j in zip(test_class, test_set):
            if i[0] == j:
                count += 1
            else:
                pass
        
    print('Final Accuracy of the Test dataset is ', count/(len(test_class)))
    
    print(confusionMatrix([item[0] for item in test_class], test_set))'''
    
    ###############################################
    
    #predicting

    k = 4
    results = [None] * len(finalTest)
    for i in range(len(finalTest)):
        results[i] = knn(dataset, finalTest[i], k)
    outputFile(results)
