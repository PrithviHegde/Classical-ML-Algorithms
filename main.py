from statistics import NormalDist
import pandas as pd
import numpy as np
import cupy as cp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import math,random,time
import matplotlib.pyplot as plt
from copy import deepcopy

start = time.time()
uneditedDataset = pd.read_csv("dataset.csv")

uneditedDataset.replace(to_replace='M',value=-1,inplace=True)
uneditedDataset.replace(to_replace='B',value=1,inplace=True)
originalDataset = uneditedDataset.dropna(axis=0)

def feature_task_1(df:pd.DataFrame) -> pd.DataFrame:
    columnlist = df.columns.to_list()
    finaldf = [df[i].to_list() for i in columnlist]
    for i in range(len(finaldf)):
        if columnlist[i] == 'id':continue
        elif columnlist[i] == 'diagnosis':
            for j in range(len(finaldf[i])):
                if type(finaldf[i][j])==float() and math.isnan(finaldf[i][j]):
                    finaldf[i][j] = df[columnlist[i]].mode()
        else:
            for j in range(len(finaldf[i])):
                if math.isnan(float(finaldf[i][j])):
                    finaldf[i][j] = df[columnlist[i]].mean()
    fdf = np.array(finaldf)
    fdf = fdf.T

    finaldf = pd.DataFrame(fdf, columns=columnlist)
    return finaldf

def feature_task_2(df:pd.DataFrame) -> pd.DataFrame:
    columnlist = df.columns.to_list()
    finaldf = [df[i].to_list() for i in columnlist]
    finaldf = [np.array(finaldf[i],dtype=float) if i!=1 else np.array(finaldf[i]) for i in range(len(finaldf))]
    for i in range(len(finaldf)):
        if (columnlist[i] == 'id') or (columnlist[i] == 'diagnosis'):continue
        else:
            finaldf[i] = (finaldf[i] - finaldf[i].mean())/finaldf[i].std()
    fdf = np.array([i.tolist() for i in finaldf]).T

    finaldf = pd.DataFrame(fdf, columns=columnlist)
    return finaldf

ft1 = feature_task_1(originalDataset)
ft2 = feature_task_2(feature_task_1(originalDataset))

def prepare_dataset(df:pd.DataFrame):
    train = df.sample(frac=0.67,random_state=45)
    # train = df.sample(frac=0.67)
    test = df.drop(train.index)


    train_y = train['diagnosis'].to_list()
    # train_t = [1 if i=='B' else -1 for i in train_y]
    train_t = np.array(train_y).reshape(len(train_y),1)

    test_y = test['diagnosis'].to_list()
    # test_t = [1 if i=='B' else -1 for i in test_y]
    test_t = np.array(test_y).reshape(len(test_y),1)

    train_x = train.drop(['id', 'diagnosis'], axis=1).to_numpy(dtype=float)
    test_x = test.drop(['id', 'diagnosis'], axis=1).to_numpy(dtype=float)

    ones = np.ones([train_x.shape[0],1])
    train_x = np.concatenate((ones,train_x),axis=1)
    ones = np.ones([test_x.shape[0],1])
    test_x = np.concatenate((ones,test_x),axis=1)

    return train_x,train_t,test_x,test_t

# PART A: Perceptron Algorithm Models   
def activation_function(x):
    if(x>=0):
        return 1
    else:
        return -1

def perceptron(X,y,eta=1,iter=1000):
    m = X.shape[0]
    n = X.shape[1]
    converge = False
    w = np.zeros((n,1))
    while(iter):
        misclass_count = 0
        for i in range(len(X)):
            row = X[i]
            Xn = row.reshape(n,1)
            mul_result = (w.T)@Xn
            classification_output = activation_function(mul_result[0][0])
            if classification_output != y[i]:
                misclass_count+=1
                w += eta*(y[i]*Xn)
        if(misclass_count==0):
            converge = True
            break
        iter-=1
    return w,converge

def PM1(df:pd.DataFrame,printdata=True):
    trainX,trainy,testX,testy = prepare_dataset(df)
    w,converge = perceptron(trainX,trainy)
    if printdata:
        if converge:
            print("PM1: The dataset is linearly separable.")
        else:
            print("PM1: The dataset is not linearly separable.")
    
    result = testX @ w

    result = [activation_function(i) for i in result]
    result = np.array(result).reshape(testy.shape)

    mcr = ((np.abs(result - testy)).mean())/2
    return mcr

def PM2(df:pd.DataFrame,printdata=True):
    trainX,trainy,testX,testy = prepare_dataset(df)
    train = np.concatenate((trainX,trainy),axis=1)
    np.random.shuffle(train)
    trainX,trainy = np.hsplit(train, [train.shape[1]-1])
    w,converge = perceptron(trainX,trainy)
    if printdata:
        if converge:
            print("PM2: The dataset is linearly separable.")
        else:
            print("PM2: The dataset is not linearly separable.")
    
    result = testX @ w

    result = [activation_function(i) for i in result]
    result = np.array(result).reshape(testy.shape)

    mcr = ((np.abs(result - testy)).mean())/2
    return mcr

def PM3(df:pd.DataFrame,printdata=True):
    ft2 = feature_task_2(df)
    trainX,trainy,testX,testy = prepare_dataset(ft2)
    w,converge = perceptron(trainX,trainy)
    if printdata:
        if converge:
            print("PM3: The dataset is linearly separable.")
        else:
            print("PM3: The dataset is not linearly separable.")
    
    result = testX @ w

    result = [activation_function(i) for i in result]
    result = np.array(result).reshape(testy.shape)

    mcr = ((np.abs(result - testy)).mean())/2
    return mcr

def PM4(df:pd.DataFrame,printdata=True):
    ft2 = feature_task_2(df)
    trainX,trainy,testX,testy = prepare_dataset(ft2)
    splitind = trainX.shape[0]
    joined = np.vstack((trainX,testX))
    # random shuffle cols
    jt = deepcopy(joined.T)
    np.random.shuffle(jt)
    joined = jt.T

    trainX,testX = np.vsplit(joined,[splitind])
    w,converge = perceptron(trainX,trainy)
    if printdata:
        if converge:
            print("PM4: The dataset is linearly separable.")
        else:
            print("PM4: The dataset is not linearly separable.")
    
    result = testX @ w

    result = [activation_function(i) for i in result]
    result = np.array(result).reshape(testy.shape)

    mcr = ((np.abs(result - testy)).mean())/2
    return mcr

# print(PM1(originalDataset),PM2(originalDataset),PM3(originalDataset),PM4(originalDataset))

# PART B: Fischer's Linear Discriminant
def project(features, values):
    estimator = LinearDiscriminantAnalysis()
    estimator.fit(features, values.ravel())

    singleFeatureSet = estimator.transform(features)
    return singleFeatureSet, estimator

#singleFeatureSet = project(trainX, trainY)

#Gaussian generative discriminant
def findLikelyhood(xValue, mean, std):
    return NormalDist(mu=mean, sigma=std).pdf(xValue)

#finds posterior of xValue for a given class cls in yValues
def gaussianPosteriorProbability(xValue, mean1, std1, mean2, std2, prior1, prior2):
    likelyhood1 = findLikelyhood(xValue, mean1, std1)
    numerator1 = likelyhood1*prior1

    likelyhood2 = findLikelyhood(xValue, mean2, std2)
    numerator2 = likelyhood2*prior2

    evidence = numerator1+numerator2

    posterior1 = numerator1/evidence
    posterior2 = numerator2/evidence
    return posterior1, posterior2

def findDecisionBoundary(trainX, trainY):
    class1Points = []
    class2Points = []
    for index, elem in enumerate(trainX):
        if trainY[index] == 1:
            class1Points.append(elem)
        else:
            class2Points.append(elem)

    mean1  = np.average(class1Points)
    std1 = np.std(class1Points)
    mean2  = np.average(class2Points)
    std2 = np.std(class2Points)
    prior1 = len(class1Points)/len(trainX)
    prior2 = len(class2Points)/len(trainX)

    _max = max(max(class1Points), max(class2Points))
    _min = min(min(class1Points), min(class2Points))


    return mean1, std1, mean2, std2, prior1, prior2 , _max, _min

def fischersLD(trainX, trainY, testX, model, notPlotted):
    ans = []
    trainingFeature, estimator = project(trainX, trainY)
    testingFeature = estimator.transform(testX)
    
    mean1, std1, mean2, std2, prior1, prior2, _max, _min = findDecisionBoundary(trainingFeature, trainY)
    for value in testingFeature:
        posterior1, posterior2 = gaussianPosteriorProbability(value, mean1, std1, mean2, std2, prior1, prior2)
        if posterior1 > posterior2:
            ans.append(1)
        else:
            ans.append(-1)
    temp = []
    _range = np.linspace(_min, _max, 1000)
    for i in _range:
        pos1, pos2 = gaussianPosteriorProbability(i[0], mean1, std1, mean2, std2, prior1, prior2)
        temp.append((i[0],abs(pos1-pos2)))

    temp.sort(key = lambda x: x[1])
    boundary = temp[0][0]
    
    if notPlotted:
        fig, ax = plt.subplots()
        for index, elem in enumerate(trainingFeature):
            if trainY[index] == 1:
                ax.plot(elem,0, marker = 'o', color = 'r', markersize = 2)
            else:
                ax.plot(elem,0, marker = 'o', color = 'b',markersize = 2)

        ax.plot([boundary, boundary], [-5, 5], 'bo',linestyle='-', label = "Decision Boundary (x = "+str(boundary) + ')')
        ax.legend()
        ax.set_xlabel("Projected Points")
        ax.set_ylabel("")
        ax.set_title("Projected Points in 1D with Decision Boundary for "+ model)

        # plt.show()
        plt.savefig("./Plots/"+model+"-plot.png")
        plt.close(fig)

    return ans, boundary

def FLDM1(df:pd.DataFrame, notPlotted = True):
    ft1 = feature_task_1(df)
    ft2 = feature_task_2(ft1)
    trainX,trainy,testX,testy = prepare_dataset(ft2)
    result, boundary = fischersLD(trainX,trainy,testX, "FLDM1",notPlotted)
    result = np.array(result).reshape(testy.shape)
    mcr = ((np.abs(result - testy)).mean())/2

    return mcr, boundary

def FLDM2(df:pd.DataFrame, notPlotted = True):
    ft1 = feature_task_1(df)
    ft2 = feature_task_2(ft1)
    trainX,trainy,testX,testy = prepare_dataset(ft2)
    splitind = trainX.shape[0]
    joined = np.vstack((trainX,testX))
    # random shuffle cols
    jt = deepcopy(joined.T)
    np.random.shuffle(jt)
    joined = jt.T

    trainX,testX = np.vsplit(joined,[splitind])
    result, boundary = fischersLD(trainX,trainy,testX, "FLDM2",notPlotted)
    result = np.array(result).reshape(testy.shape)
    mcr = ((np.abs(result - testy)).mean())/2
    return mcr, boundary

# PARTC: Logistic Regression
def sigmoid_fn(x):

    new = np.zeros(x.shape)
    for i in range(len(x)):
        for j in range(len(x[i])):
            new[i][j] = 1/(1 + math.exp(-x[i][j]))
    return new

def cost(x, y, theta):
    # np.seterr(all="ignore") 
    m = x.shape[0]
    h = sigmoid_fn(np.matmul(x, theta))
    cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
    return cost[0][0]

def batch_gradient_descent(X,y,alpha=0.01,iter=100,cutoff=1e-6):
    costs = []
    w = cp.zeros((len(X[0]),1))
    c = None
    for i in range(iter):
        w = w - (alpha/X.shape[0])*(X.T @ (sigmoid_fn(X @ w) - y))
        cst = cost(X,y,w)
        if c and (abs(c-cst)<=cutoff):
            break
        c = cst
        costs.append(c)
    return w,costs

def stochastic_gradient_descent(X,y,alpha=0.01,iter=100,cutoff=1e-6):
    costs = []
    w = cp.zeros((len(X[0]),1))
    c = None
    for epoch in range(iter):
        ind = [i for i in range(X.shape[0])]
        random.Random(5).shuffle(ind)
        for i in ind:
            xi = X[i].reshape((1,X.shape[1]))
            yi = y[i]
            w = w - alpha*(xi.T @ (sigmoid_fn(xi @ w) - yi))
        cst = cost(X,y,w)
        if c and (abs(c-cst)<=cutoff):
            break
        c = cst
        costs.append(c)
    return w,costs
            
def minibatch_gradient_descent(X,y,batch_size,alpha=0.01,iter=100,cutoff=1e-6):
    costs = []
    w = cp.zeros((len(X[0]),1))
    c = None
    for epoch in range(iter):
        ind = [i for i in range(X.shape[0])]
        random.Random(5).shuffle(ind)
        X_shuffled = X[ind]
        y_shuffled = y[ind]
        for i in range(0,X.shape[0],batch_size):
            xbatch = X_shuffled[i:i+batch_size]
            ybatch = y_shuffled[i:i+batch_size]
            w = w - (alpha/xbatch.shape[0])*(xbatch.T @ (sigmoid_fn(xbatch @ w) - ybatch))
        cst = cost(X,y,w)
        if c and (abs(c-cst)<=cutoff):
            break
        c = cst
        costs.append(c)
    return w,costs

def plot(allcosts,alpha,model):
    fig, ax = plt.subplots()
    mxsize = max([len(allcosts[0]),len(allcosts[1]),len(allcosts[2])])
    for i in range(len(allcosts)):
        for j in range(mxsize-len(allcosts[i])):
            allcosts[i].append(allcosts[i][-1])
    
    labels = ["Batch Gradient Descent","Minibatch Gradient Descent","Stochastic Gradient Descent"]
    for i,costs in enumerate(allcosts):
        ax.plot([j for j in range(1,len(costs)+1)],costs, label = labels[i])
    ax.legend()
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Cost")
    ax.set_title("Cost Function vs Epochs for alpha = " + str(alpha))
    # plt.show()
    plt.savefig("./Plots/"+model+"- Cost vs Iter for "+str(alpha)+".png")
    plt.close(fig)

def classify(x,db):
    if x>=db:
        return 1
    else:
        return 0

def LR1(df:pd.DataFrame, maxIter=1000, printdata = True):
    trainX,trainy,testX,testy = prepare_dataset(df)

    for row in range(len(trainy)):
        for i in range(len(trainy[row])):
            trainy[row][i] = max(trainy[row][i],0)
    for row in range(len(testy)):
        for i in range(len(testy[row])):
            testy[row][i] = max(testy[row][i],0)
    splitind = trainX.shape[0]
    joined = cp.vstack((trainX,testX))
    joined = (joined - joined.mean())/joined.std()
    trainX,testX = cp.vsplit(joined,[splitind])
    errs = []
    wpluscosts = dict()
    for lr in [0.01,0.001,0.0001]:
        w1,costs1 = batch_gradient_descent(trainX,trainy,lr, iter = maxIter)
        errs.append((0,lr,costs1[-1]))
        wpluscosts[(0,lr)] = w1
        w2,costs2 = minibatch_gradient_descent(trainX,trainy,int(math.sqrt(trainX.shape[0])),lr, iter=maxIter)
        errs.append((1,lr,costs2[-1]))
        wpluscosts[(1,lr)] = w2
        w3,costs3 = stochastic_gradient_descent(trainX,trainy,lr, iter=maxIter)
        errs.append((2,lr,costs3[-1]))
        wpluscosts[(2,lr)] = w3
        if printdata:plot([costs1,costs2,costs3],lr,"LR1")
    
    testerrors = []
    grModels = ["Batch Gradient Descent","Minibatch Gradient Descent","Stochastic Gradient Descent"]
    
    if printdata:print("For the Classification Model LR1: ")    
    
    for i,model in enumerate(grModels):
        if printdata:print("\tFor the "+model+" model,")
        for j in [0.01,0.001,0.0001]:
            if printdata:print("\t\tUsing learning rate of "+str(j)+" ,")
            for k in [0.3, 0.4, 0.5, 0.6, 0.7]:
                w = wpluscosts[(i,j)]
                result = sigmoid_fn(testX @ w)

                result = [classify(i,k) for i in result]
                result = cp.array(result).reshape(testy.shape)

                mcr = (cp.abs(result - testy)).mean()
                testerrors.append(mcr)
                if printdata:print("\t\t\tTesting accuracy using decision probability threshold of "+str(k)+" is : "+str(1-mcr))

    return min(testerrors)

def LR2(df:pd.DataFrame, maxIter=1000,printdata = True):
    ft1 = feature_task_1(df)
    ft2 = feature_task_2(ft1)
    trainX,trainy,testX,testy = prepare_dataset(ft2)
    
    for row in range(len(trainy)):
        for i in range(len(trainy[row])):
            trainy[row][i] = max(trainy[row][i],0)
    for row in range(len(testy)):
        for i in range(len(testy[row])):
            testy[row][i] = max(testy[row][i],0)
    splitind = trainX.shape[0]
    joined = cp.vstack((trainX,testX))
    joined = (joined - joined.mean())/joined.std()
    trainX,testX = cp.vsplit(joined,[splitind])
    errs = []
    wpluscosts = dict()
    for lr in [0.01,0.001,0.0001]:
        w1,costs1 = batch_gradient_descent(trainX,trainy,lr, iter=maxIter)
        errs.append((0,lr,costs1[-1]))
        wpluscosts[(0,lr)] = w1
        w2,costs2 = minibatch_gradient_descent(trainX,trainy,int(math.sqrt(trainX.shape[0])),lr, iter=maxIter)
        errs.append((1,lr,costs2[-1]))
        wpluscosts[(1,lr)] = w2
        w3,costs3 = stochastic_gradient_descent(trainX,trainy,lr, iter=maxIter)
        errs.append((2,lr,costs3[-1]))
        wpluscosts[(2,lr)] = w3
        if printdata:plot([costs1,costs2,costs3],lr,"LR2")
    
    testerrors = []
    grModels = ["Batch Gradient Descent","Minibatch Gradient Descent","Stochastic Gradient Descent"]
    
    if printdata:print("For the Classification Model LR2: ")
    
    for i,model in enumerate(grModels):
        if printdata:print("\tFor the "+model+" model,")
        for j in [0.01,0.001,0.0001]:
            if printdata:print("\t\tUsing learning rate of "+str(j)+" ,")
            for k in [0.3, 0.4, 0.5, 0.6, 0.7]:
                w = wpluscosts[(i,j)]
                result = sigmoid_fn(testX @ w)

                result = [classify(i,k) for i in result]
                result = cp.array(result).reshape(testy.shape)

                mcr = (cp.abs(result - testy)).mean()
                testerrors.append(mcr)
                if printdata:print("\t\t\tTesting accuracy using decision probability threshold of "+str(k)+" is : "+str(1-mcr))

    return min(testerrors)



comparativeStudyData = []

errorarray = [0,0,0,0,0,0,0]
print("PART A:")
print("\nPM1:")
err = PM1(originalDataset)
print("Misclassification Rate: ", err)
errorarray[0] = 1-err
print("\nPM2:")
err = PM2(originalDataset)
print("Misclassification Rate: ", err)
print("\nPM3:")
err = PM3(originalDataset)
print("Misclassification Rate: ", err)
errorarray[1] = 1-err
print("\nPM4:")
err = PM4(originalDataset)
print("Misclassification Rate: ", err)
errorarray[2] = 1-err

print("\nPART B:\n")
err, boundary = FLDM1(uneditedDataset)
print("Misclassification Rate for model FLDM1: ", err)
print("Decision Boundary for FLDM1 is x =  ", boundary)
errorarray[3] = 1-err
err, boundary = FLDM2(uneditedDataset)
print("\nMisclassification Rate for model FLDM2: ", err)
print("Decision Boundary for FLDM2 is x =  ", boundary)
errorarray[4] = 1-err

print("\nPART C:\n")
err = LR1(originalDataset)
errorarray[5] = 1-err
print()
err = LR2(uneditedDataset)
errorarray[6] = 1-err

print("\nPART D\n")
print("Running tests on sample 1")

comparativeStudyData.append(errorarray)
for i in range(9):
    print("Running tests on sample",i+2)
    uneditedDataset = uneditedDataset.sample(frac=1)
    originalDataset = uneditedDataset.dropna(axis=0)
    errorarray = [0,0,0,0,0,0,0]
    errorarray[0] = 1-PM1(originalDataset,printdata=False)
    errorarray[1] = 1-PM3(originalDataset,printdata=False)
    errorarray[2] = 1-PM4(originalDataset,printdata=False)
    ans, _none = FLDM1(uneditedDataset,notPlotted = False)
    errorarray[3] = 1-ans
    ans, _none = FLDM2(uneditedDataset,notPlotted = False)
    errorarray[4] = 1-ans
    errorarray[5] = 1-LR1(originalDataset,maxIter=249,printdata=False)
    errorarray[6] = 1-LR2(uneditedDataset,maxIter=249,printdata=False)
    comparativeStudyData.append(errorarray)

print()
print("Testing accuracy % over 10 random train test splits: ")
comparativeStudyData = cp.array(comparativeStudyData)
comparativeStudyData = 100*comparativeStudyData
averages = cp.array(comparativeStudyData.mean(axis=0),dtype='<U5').reshape((1,7))
variances = cp.array(comparativeStudyData.var(axis=0),dtype='<U5').reshape((1,7))
comparativeStudyData = cp.array(comparativeStudyData,dtype='<U5')
comparativeStudyData = cp.append(comparativeStudyData,averages,axis=0)
comparativeStudyData = cp.append(comparativeStudyData,variances,axis=0)
labels = cp.array(["Random Split "+str(i)+'  ' for i in range(1,comparativeStudyData.shape[0]-1)]+["Average "]+["Variance "]).reshape((comparativeStudyData.shape[0],1))
comparativeStudyData = cp.concatenate((labels,comparativeStudyData),axis=1)
comparativeStudyData = pd.DataFrame(comparativeStudyData, columns=["Split Number/Model  ","PM1","PM3","PM4","FLDM1","FLMD2","LR1","LR2"])

print()
print(comparativeStudyData.to_markdown())
print()
averages = averages.tolist()[0]
mdls = ["PM1","PM3","PM4","FLDM1","FLMD2","LR1","LR2"]
print("The best model on average over all 10 random samples is:",mdls[averages.index(max(averages))], end = ' ')
print(", with testing accuracy rate of:",max(averages),"%")
end = time.time()
print("Time Taken: ", round(end - start,3),"s",sep='')
