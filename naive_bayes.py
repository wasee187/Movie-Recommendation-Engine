import numpy as np
from collections import defaultdict
X_train = np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,1,0]
])

Y_train = ["Y", "N", "Y", "Y"]
X_test = np.array([[1,1,0]])

#function for grouping the output of liked or not liked data in an index format
def get_label_indetity(labels):
    label_identity = defaultdict(list)
    for index, label in enumerate(labels): #enumerate(labels) turns the list into pairs of index and value
        label_identity[label].append(index)
    return label_identity #this will return which class(Y/N) is belong to which row[only row numbers]

label_identities = get_label_indetity(Y_train)
print("label_identity:\n",label_identities)

###getting prior 
def get_prior(label_identity):
    prior={}
    for label, indexes in label_identity.items():#.items() gives key, value pair from a dictionary
        prior[label] = len(indexes)
    total_count = sum(prior.values()) #count total number of values(y and N) which is 4
    for label in prior:
        prior[label] /= total_count
    return prior

prior = get_prior(label_identities)
print('Prior :', prior)

###getting likelihood, P(features|class)
def get_likelihood(features, label_identity, smoothing=0):
    likelihood = {}
    for label, index in label_identity.items():
        class_features = features[index, :] #sorting the features(movie 1,2,3) according to the index of classes(Y,N)
        feature_count = class_features.sum(axis=0) #counting accross virtically of the features
        likelihood[label] = feature_count + smoothing #appling smoothing/ Laplace

        total_class_count = len(index) #ffinding the total sample class 

        likelihood[label] = likelihood[label] / (total_class_count + 2*smoothing) #appling smoothing formula 
    return likelihood
smoothing = 1
likelihood = get_likelihood(X_train,label_identities, smoothing)
print("likelihood : ", likelihood)

#getting posterior for testing
def get_posterior(X_sample, prior, likelihood):
    #using two posterior variable beacuse of multiple test example
    posteriors = []
    posterior = []
    for x_test in X_sample: #outer loop : Loops through each test sample in X
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items(): #middle loop : Goes through each class (like "Y" or "N") and its likelihood vector. 
            for index, bool_value in enumerate(x_test): #inner loop: Goes feature by feature in the test sample x.
                if bool_value == 1: 
                    #if test_sample has 1 than just multiply with corresponding index number of likelihood
                    posterior[label] *= likelihood_label[index]
                else : 
                    #if test_sample has 0 than minus the corresponding index number of likelihood from 1 and multiply it 
                    posterior[label] *= (1-likelihood_label[index])
    #normalization 
    sum_posterior = sum(posterior.values()) #summing all posterior values
    for label in posterior:
        if posterior[label] == float("inf"): #Sometimes (rarely), multiplying many probabilities can underflow or overflow.
            posterior[label] = 1.0
        else : 
            posterior[label] /= sum_posterior

    posteriors.append(posterior.copy())
    return posteriors

posterior = get_posterior(X_test,prior, likelihood)
print("posterior : \n", posterior)