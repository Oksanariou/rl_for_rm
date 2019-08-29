import numpy as np
import matplotlib.pyplot as plt


def compute_probas_logit(representative_utilities):
    """
        Input: List of the representative utilities of the different alternatives
        Output: List of the probabilities to choose each of the alternatives
    """
    numerators = np.exp([k for k in representative_utilities])
    normalization = sum(numerators)

    return numerators / normalization


def compute_probas_nested_logit(nests):
    """
        Input: List of nests. Each nest is a dictionary with two elements: a parameter lambda and a list of representative utilities.
        Output: List of the size of the number of nests made of lists containing the probabilities of the alternatives of the corresponding nest
    """
    probas_all = []
    sum_nests = []
    nb_nests = len(nests)
    for k in range(nb_nests):
        representative_utilities = nests[k]["representative_utilities"]
        lamb = nests[k]["lambda"]
        sum_nests.append(sum(np.exp([i / lamb for i in representative_utilities])))
    for k in range(nb_nests):
        probas_nest = []
        representative_utilities = nests[k]["representative_utilities"]
        lamb = nests[k]["lambda"]
        for representative_utility in representative_utilities:
            numerator = np.exp(representative_utility / lamb) * (sum_nests[k] ** (lamb - 1))
            normalization = np.sum(sum_nests[k] ** nests[k]["lambda"] for k in range(nb_nests))
            probas_all.append(numerator / normalization)
        # probas_all.append(probas_nest)

    return probas_all


if __name__ == '__main__':
    beta = 0.02
    k1 = 1.5
    k2 = 1.5

    X = [50, 70]
    X_logit = [0, k1 - beta * X[0], k2 - beta * X[1]]
    # X_logit = [0, k2 - beta * X[1]]
    print(compute_probas_logit(X_logit))

    nest_1 = {}
    nest_1["lambda"] = 1
    nest_1["representative_utilities"] = [0]

    X_nestlogic = [k1 - beta * X[0], k2 - beta * X[1]]
    nest_2 = {}
    nest_2["lambda"] = 0.3
    nest_2["representative_utilities"] = X_nestlogic

    nests = [nest_1, nest_2]

    print(compute_probas_nested_logit(nests))

    f2 = 100
    lamb_list = [0.02, 0.1, 0.5]
    plt.figure()
    for lamb in lamb_list:
        ord_nested = []
        ord = []
        absc = []
        for f1 in range(0, 200):
            X = [f1, f2]
            X_logit = [0, k1 - beta * X[0], k2 - beta * X[1]]
            P = compute_probas_logit(X_logit)
            X_nestlogic = [k1 - beta * X[0], k2 - beta * X[1]]
            nest_2 = {}
            nest_2["lambda"] = lamb
            nest_2["representative_utilities"] = X_nestlogic
            nests = [nest_1, nest_2]
            P_nested = compute_probas_nested_logit(nests)
            absc.append(f1)
            ord.append(P[1])
            ord_nested.append(P_nested[1][0])
        plt.plot(ord_nested, label="Lambda = "+str(lamb))
    plt.ylabel("Purchase probability of A1")
    plt.xlabel("Prices of A1")
    plt.axvline(x=f2, color='g', label="Price of A2", linestyle='--')
    plt.plot(absc, ord, label="Logit model (Lambda = 1)")
    plt.legend()
    plt.show()

    beta = 0.02
    lamb = 1
    plt.figure()
    k1 = 1
    k2 = 1.5
    al1 = []
    al2 = []
    nogo = []
    absc = []
    for f1 in range(0, 200):
        X = [f1, f1]
        X_nestlogic = [k1 - beta * X[0], k2 - beta * X[1]]
        nest_2 = {}
        nest_2["lambda"] = lamb
        nest_2["representative_utilities"] = X_nestlogic
        nests = [nest_1, nest_2]
        P_nested = compute_probas_nested_logit(nests)
        absc.append(f1)
        al1.append(P_nested[1][0])
        al2.append(P_nested[1][1])
        nogo.append(P_nested[0][0])
    plt.plot(al1, label="Purchase probability of A1")
    plt.plot(al2, label="Purchase probability of A2")
    plt.xlabel("Prices of A1")
    plt.axvline(x=f2, color='g', label="Price of A2", linestyle='--')
    plt.plot(nogo, label="Probability of not going")
    plt.legend()
    plt.show()


    lamb = 0.7
    alpha = 0.8
    ord = []
    A = [k for k in range(50, 200)]

    D = [lamb * np.exp(-alpha*(p/A[0] - 1)) for p in A]
    nD = [1 - lamb * np.exp(-alpha*(p/A[0] - 1)) for p in A]

    plt.plot(A, D, label="probability of going")
    plt.plot(A, nD, label="probability of not going")
    plt.legend()
    plt.show()

    k = 0
    p = 200
    beta = np.log((1 / lamb) * np.exp(alpha*( p / A[0] - 1)) - 1 ) / p

    # beta = [np.log((1 / lamb) * np.exp(alpha*( p / A[0] - 1)) - 1 ) / p for p in A]
    representative_utilities = [0, -beta*p, -beta*50]
    D_logit = compute_probas_logit(representative_utilities)
    # D_logit = [np.exp(k - beta[A.index(p)]*p)/(1 + np.exp(k - beta[A.index(p)]*p)) for p in A]
    # np.log((1 / lamb) * np.exp(alpha*( p / A[0] - 1)) - 1 ) / p

    # plt.plot(A, D_logit)
    # plt.show()



