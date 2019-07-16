import numpy as np


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
            probas_nest.append(numerator / normalization)
        probas_all.append(probas_nest)

    return probas_all


if __name__ == '__main__':
    beta = 0.01
    k1 = 1.5
    k2 = 1.5

    X = [50, 200]
    X_logit = [0, k1 - beta * X[0], k2 - beta * X[1]]

    print(compute_probas_logit(X_logit))

    nest_1 = {}
    nest_1["lambda"] = 1
    nest_1["representative_utilities"] = [0]

    X_nestlogic = [k1 - beta * X[0], k2 - beta * X[1]]
    nest_2 = {}
    nest_2["lambda"] = 0.1
    nest_2["representative_utilities"] = X_nestlogic

    nests = [nest_1, nest_2]

    print(compute_probas_nested_logit(nests))
