#!/usr/bin/env python

import numpy as np
from sklearn.metrics import silhouette_score
import kmeans
from utils import cal_balance_v2, dv_ae

if __name__ == "__main__":
    filename = [0, 2, 6, 7]
    for i in filename:
        data_name = "2d-4c-no" + str(i) + ".dat"
        f = open("no" + str(i) + "_size.txt", "a")
        Data = np.loadtxt(data_name)
        X = Data[:, 0:-1].T
        sens = Data[:, -1]
        k = 4
        param = 1e1
        n = Data.shape[0]
        tmp = [i for i in range(10, 11)]
        for j in tmp:
            size = n/(j*k)
            t = 0
            while (t < 20):
                clustering = kmeans.FairKMeans(n_clusters=k, fair_param=param).fit(X, sens)
                F = clustering.F_
                if any(sum(F) < size):
                    continue
                else:
                    labels = np.nonzero(F)[1]
                    score = silhouette_score(X.T, labels)
                    cost = clustering.inertia_
                    balance = cal_balance_v2(labels, sens)
                    ae = dv_ae(labels, sens)
                    f.write(str(j) + "  " + str(score) + "  " + str(cost)
                            + "  " + str(balance) + "  " + str(ae) + "\n")
                    t += 1
        f.close()
