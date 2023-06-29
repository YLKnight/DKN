import os
import pandas as pd
from operators import *
from Gnrt_data import *
from DKN_pro import DKN_R
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from criterion import Acc_class
rng = np.random.RandomState(666)

params_model = {'max_itr': 10, 'tol': 1e-5}
task = 'cla'
num_of_Circles = 1
D, P = 128, 128
n, r = 1000, 0.8
N = int(n / r)
SoN = 1

C = Gnrt_circle((D, P), (40, 88), 10)
X, logit = Gnrt_data(C, N, sigma_of_X=1, sigma_of_noise=0)
Y = np.array([rng.binomial(1, p) for p in expit(logit)])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=r, random_state=0)
print('Data | Sample size | Train: {}, Test: {}'.format(len(Y_train), len(Y_test)))
print('Y | Positive Rate | Train: {:.2f}, Test: {:.2f}'.format(Y_train.sum() / len(Y_train), Y_test.sum() / len(Y_test)))

# ########################## DKN - Alternating #######################
L = 7
blockshapes = [[2, 2] for i in range(L)]
R = 1
dkn = DKN_R(blockshapes, task=task, R=R)
C_hat, loss_list = dkn.fit((X_train, Y_train), (X_test, Y_test), **params_model)

rmse = mean_squared_error(C.ravel(), C_hat.ravel(), squared=False)
print('For C | RMSE: {:.4f}'.format(rmse))
Y_inprob = dkn.predict(X_train)
Train_Acc = Acc_class(Y_train, Y_inprob)
Y_outprob = dkn.predict(X_test)
Test_Acc = Acc_class(Y_test, Y_outprob)
print('For Y | Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(Train_Acc, Test_Acc))

Table = [[rmse, Train_Acc, Test_Acc]]
SUMMARY = pd.DataFrame(Table, index=['DKN'], columns=['Estimate', 'Train', 'Test'])
print(SUMMARY)

# Visualization
grid = plt.GridSpec(1, 2, wspace=0.3, hspace=0.3)
plt.figure(figsize=(10, 5))
plt.suptitle('Recovery', fontsize=15, fontweight='bold')
plt.subplot(grid[0, 0])
plt.title('True C')
plt.imshow(C)
plt.colorbar()
plt.subplot(grid[0, 1])
plt.title('C_hat from DKN')
plt.imshow(C_hat)
plt.colorbar()
plt.show()
