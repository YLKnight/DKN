import os
import pandas as pd
from operators import *
from Gnrt_data import *
import matplotlib.pyplot as plt
from DKN_pro import DKN_R
from sklearn.metrics import mean_squared_error

params_model = {'max_itr': 10, 'tol': 1e-5}
task = 'reg'
num_of_Circles = 1
D, P = 128, 128
n, r = 1000, 0.8
N = int(n / r)
SoN = 1

C = Gnrt_circle((D, P), (40, 88), 10)
X, Y = Gnrt_data(C, N, sigma_of_X=1, sigma_of_noise=SoN)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=r, random_state=0)
print('Data | Sample size | Train: {}, Test: {}'.format(len(Y_train), len(Y_test)))
print('Y | S.D. | Train: {:.1f}, Test: {:.1f}'.format(Y_train.std(), Y_test.std()))

# ########################## DKN #######################
L = 7
blockshapes = [[2, 2] for i in range(L)]
R = 1
dkn = DKN_R(blockshapes, task=task, R=R)
C_hat, loss_list = dkn.fit((X_train, Y_train), (X_test, Y_test), **params_model)

plt.plot(loss_list[0], label='Training')
plt.plot(loss_list[1], label='Validation')
plt.legend()
plt.show()

rmse = mean_squared_error(C, C_hat, squared=False)
print('For C | RMSE: {:.4f}'.format(rmse))
Y_inhat = dkn.predict(X_train)
Train_rmse = mean_squared_error(Y_train, Y_inhat, squared=False)
Y_outhat = dkn.predict(X_test)
Test_rmse = mean_squared_error(Y_test, Y_outhat, squared=False)
print('For Y | Train RMSE: {:.4f}, Test RMSE: {:.4f}'.format(Train_rmse, Test_rmse))

Table = [[rmse, Train_rmse, Test_rmse]]
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


