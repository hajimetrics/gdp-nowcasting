####################################
###### ADVANCED VISUALIZATION ######
####################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1 hyparam adviz X: log10(hyparam_list1), Y: rmse
def adviz_Xlog10_Yrmse(model, period):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    X = np.log10(model.hyparam_list1)
    Y = model.valid_rmse[:, period]
    ax.scatter(X, Y, marker='o', s=100)
    ax.set_xlabel('log10 hyper parameter list')
    ax.set_ylabel('rmse')
    ax.set_title(str(model.dataset.setting) + ' increasing window nowcast validation: ' + str(model.method) + ' period ' + str(period))
    plt.show()

# 1 hyparam adviz X: log10(hyparam_list1), Y: predict_period, Z: rmse
def adviz_Xlog10_Yperiod_Zrmse(model, view_init=[30, -45 * 3.5]):
    X = np.log10(model.hyparam_list1)
    Y = np.linspace(0, model.predict_period-1, model.predict_period)
    XX, YY = np.meshgrid(X, Y) # XX, YY shape (Y, X)
    l = []
    for j in range(model.valid_rmse.shape[1]): # model.valid_rmse.shape = (X, Y) 
        for i in range(model.valid_rmse.shape[0]): 
            l.append(model.valid_rmse[i][j])
    #Z = np.array(model.valid_rmse).T
    l = np.array(l)
    Z = l.reshape(len(Y), len(X)) #順番注意、Z shape (Y, X)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(XX, YY, Z, marker='o')
    #ax.contour(XX, YY, Z)
    #ax.plot_surface(XX, YY, Z)
    ax.plot_wireframe(XX, YY, Z)
    ax.set_xlabel('log10 hyper parameter list')
    ax.set_ylabel('predict period')
    ax.set_zlabel('rmse')
    ax.set_title(str(model.dataset.setting) + ' increasing window nowcast validation: ' + str(model.method))
    ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.show()

# 2 hyparam adviz X: log10(hyparam_list1), Y: log10(hyparam_list2), Z: rmse
def adviz_Xlog10_Ylog10_Zrmse(model, period, view_init=[30, -45 * 3.5]):
    X = np.log10(model.hyparam_list1)
    Y = np.log10(model.hyparam_list2)
    XX, YY = np.meshgrid(X, Y)
    vmse = model.valid_rmse[:, period]
    l = []
    for i in vmse: # xr.DataArray.values取り出し
        l.append(i)
    l = np.array(l)
    Z = l.reshape(len(model.hyparam_list2), len(model.hyparam_list1)) #順番注意、Z shape (Y, X)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(XX, YY, Z, marker='o')
    #ax.contour(XX, YY, Z)
    #ax.plot_surface(XX, YY, Z)
    ax.plot_wireframe(XX, YY, Z)
    ax.set_xlabel('log10 hyper parameter list 1')
    ax.set_ylabel('log10 hyper parameter list 2')
    ax.set_zlabel('rmse')
    ax.set_title(str(model.dataset.setting) + ' increasing window nowcast validation: ' + str(model.method) + ' period ' + str(period))
    ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.show()
    
# 2 hyparam adviz X: log10(hyparam_list1), Y: log10(hyparam_list2), Z: rmse
def adviz_X_Y_Zrmse(model, period, view_init=[30, -45 * 3.5]):
    X = model.hyparam_list1
    Y = model.hyparam_list2
    XX, YY = np.meshgrid(X, Y)
    vmse = model.valid_rmse[:, period]
    l = []
    for i in vmse: # xr.DataArray.values取り出し
        l.append(i)
    l = np.array(l)
    Z = l.reshape(len(model.hyparam_list2), len(model.hyparam_list1)) #順番注意、Z shape (Y, X)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter3D(XX, YY, Z, marker='o')
    #ax.contour(XX, YY, Z)
    #ax.plot_surface(XX, YY, Z)
    ax.plot_wireframe(XX, YY, Z)
    ax.set_xlabel('log10 hyper parameter list 1')
    ax.set_ylabel('log10 hyper parameter list 2')
    ax.set_zlabel('rmse')
    ax.set_title(str(model.dataset.setting) + ' increasing window nowcast validation: ' + str(model.method) + ' period ' + str(period))
    ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.show()