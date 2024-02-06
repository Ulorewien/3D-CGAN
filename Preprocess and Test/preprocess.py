import pandas as pd
import numpy as np
import time

'''
The csv file contains the following columns:
['x', 'y', 'z', 'design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 
    'force_x', 'force_y', 'force_z', 'density']

Number of samples for training and testing:
1) Disc simple - (1509, 200) (88%, 12%)
2) Disc complex - (7419, 200) (97%, 3%)
3) Sphere simple - (150, 38) (79%, 21%)
4) Sphere complex - (378, 38) (90%, 10%)

X and Y data for GAN:
X - ['design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 'force_x', 'force_y', 'force_z']
Y - ['x', 'y', 'z', 'density']

Number of voxels in the model:
1) Disc - 6084
2) Sphere - 31941

Number of points on X,Y,Z axes:
1) Disc - (39, 39, 4)
1) Sphere - (39, 39, 21)
'''

folder = "D:\Studies\Sem8\Project\\3D_Dataset"
models = ["disc_simple", "disc_complex", "sphere_simple", "sphere_complex"]

def get_shape(model_type):
    if(model_type == "disc_simple"):
        return [(1509,6084,7),(200,6084,7),(1509,6084,4),(200,6084,4)]  # (x_train, x_test, y_train, y_test)
    if(model_type == "disc_complex"):
        return [(7419,6084,7),(200,6084,7),(7419,6084,4),(200,6084,4)]  
    if(model_type == "sphere_simple"):
        return [(150,31941,7),(38,31941,7),(150,31941,4),(38,31941,4)]   
    if(model_type == "sphere_complex"):
        return [(378,31941,7),(38,31941,7),(378,31941,4),(38,31941,4)]

def get_train(model_type, shape_x, shape_y):
    x_train = np.zeros(shape_x)
    y_train = np.zeros(shape_y)
    for i in range(shape_x[0]):
        df = pd.read_csv(folder + "\\" + model_type + "_train\\" + str(i) + ".csv", header=None, names=['x', 'y', 'z', 'design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 'force_x', 'force_y', 'force_z', 'density'])
        x_train[i] = df[['design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 'force_x', 'force_y', 'force_z']].to_numpy()
        y_train[i] = df[["x","y","z",'density']].to_numpy()

    return x_train, y_train

def get_test(model_type, shape_x, shape_y):
    x_test = np.zeros(shape_x)
    y_test = np.zeros(shape_y)
    for i in range(shape_x[0]):
        df = pd.read_csv(folder + "\\" + model_type + "_test\\" + str(i) + ".csv", header=None, names=['x', 'y', 'z', 'design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 'force_x', 'force_y', 'force_z', 'density'])
        x_test[i] = df[['design_space', 'dirichlet_x', 'dirichlet_y', 'dirichlet_z', 'force_x', 'force_y', 'force_z']].to_numpy()
        y_test[i] = df[["x","y","z",'density']].to_numpy()

    return x_test, y_test

def get_train_test_data(model_type):
    shape = get_shape(model_type)
    x_train, y_train = get_train(model_type, shape[0], shape[2])
    x_test, y_test = get_test(model_type, shape[1], shape[3])
    return x_train, y_train, x_test, y_test

def get_shape_xy(model_type):
    if(model_type[0] == "d"):
        return (39, 39, 4)
    elif(model_type[0] == "s"):
        return (39, 39, 21)

start = time.time()

for model_type in models:
    model_start = time.time()
    x_train, y_train, x_test, y_test = get_train_test_data(model_type)
    shape = get_shape_xy(model_type)
    voxel_data_train = []
    for model in y_train:
        model_data = np.zeros(shape)
        for i in range(shape[0]*shape[1]*shape[2]):
            x = int(model[i][0])
            y = int(model[i][1])
            z = int(model[i][2])
            d = int(model[i][3])
            model_data[x][y][z] = d
        voxel_data_train.append(model_data)
    voxel_data_train = np.array(voxel_data_train)
    np.save(model_type+"_voxel_data_train.npy", voxel_data_train)

    voxel_data_test = []
    for model in y_test:
        model_data = np.zeros(shape)
        for i in range(shape[0]*shape[1]*shape[2]):
            x = int(model[i][0])
            y = int(model[i][1])
            z = int(model[i][2])
            d = int(model[i][3])
            model_data[x][y][z] = d
        voxel_data_test.append(model_data)
    voxel_data_test = np.array(voxel_data_test)
    np.save(model_type+"_voxel_data_test.npy", voxel_data_test)

    labeled_data_train = []
    for i in range(x_train.shape[0]):
        model_x = x_train[i]
        model_y = y_train[i]
        model_data = np.zeros(shape)
        for i in range(shape[0]*shape[1]*shape[2]):
            fx = int(model_x[i][4])
            fy = int(model_x[i][5])
            fz = int(model_x[i][6])
            if(fx != 0 or fy != 0 or fz != 0):
                x = int(model_y[i][0])
                y = int(model_y[i][1])
                z = int(model_y[i][2])
                model_data[x][y][z] = 1
        labeled_data_train.append(model_data)
    labeled_data_train = np.array(labeled_data_train)
    np.save(model_type+"_labeled_data_train.npy", labeled_data_train)

    labeled_data_test = []
    for i in range(x_test.shape[0]):
        model_x = x_test[i]
        model_y = y_test[i]
        model_data = np.zeros(shape)
        for i in range(shape[0]*shape[1]*shape[2]):
            fx = int(model_x[i][4])
            fy = int(model_x[i][5])
            fz = int(model_x[i][6])
            if(fx != 0 or fy != 0 or fz != 0):
                x = int(model_y[i][0])
                y = int(model_y[i][1])
                z = int(model_y[i][2])
                model_data[x][y][z] = 1
        labeled_data_test.append(model_data)
    labeled_data_test = np.array(labeled_data_test)
    np.save(model_type+"_labeled_data_test.npy", labeled_data_test)
    model_end = time.time()
    print(f"{model_type} time taken: {(model_end-model_start)*1000:.2f}ms")

end = time.time()
print(f"Total time taken: {(end-start)*1000:.2f}ms")