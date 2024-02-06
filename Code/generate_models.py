import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.widgets import Button

folder = "D:\Studies\Sem8\Project\\3DCGAN\Trained_Generator"
model_types = ["disc_simple", "disc_complex", "sphere_simple", "sphere_complex"]

def generate_latent_points(input_shape, n_models=1):
    latent_input = np.random.randn(n_models*input_shape[0]*input_shape[1]*input_shape[2])
    latent_input = latent_input.reshape(n_models, input_shape[0], input_shape[1], input_shape[2])
    labeled_input = np.random.randint(2, size=(n_models, input_shape[0], input_shape[1], input_shape[2]))
    return [latent_input, labeled_input]

def show_model(model):
    X, Y, Z = [], [], []
    for i in range(model.shape[0]):
        for j in range(model.shape[1]):
            for k in range(model.shape[2]):
                if(model[i,j,k] > 0.5):
                    X.append(i)
                    Y.append(j)
                    Z.append(k)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in zip(X, Y, Z):
        ax.bar3d(x, y, z, 1, 1, 1, color='grey', alpha=1)
    ax.set_zlim3d(-10, 20)
    ax.set_title("Disc Simple Generated")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

for model_type in model_types:
    generator = tf.keras.models.load_model(folder+"\\"+model_type+"_generator.h5")
    generator.compile()

    latent_input, labeled_input = generate_latent_points((39,39,4))
    new_model = generator([latent_input, labeled_input], training=False)
    # new_model = generator.predict([latent_input, labeled_input])
    for i in range(new_model.shape[1]):
        for j in range(new_model.shape[2]):
            for k in range(new_model.shape[3]):
                if(new_model[0,i,j,k] > 0):
                    new_model[0,i,j,k] = 1
                else:
                    new_model[0,i,j,k] = 0
    print(new_model[:5])
    # show_model(new_model[0])

    break