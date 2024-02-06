import tensorflow as tf
from keras import Model, layers, optimizers
import numpy as np
import time
import matplotlib.pyplot as plt

folder = "D:\Studies\Sem8\Project\\3DCGAN\Data"
model_types = ["disc_simple", "disc_complex", "sphere_simple", "sphere_complex"]

def define_discriminator(input_shape):
    # input_shape = input[0].shape
    input_model = layers.Input(shape=input_shape)
    input_label = layers.Input(shape=input_shape)
    input_layer = layers.Concatenate()([input_model, input_label])

    layer1 = layers.Conv2D(64, (2,2), strides=(1,1), padding="same")(input_layer)
    layer1 = layers.LeakyReLU(alpha=0.1)(layer1)

    layer2 = layers.Conv2D(128, (2,2), strides=(1,1), padding="same")(layer1)
    layer2 = layers.BatchNormalization()(layer2)
    layer2 = layers.LeakyReLU(alpha=0.1)(layer2)

    layer3 = layers.Conv2D(256, (2,2), strides=(1,1), padding="same")(layer2)
    layer3 = layers.BatchNormalization()(layer3)
    layer3 = layers.LeakyReLU(alpha=0.1)(layer3)

    layer4 = layers.Conv2D(512, (2,2), strides=(1,1), padding="same")(layer3)
    layer4 = layers.BatchNormalization()(layer4)
    layer4 = layers.LeakyReLU(alpha=0.1)(layer4)

    layer5 = layers.Conv2D(512, (2,2), padding="same")(layer4)
    layer5 = layers.BatchNormalization()(layer5)
    layer5 = layers.LeakyReLU(alpha=0.1)(layer5)

    output_layer = layers.Conv2D(4, (2,2), strides=(1,1), padding="same", activation="sigmoid")(layer5)

    model = Model([input_model, input_label], output_layer)
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model

def define_generator(input_shape):
    # input_shape = input[0].shape
    input_latent = layers.Input(shape=input_shape)
    input_label = layers.Input(shape=input_shape)
    input_layer = layers.Concatenate()([input_latent, input_label])

    layer1 = layers.Conv2D(64, (2,2), strides=(3,3), padding="same")(input_layer)
    layer1 = layers.LeakyReLU(alpha=0.1)(layer1)

    layer2 = layers.Conv2D(128, (2,2), strides=(1,1), padding="same")(layer1)
    layer2 = layers.BatchNormalization()(layer2)
    layer2 = layers.LeakyReLU(alpha=0.1)(layer2)

    layer3 = layers.Conv2D(256, (2,2), strides=(1,1), padding="same")(layer2)
    layer3 = layers.BatchNormalization()(layer3)
    layer3 = layers.LeakyReLU(alpha=0.1)(layer3)

    layer4 = layers.Conv2D(512, (2,2), strides=(1,1), padding="same")(layer3)
    layer4 = layers.BatchNormalization()(layer4)
    layer4 = layers.LeakyReLU(alpha=0.1)(layer4)

    layer5 = layers.Conv2D(512, (2,2), strides=(1,1), padding="same")(layer4)
    layer5 = layers.BatchNormalization()(layer5)
    layer5 = layers.LeakyReLU(alpha=0.1)(layer5)

    layer6 = layers.Conv2D(512, (2,2), strides=(1,1), padding="same")(layer5)
    layer6 = layers.ReLU()(layer6)

    layer7 = layers.Conv2DTranspose(1024, (2,2), strides=(1,1), padding="same")(layer6)
    layer7 = layers.BatchNormalization()(layer7)
    layer7 = layers.ReLU()(layer7)

    layer8 = layers.Conv2DTranspose(1024, (2,2), strides=(1,1), padding="same")(layer7)
    layer8 = layers.BatchNormalization()(layer8)
    layer8 = layers.ReLU()(layer8)

    layer9 = layers.Conv2DTranspose(512, (2,2), strides=(1,1), padding="same")(layer8)
    layer9 = layers.BatchNormalization()(layer9)
    layer9 = layers.ReLU()(layer9)

    layer10 = layers.Conv2DTranspose(256, (2,2), strides=(1,1), padding="same")(layer9)
    layer10 = layers.BatchNormalization()(layer10)
    layer10 = layers.ReLU()(layer10)

    layer11 = layers.Conv2DTranspose(128, (2,2), strides=(1,1), padding="same")(layer10)
    layer11 = layers.BatchNormalization()(layer11)
    layer11 = layers.ReLU()(layer11)

    output_layer = layers.Conv2DTranspose(4, (2,2), strides=(3,3), padding="same", activation="tanh")(layer11)

    model = Model([input_latent, input_label], output_layer)

    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False
    g_noise, g_label = generator.input
    g_output = generator.output
    gan_output = discriminator([g_output, g_label])
    model = Model([g_noise, g_label], gan_output)
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    return model

def get_real_models(x_data, y_data, n_models):
    index = np.random.randint(0, x_data.shape[0], n_models)
    labels, models = x_data[index], y_data[index]
    y = np.ones((n_models, y_data.shape[1], y_data.shape[2], y_data.shape[3]))
    return [labels, models], y

def generate_latent_points(n_models, input_shape):
    latent_input = np.random.randn(n_models*input_shape[0]*input_shape[1]*input_shape[2])
    latent_input = latent_input.reshape(n_models, input_shape[0], input_shape[1], input_shape[2])
    labeled_input = np.random.randint(2, size=(n_models, input_shape[0], input_shape[1], input_shape[2]))
    return [latent_input, labeled_input]

def generate_fake_models(generator, n_models, input_shape):
    latent_input, labeled_input = generate_latent_points(n_models, input_shape)
    models = generator.predict([latent_input, labeled_input])
    labels = np.zeros((n_models, input_shape[0], input_shape[1], input_shape[2]))
    return [models, labeled_input], labels

def plot_losses(d_losses_real, d_losses_fake, g_losses):
    y = [0, 1]
    plt.plot(d_losses_real, y)
    plt.plot(d_losses_fake, y)
    plt.plot(g_losses, y)
    plt.show()

def train_gan(model_type, generator, discriminator, gan, x_train, y_train, input_shape, n_epochs=100, n_batch=128):
    d_losses_real = []
    d_losses_fake = []
    g_losses = []
    
    batch_per_epoch = int(y_train.shape[0] / n_batch)
    half_batch = int(n_batch/2)

    for epoch in range(n_epochs):
        for batch in range(batch_per_epoch):
            [labels_real, models_real], y_real = get_real_models(x_train, y_train, half_batch)
            d_loss1 = discriminator.train_on_batch([models_real, labels_real], y_real)
            [X_fake, labels_fake], y_fake = generate_fake_models(generator, half_batch, input_shape)
            d_loss2 = discriminator.train_on_batch([X_fake, labels_fake], y_fake)
            [latent_input, labeled_input] = generate_latent_points(n_batch, input_shape)
            y_gan = np.ones((n_batch, input_shape[0], input_shape[1], input_shape[2]))
            g_loss = gan.train_on_batch([latent_input, labeled_input], y_gan)
            print(f"Epoch {epoch+1} -> Batch {batch+1}/{batch_per_epoch} : d_loss_real={d_loss1:.3f} d_loss_fake={d_loss2:.3f} g_loss={g_loss:.3f}")
            d_losses_real.append(d_loss1)
            d_losses_fake.append(d_loss2)
            g_losses.append(g_loss)
    
    generator.save("Trained_Generator\\"+model_type+"_generator.h5")
    gan.save("Trained_GAN\\"+model_type+"_gan.h5")
    # plot_losses(d_losses_real, d_losses_fake, g_losses)

start = time.time()

for model_type in model_types:
    print(f"\nModel type - {model_type}\n")

    x_train = np.load(folder+"\\"+model_type+"_labeled_data_train.npy")
    x_test = np.load(folder+"\\"+model_type+"_labeled_data_test.npy")
    y_train = np.load(folder+"\\"+model_type+"_voxel_data_train.npy")
    y_test = np.load(folder+"\\"+model_type+"_voxel_data_test.npy")
    print(f"X train shape: {x_train.shape}\nX test shape: {x_test.shape}\nY train shape: {y_train.shape}\nY test shape: {y_test.shape}\n")

    x_train = x_train.astype('float32')
    x_train = (x_train - 0.5) * 2.0
    x_test = x_test.astype('float32')
    x_test = (x_test - 0.5) * 2.0
    y_train = y_train.astype('float32')
    y_train = (y_train - 0.5) * 2.0
    y_test = y_test.astype('float32')
    y_test = (y_test - 0.5) * 2.0

    input_shape = y_train[0].shape
    n_models = 5

    discriminator = define_discriminator(input_shape)
    generator = define_generator(input_shape)
    gan = define_gan(generator, discriminator)

    train_gan(model_type, generator, discriminator, gan, x_train, y_train, input_shape)

    # break

end = time.time()
print(f"Total time taken: {(end-start)*1000:.2f}ms")