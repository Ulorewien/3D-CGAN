import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Hyperparameters
latent_dim = 100
input_shape = (39, 39, 4, 1)
force_shape = (39, 39, 4, 1)
folder = "D:\Studies\Sem8\Project\\3DCGAN\Data"
model_type = "disc_simple"
parts = np.load(folder+"\\"+model_type+"_labeled_data_train.npy")
forces = np.load(folder+"\\"+model_type+"_voxel_data_train.npy")
epochs = 100
batch_size = 128

def build_generator():
    inputs = keras.Input(shape=(latent_dim,))
    forces = keras.Input(shape=force_shape)

    # Reshape the force input and concatenate with the latent vector
    forces_reshaped = layers.Reshape(input_shape)(forces)
    concatenated = layers.Concatenate(axis=-1)([inputs, forces_reshaped])

    # Upsample the concatenated input using Conv3DTranspose layers
    x = layers.Dense(8 * 8 * 8 * 128)(concatenated)
    x = layers.Reshape((8, 8, 8, 128))(x)
    x = layers.Conv3DTranspose(64, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv3DTranspose(32, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv3DTranspose(1, 5, strides=2, padding="same")(x)
    outputs = layers.Activation("sigmoid")(x)

    # Instantiate the generator model
    generator = keras.Model([inputs, forces], outputs, name="generator")
    return generator

def build_discriminator():
    inputs = keras.Input(shape=input_shape)

    # Use Conv3D layers to downsample the input
    x = layers.Conv3D(32, 5, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv3D(64, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv3D(128, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Flatten and classify as real or fake
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)

    # Instantiate the discriminator model
    discriminator = keras.Model(inputs, outputs, name="discriminator")
    return discriminator

def build_cgan(discriminator, generator):
    # Combine the generator and discriminator into a CGAN
    discriminator.trainable = False
    inputs = keras.Input(shape=(latent_dim,))
    forces = keras.Input(shape=force_shape)
    generated_parts = generator([inputs, forces])
    discriminator_output = discriminator(generated_parts)
    cgan = keras.Model([inputs, forces], discriminator_output, name="cgan")
    cgan.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        input_shape=input_shape,
    )

    return cgan

def train(generator, discriminator, cgan, parts, forces, latent_dim, epochs, batch_size):
    # Rescale the parts and forces to be between -1 and 1
    parts = (parts.astype("float32") - 0.5) * 2
    forces = (forces.astype("float32") - 0.5) * 2
    
    # Calculate the number of batches per epoch
    batch_per_epoch = parts.shape[0] // batch_size
    
    # Define the labels for real and fake parts
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Start the training loop
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            # Generate a batch of latent vectors
            latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            
            # Get a batch of real parts and forces
            batch_parts = parts[batch * batch_size : (batch + 1) * batch_size]
            batch_forces = forces[batch * batch_size : (batch + 1) * batch_size]
            
            # Generate a batch of fake parts using the generator
            generated_parts = generator.predict([latent_vectors, batch_forces])
            
            # Train the discriminator on real and fake parts
            discriminator_loss_real = discriminator.train_on_batch(batch_parts, real)
            discriminator_loss_fake = discriminator.train_on_batch(generated_parts, fake)
            discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_fake)
            
            # Train the generator on the combined loss
            generator_loss = cgan.train_on_batch([latent_vectors, batch_forces], real)
            
            # Print the progress
            print(
                f"Epoch {epoch+1}/{epochs}, Batch {batch}/{batch_per_epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}"
            )

# Build the generator
generator = build_generator()

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    input_shape=input_shape,
)

cgan = build_cgan(discriminator, generator)

train(generator, discriminator, cgan, parts, forces, latent_dim, epochs, batch_size)

# Save the generator
generator.save(model_type+"_generator.h5")

# Save the discriminator
discriminator.save(model_type+"_discriminator.h5")

# Save the CGAN
cgan.save(model_type+"_cgan.h5")