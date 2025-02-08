# image-processing
tensorflow image retrievetion
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import glob
import urllib.request
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

layers = keras.layers
K = tf.keras.backend

os.makedirs("images", exist_ok=True)

def download_image(i):
    url = f"https://picsum.photos/128/128?random={i}"
    path = f"images/image_{i}.jpg"
    urllib.request.urlretrieve(url, path)

with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download_image, range(500))

image_paths = glob.glob("images/*.jpg")

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return image.astype(np.float32)

images = np.asarray([load_and_preprocess_image(path) for path in image_paths])

def random_pixel_dropout(image, dropout_rate=0.3):
    mask = np.random.rand(*image.shape) > dropout_rate
    return image * mask, mask

masked_images, _ = zip(*[random_pixel_dropout(img) for img in images])
masked_images = np.asarray(masked_images)

dataset = tf.data.Dataset.from_tensor_slices((masked_images, images)) \
    .shuffle(len(images)) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)

latent_dim = 128

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

encoder_input = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(16 * 16 * 256, activation='relu')(decoder_input)
x = layers.Reshape((16, 16, 256))(x)
x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2)(x)
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
decoder = keras.Model(decoder_input, decoded, name="decoder")

class VAE(keras.Model):
    def _init_(self, encoder, decoder, **kwargs):
        super(VAE, self)._init_(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def compute_loss(self, x, y):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.square(y - reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return reconstruction_loss + 0.0001 * kl_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
vae.fit(dataset, epochs=50)

_, _, z_sample = encoder.predict(masked_images)
np.save("compressed_data.npy", z_sample)

def reconstruct_images():
    compressed_data = np.load("compressed_data.npy")
    reconstructed_images = decoder.predict(compressed_data)
    
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    for i in range(5):
        axes[i, 0].imshow(masked_images[i])
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed_images[i])
        axes[i, 1].axis('off')
    axes[0, 0].set_title("Masked")
    axes[0, 1].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

reconstruct_images()

def build_deblurring_model(input_shape=(128, 128, 3)):
    inputs = keras.Input(shape=input_shape)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)
    b = layers.Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    b = layers.Conv2D(512, (3,3), activation='relu', padding='same')(b)
    u1 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(b)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    u2 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c5)
    u3 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u3)
    c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c6)
    outputs = layers.Conv2D(3, (1,1), activation='sigmoid')(c6)
    model = keras.Model(inputs, outputs, name="DeblurringUNet")
    return model

deblur_model = build_deblurring_model(input_shape=(128, 128, 3))
deblur_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

_, _, z_sample = encoder.predict(masked_images)
blurred_images = decoder.predict(z_sample)

deblur_model.fit(blurred_images, images, epochs=50, batch_size=32, validation_split=0.1)

def reconstruct_and_deblur_images():
    _, _, z_sample = encoder.predict(masked_images)
    blurred_images = decoder.predict(z_sample)
    deblurred_images = deblur_model.predict(blurred_images)
    fig, axes = plt.subplots(5, 3, figsize=(15, 10))
    for i in range(5):
        axes[i, 0].imshow(masked_images[i])
        axes[i, 0].axis('off')
        axes[i, 1].imshow(blurred_images[i])
        axes[i, 1].axis('off')
        axes[i, 2].imshow(deblurred_images[i])
        axes[i, 2].axis('off')
    axes[0, 0].set_title("Masked")
    axes[0, 1].set_title("Reconstructed (Blurred)")
    axes[0, 2].set_title("Deblurred (Crystal Clear)")
    plt.tight_layout()
    plt.show()

reconstruct_and_deblur_images()
