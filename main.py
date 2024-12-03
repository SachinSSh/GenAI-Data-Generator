import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.preprocessing import StandardScaler

class GenerativeDatasetModel:
    def __init__(self, input_dim, latent_dim=64, learning_rate=0.0002):
        """
        Initialize a generative AI model for dataset generation
        
        Args:
            input_dim (int): Dimension of input data
            latent_dim (int): Dimension of latent space
            learning_rate (float): Learning rate for optimization
        """
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Optimizers
        self.generator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)
        self.discriminator_optimizer = optimizers.Adam(learning_rate, beta_1=0.5)
        
        # Loss function
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)
    
    def _build_generator(self):
        """
        Create generator neural network
        
        Returns:
            tf.keras.Model: Generator neural network
        """
        model = models.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.input_dim, activation='linear')
        ])
        return model
    
    def _build_discriminator(self):
        """
        Create discriminator neural network
        
        Returns:
            tf.keras.Model: Discriminator neural network
        """
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Calculate discriminator loss
        
        Args:
            real_output (tf.Tensor): Discriminator output for real samples
            fake_output (tf.Tensor): Discriminator output for fake samples
        
        Returns:
            tf.Tensor: Discriminator loss
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        """
        Calculate generator loss
        
        Args:
            fake_output (tf.Tensor): Discriminator output for generated samples
        
        Returns:
            tf.Tensor: Generator loss
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, real_samples):
        """
        Perform a single training step
        
        Args:
            real_samples (tf.Tensor): Batch of real data samples
        
        Returns:
            tuple: Generator and discriminator losses
        """
        # Generate noise
        noise = tf.random.normal([real_samples.shape[0], self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake samples
            generated_samples = self.generator(noise, training=True)
            
            # Discriminator predictions
            real_output = self.discriminator(real_samples, training=True)
            fake_output = self.discriminator(generated_samples, training=True)
            
            # Calculate losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss, 
            self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, 
            self.discriminator.trainable_variables
        )
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def train(self, real_data, epochs=1000, batch_size=32):
        """
        Train the generative model
        
        Args:
            real_data (np.ndarray): Training dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # Normalize data
        scaler = StandardScaler()
        real_data_scaled = scaler.fit_transform(real_data)
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(real_data_scaled)
        dataset = dataset.shuffle(buffer_size=len(real_data_scaled))
        dataset = dataset.batch(batch_size)
        
        # Training loop
        for epoch in range(epochs):
            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}:")
                print(f"Generator Loss: {gen_loss.numpy()}")
                print(f"Discriminator Loss: {disc_loss.numpy()}")
    
    def generate_data(self, num_samples):
        """
        Generate new synthetic data samples
        
        Args:
            num_samples (int): Number of samples to generate
        
        Returns:
            np.ndarray: Generated synthetic data
        """
        noise = tf.random.normal([num_samples, self.latent_dim])
        generated_data = self.generator(noise, training=False)
        return generated_data.numpy()

def main():
    try:
        # Disable oneDNN custom operations to avoid floating-point warnings
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Simulate some real-world dataset (e.g., customer data)
        np.random.seed(42)
        real_data = np.random.normal(loc=0, scale=1, size=(1000, 10))
        
        # Initialize and train model
        gen_model = GenerativeDatasetModel(input_dim=10)
        gen_model.train(real_data, epochs=1000, batch_size=32)
        
        # Generate new synthetic data
        synthetic_data = gen_model.generate_data(num_samples=500)
        print("Synthetic Data Shape:", synthetic_data.shape)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
