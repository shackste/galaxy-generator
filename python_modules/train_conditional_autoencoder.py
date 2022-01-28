from conditional_autoencoder import train_autoencoder

track = True # track losses during training using wandb

train_autoencoder(batch_size=32, track=track, epochs=2000, plot_images=1)

