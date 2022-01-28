from conditional_autoencoder import train_autoencoder

track = True # track losses during training using wandb

train_autoencoder(batch_size=64, track=track, epochs=250, plot_images=25)

