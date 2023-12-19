# autoencoder_attacks
A repository to generate adversarial examples for Autoencoders

## Training autoencoders

Run the file `autoencoder_experiments.py ` from root folder using `python train_aautoencoders/autoencoder_experiments.py` . In this file you will find `train_AE_MLP`, `train_AE_REG`, `train_CNN_AE`, `train_Contra_AE`, `train_MLPVAE`, `train_CNN_VAE`. Declare the model you want to train as `True`. 

For example, to train the model `train_CNN_AE`, set `train_CNN_AE = True` and run `python train_aautoencoders/autoencoder_experiments.py`. 