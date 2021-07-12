# Vanilla Probabilistic VAE

# Paper abstract
The autoencoder, a well-known neural network model, is usually fitted using a mean squared error loss or a cross-entropy loss. Both losses have a probabilistic interpretation: they are equivalent to maximizing the likelihood of the dataset when one uses a normal distribution or a categorical distribution respectively. We trained autoencoders on image datasets using different distributions and noticed the differences from the initial autoencoder: the quality of the reconstructed images increases and the dataset can be augmented if a mixture of distributions is used, and one can often view the reconstructed image and also the variances corresponding to each pixel. 

# Code
- .py version
  - run "run.py"; this calls "main.py" with all the possible values for the command line arguments
- .ipynb version [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/vanilla-probabilistic-ae/blob/main/AE_distribution_at_output.ipynb)
  - run for a given set of arguments the code interactively
