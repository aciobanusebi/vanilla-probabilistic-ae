# Vanilla Probabilistic Autoencoder
https://easychair.org/publications/paper/R2l1
# Paper abstract
The autoencoder, a well-known neural network model, is usually fitted using a mean squared error loss or a cross-entropy loss. Both losses have a probabilistic interpretation: they are equivalent to maximizing the likelihood of the dataset when one uses a normal distribution or a categorical distribution respectively. We trained autoencoders on image datasets using different distributions and noticed the differences from the initial autoencoder: if a mixture of distributions is used the quality of the reconstructed images may increase and the dataset can be augmented; one can often visualize the reconstructed image along with the variances corresponding to each pixel. 

# Code
- .py version
  - run "run.py"; this calls "main.py" with all the possible values for the command line arguments; if you encounter problems because of the indentation, consider using "main-weird-indent.py" instead of "main.py"
- .ipynb version [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aciobanusebi/vanilla-probabilistic-ae/blob/main/AE_distribution_at_output.ipynb)
  - run for a given set of arguments the code interactively
