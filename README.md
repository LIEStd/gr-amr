# gnuradio-deep-modulation-classification
Deep Modulation classification using gnuradio and pytorch


Put your neural network model in python/models.py and name it as NN.
Or use the default cnn neural network.
if RNN are used, reshape the default input shape [batch_size, 1, 2, N] to [batch_size, N, 2] in the forward function.
