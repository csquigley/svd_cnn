# svd_cnn
CNN Trained on Low Rank Reconstructed Images Using Singular Value Decomposition 

In this project I first built a converter to take the CIFAR-10 images and convert them into low rank approximations. 
I then built a standard CNN to classify the images. I wanted to know how well a CNN would perform when training with images that
contain less information. 

Interestingly, the CNN still performs quite well, at around 50% accuracy, even when the rank is so low that I cannot tell what I am looking at.
