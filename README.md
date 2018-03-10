# Siamese-Networks-for-One-Shot-Learning (Under Construction)

This repository was created for me to familiarize with One Shot Learning. The code uses Keras library and the Omniglot dataset.
This repository tries to implement the code for Siamese Neural Networks for One-shot Image Recognition by Koch _et al._.

## One-Shot Learning

Currently most deep learning models need generally thousands of labeled samples per class. Data acquisition for most tasks is very expensive. The possibility to have models that could learn from one or a few samples is a lot more interesting than having the need of acquiring and labeling thousands of samples. One could argue that a young child can learn a lot of concepts without needing a large number of examples.  This is where one-shot learning appears: the task of classifying with only having access of one example of each possible class in each test task. This ability of learning from little data is very interesting and could be used in many machine learning problems. 

Despite this paper is focused on images, this concept can be applied to many fields. To fully understand the problem we should describe what is considered an example of an one-shot task. Given a test sample, X, an one-shot task would aim to classify this test image into one of C categories. For this support set of samples with a representing N unique categories (N-way one shot task) is given to the model in order to decide what is the class of the test images. Notice that none of the samples used in this one-shot task have been seen by the model (the categories are different in training and testing). 

Frequently for one-shot learning tasks, the Omniglot dataset is used for evaluating the performance of the models. Let’s take a deeper look to this database, since it was the dataset used in the paper (MNIST was also tested but we will stick with Omniglot).

## Omniglot Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/36079867-c94b19fe-0f7f-11e8-9ef8-6f017d214d43.png" alt="Omniglot Dataset"/>
</p>

The Omniglot dataset consists in 50 different alphabets, 30 used in a background set and 20 used in a evaluation set. Each alphabet has a number of characters from 14 to 55 different characters drawn by 20 different subjects, resulting in 20 105x105 images for each character. The background set should be used in training for hyper parameter tuning and feature learning, leaving the final results to the remaining 20 alphabets, never seen before by the models trained in the background set. Despite that this paper uses 40 background alphabets and 10 evaluation alphabets. 

This dataset is considered as sort of a MNIST transpose, where the number of possible classes is considerably higher than the number of training samples, making it suitable to one-shot tasks. 

The authors use 20-way one-shot task for evaluating the performance in the evaluation set. For each alphabet it is performed 40 different one-shot tasks, completing a total of 400 tasks for the 10 evaluation alphabets. An example of one one-shot task in this dataset can be seen in the following figure: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/36079892-1df60568-0f80-11e8-8297-a7c6beec4491.png" alt="One-Shot Task"/>
</p>

Let's dive into the methodology proposed by Koch_et al._ to solve this one-shot task problem.

## Methodology (Under Construction)

To solve this methodology, the authors propose the use of a Deep Convolutional Siamese Networks.  Siamese Nets were introduced by Bromley and Yan LeCun in the 90s for a verification problem. 
Siamese nets  are two twin networks that accept distinct inputs but are joined in by a energy function that calculates a distance metric between the outputs of the two nets. 
The weights of both networks are tied, allowing them to compute the same function. 
In this paper the weighed L1 distance between twin feature vectors is used as energy function, combined with a sigmoid activations. 

This architecture seems to be designed for verification tasks, and this is exactly how the authors approach the problem. 

In the paper a convolutional neural net was used. 3 Blocks of Cov-RELU-Max Pooling are used followed by a Conv-RELU connected to a fully-connected layer with a sigmoid function. This layer produces the feature vectors that will be fused by the L1 weighed distance layer. The output is fed to a final layer that outputs a value between 1 and 0 (same class or different class).  To assess the best architecture, Bayesian hyper-parameter tuning was performed. The best architecture is depicted in the following image:

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/36121224-71403aa0-103d-11e8-81c6-6caae24a835c.png" alt="best_architecture"/>
</p>

L2-Regularization is used in each layer, and as an optimizer it is used Stochastic Gradient Descent with momentum. As previously mentioned, Bayesian hyperparameter optimization was used to find the best parameters for the following topics:
- Layer-wise Learning Rates (search from 0.0001 to 0.1) 
- Layer-wise Momentum (search from 0 to 1)
- Layer-wise L2-regularization penalty (from 0 to 0.1)
- Filter Size from 3x3 to 20x20
- Filter numbers from 16 to 256 (using multipliers of 16)
- Number of units in the fully connected layer from 128 to 4096 (using multipliers of 16)

For training some details were used:
- The learning rate is defined layer-wise and it is decayed by 1% each epoch.
- In every layer the momentum is fixed at 0.5 and it is increased linearly each epoch until reaching a value mu.
- 40 alphabets were used in training and validation and 10 for evaluation
- The problem is considered a verification task since the train consists in classifying pairs in same or different character. - After that in evaluation phase, the test image is paired with each one of the support set characters. The pair with higher probability output is considered the class for the test image. 
- Data Augmentation was used with affine distortions (rotations, translations, shear and zoom)

## Implementation Details (Under Construction)

When comparing to the original paper, there are some differences in this implementation, namely:
- The organization of training/validation/evaluation is different from the original paper. In the paper they follow the division suggested by the paper that introduced the Omniglot dataset, while in this implementation I used a different approach: from the 30 alphabets background set, 80% (24) are used for training and 20% (6) are using for validation one-shot tasks.
- In the paper it is said that the momentum evolves linearly along epochs, but no details about this are present. Therefore I introduced a _momentum_slope_ parameter that controls how the momentum evolves across the epochs. 
- In the paper the learning rate decays 1% each epoch, while in this implementation it decays 1% each 500 iterations. 
- The hyperparameter optimization does not include the Siamese network architecture tuning. Since the paper already describes the best architecture, I decided to reduce the hyperparameter space search to just the other parameters. 

### Code Details

There are two main files to run the code in this repo: 
- *train_siamese_networks.py* that allows you to train a siamese network with a specific set of parameters. 
- *bayesian_hyperparameter_optimization.py* that does Bayesian hyperparameter optimization as described in the paper. 

Both files store the tensorflow curve logs that can be consulted in tensorboard (in a logs folder that is created), also the models with higher validation one-shot task accuracy are saved in a models folder, allowing to keep the best models. 

Regarding the rest of the code:
- omniglot_loader is a class used to load the dataset and prepare it to the train and one-shot tasks. 
- image_augmentor is used by omniglot_loader to augment data like described in the paper. Most of this code is adapted from keras image generator code
- modified_sgd is an adaptation of the original keras sgd, but it is modified to allow layer_wise learning rate and momentums. 
- siamese_net is the main class that holds the model and trains it. 

**Notes:**
- I noticed that some combination of hyperparameters (especially with high learning rates) would lead to train accuracy stabilizing in 0.5, leading to output always the same probability for all images. Therefor I added some early stop conditions to the code.

## References
- Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." ICML Deep Learning Workshop. Vol. 2. 2015.

## Credits

I would like to give credit to a blog post that introduced me to this paper. The blog post has also include code for this paper, despite having some differences regarding this repo (Adam optimizer is used, layerwise learning-rate option is not available). It is a great blog post go check it out: 

- [One Shot Learning and Siamese Networks in Keras](https://sorenbouma.github.io/blog/oneshot/)
