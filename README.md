# Deep Learning Interview Topics

This repo contains a list of topics which we feel that one should be comfortable with before appearing for a DL interview. This list is by no means exhaustive (as the field is very wide and ever growing).

## Mathematics

1. Linear Algebra([notes](http://cs229.stanford.edu/section/cs229-linalg.pdf))
	+ Linear Dependence and Span
	+ Eigendecomposition
		+ Eigenvalues and Eigenvectors
	+ Singular Value Decomposition
2. Probability and Statistics
	+ Expectation, Variance and Co-variance
	+ Distributions
	+ Bias and Variance
		+ Bias Variance Trade-off
	+ Estimators
		+ Biased and Unbiased
	+ Maximum Likelihood Estimation
	+ Maximum A Posteriori (MAP) Estimation
3. Information Theory
	+ (Shannon) Entropy
	+ Cross Entropy
	+ KL Divergence
		+ Not a distance metric
		+ Derivation from likelihood ratio ([Blog](https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence-b0d57ee10e0a))
		+ Always greater than 0
			+ Proof by Jensen's Inequality
		+ Relation with Entropy ([Explanation](https://stats.stackexchange.com/questions/265966/why-do-we-use-kullback-leibler-divergence-rather-than-cross-entropy-in-the-t-sne))


## Basics

1. Backpropogation
	+ Vanilla ([blog](http://cs231n.github.io/optimization-2/))
	+ Backprop in CNNs
		+ Gradients in Convolution and Deconvolution Layers
	+ Backprop through time
2. Loss Functions
	+ MSE Loss
		+ Derivation by MLE and MAP
	 + Cross Entropy Loss
		 + Binary Cross Entropy
		 + Categorical Cross Entropy
3. Activation Functions (Sigmoid, Tanh, ReLU and variants) ([blog](https://mlfromscratch.com/activation-functions-explained/))
4. Optimizers
5. Regularization
	+ Early Stopping
	+ Noise Injection
	+ Dataset Augmentation
	+ Ensembling
	+ Parameter Norm Penalties
		+ L1 (sparsity)
		+ L2 (smaller parameter values)
	+ BatchNorm ([Paper]())
		+ Internal Covariate Shift
		+ BatchNorm in CNNs ([Link](https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network))
		+ Backprop through BatchNorm Layer ([Explanation](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html))
	+ Dropout ([Paper]()) ([Notes](https://vlgiitr.github.io/notes/2018-08-15-Dropout/))


## Computer Vision

1. ILSVRC
	+ AlexNet
	+ ZFNet
	+ VGGNet ([Notes](https://vlgiitr.github.io/notes/2018-10-11-VGG_Notes/))
	+ InceptionNet ([Notes](https://vlgiitr.github.io/notes/2018-10-17-InceptionNet_Notes/))
	+ ResNet ([Notes](https://vlgiitr.github.io/notes/2018-10-29-ResNet_Notes/))
	+ DenseNet
	+ SENet
2. Object Recognition ([Blog](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4))
	+ RCNN ([Notes](https://vlgiitr.github.io/notes/2018-10-29-RCNN_Notes/))
	+ Fast RCNN
	+ Faster RCNN ([Notes](https://vlgiitr.github.io/notes/2018-01-02-Deep_Gen_models/))
	+ Mask RCNN
	+ YOLO v3 (Real-time object recognition) 
3. Convolution
	+ Cross-correlation
	+ Pooling (Average, Max Pool)
	+ Strides and Padding
	+ Output volume dimension calculation
	+ Deconvolution (Transpose Conv.), Upsampling, Reverse Pooling ([Visualization](https://github.com/vdumoulin/conv_arithmetic))

## Natural Language Processing

1. Recurrent Neural Networks
	+ Architectures (Limitations and inspiration behind every model) ([Blog 1](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)) ([Blog 2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
		+ Vanilla
		+ GRU
		+ LSTM
		+ Bidirectional
	+ Vanishing and Exploding Gradients
2. Word Embeddings 
	+ Word2Vec
	+ CBOW
	+ Glove
	+ FastText
	+ SkipGram, NGram
	+ ELMO
	+ OpenAI GPT
	+ BERT ([Blog](http://jalammar.github.io/illustrated-bert/))
3. Transformers ([Paper](https://arxiv.org/abs/1706.03762)) ([Code](https://nlp.seas.harvard.edu/2018/04/03/attention.html)) ([Blog](http://jalammar.github.io/illustrated-transformer/))
	+ BERT ([Paper](https://arxiv.org/abs/1810.04805))
	+ Universal Sentence Encoder

## Generative Models

1. Generative Adversarial Networks (GANs)
	+ Basic Idea
	+ Variants
		+ Vanilla GAN ([Paper](https://arxiv.org/abs/1406.2661))
		+ DCGAN
		+ Wasserstein GAN ([Paper](https://arxiv.org/abs/1701.07875))
		+ Conditional GAN ([Paper](https://arxiv.org/abs/1411.1784))
	+ Mode Collapse
	+ GAN Hacks ([Link](https://github.com/soumith/ganhacks))
2. Variational Autoencoders (VAEs)
	+ Variational Inference ([tutorial paper](https://arxiv.org/abs/1606.05908))
	+ ELBO and Loss Function derivation
3. Normalizing Flows
	+ [Basic Idea and Applications](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)


## Misc
1. Triplet Loss
2. BLEU Score
3. Maxout Networks
4. Support Vector Machines
	+ Maximal-Margin Classifier
	+ Kernel Trick
5. PCA ([Explanation](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579))
	+ PCA using neural network
		+ Architecture
		+ Loss Function
6. Spatial Transformer Networks
7. Gaussian Mixture Models (GMMs)
8. Expectation Maximization

## More Resources

1. Stanford's CS231n Lecture Notes
2. Deep Learning Book (Goodfellow et. al.)

## Contributing

We welcome contributions to add resources such as notes, blogs, or papers for a topic. Feel free to open a pull request for the same!
