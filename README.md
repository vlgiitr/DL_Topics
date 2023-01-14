# Deep Learning Topics and Resources

This repo contains a list of topics which we feel that people seeking a career in deep learning should be familiar with. This list is by no means exhaustive given the pace of developments in deep learning.

## Resources for DL in General

1. **Blogs**
    - Lilian Weng’s Blog [[link](https://lilianweng.github.io/)]
    - AI Summer Blog [[link](https://theaisummer.com/learn-ai/)]
    - Colah’s Blog [[link](https://colah.github.io/)]
2. **Books**
    - Neural Networks and Deep Learning [[link](http://neuralnetworksanddeeplearning.com/)]
    - Deep Learning Book [[link](https://www.deeplearningbook.org/)]
    - Dive into Deep Learning [[link](https://d2l.ai/)]
    - Reinforcement Learning: An Introduction | Sutton and Barto [[link](http://www.incompleteideas.net/book/the-book-2nd.html)]
3. **Open Courses**
    - CS-229 Machine Learning Stanford | Andrew Ng [[youtube](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)] [[website](https://cs229.stanford.edu/)]
    - CS-231n Computer Vision Stanford [[youtube](https://youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)] [[website](http://cs231n.stanford.edu/)]
    - CS-224n Natural Language Processing [[youtube](https://youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)] [[website](https://web.stanford.edu/class/cs224n/)]
    - Introduction to Reinforcement Learning with David Silver [[youtube](https://youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)] [[website](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)]

## Mathematics

1. Linear Algebra ([[notes](http://cs229.stanford.edu/section/cs229-linalg.pdf)][[practice questions](https://www.geeksforgeeks.org/linear-algebra-gq/)])
    - 3Blue1Brown essence of linear algebra [[youtube](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)]
    - Gilbert Strang’s lectures on Linear Algebra [[link](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)] [[youtube](https://youtube.com/playlist?list=PL49CF3715CB9EF31D)]
    - Topics
        - Linear Transformations
        - Linear Dependence and Span
        - Eigendecomposition - Eigenvalues and Eigenvectors
        - Singular Value Decomposition [[blog](https://medium.com/vlgiitr/eli5-singular-value-decomposition-svd-955c151f9907)]
        
2. Probability and Statistics ([[notes](http://www.mxawng.com/stuff/notes/stat110.pdf)][[youtube series](https://www.youtube.com/user/joshstarmer)])
    - Harvard Statistics 110: Probability [[link](https://projects.iq.harvard.edu/stat110/home)] [[youtube](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)]
    - Topics
        - Expectation, Variance, and Co-variance
        - Distributions
        - Random Walks
        - Bias and Variance
            - Bias Variance Trade-off
        - Estimators
            - Biased and Unbiased
        - Maximum Likelihood Estimation [[blog](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)]
        - Maximum A-Posteriori (MAP) Estimation [[blog](https://towardsdatascience.com/probability-concepts-explained-bayesian-inference-for-parameter-estimation-90e8930e5348)]
        
3. Information Theory [[youtube](https://www.youtube.com/watch?v=ErfnhcEV1O8)]
    - (Shannon) Entropy [[blog](https://towardsdatascience.com/information-entropy-c037a90de58f)]
    - Cross Entropy, KL Divergence [[blog](https://towardsdatascience.com/entropy-cross-entropy-and-kl-divergence-explained-b09cdae917a)]
    - KL Divergence
        - Not a distance metric (unsymmetric)
        - Derivation from likelihood ratio ([Blog](https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence-b0d57ee10e0a))
        - Always greater than 0
            - Proof by Jensen's inequality ([Stack Overflow Link](https://stats.stackexchange.com/a/335201))
        - Relation with Entropy ([Explanation](https://stats.stackexchange.com/questions/265966/why-do-we-use-kullback-leibler-divergence-rather-than-cross-entropy-in-the-t-sne))

## Basics

1. Neural Networks Overview [[youtube](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)]
2. Backpropogation
    - Vanilla [[blog](http://cs231n.github.io/optimization-2/)]
    - Backpropagation in CNNs [[blog](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)]
    - Backprop through time [[blog](https://towardsdatascience.com/backpropagation-in-rnn-explained-bdf853b4e1c2)]
3. Loss Functions
    - MSE Loss
        - Derivation by MLE and MAP
    - Cross Entropy Loss
        - Binary Cross Entropy
        - Categorical Cross Entropy
4. Activation Functions (Sigmoid, Tanh, ReLU and variants) ([blog](https://mlfromscratch.com/activation-functions-explained/))
5. Optimizers 
6. Regularization
    - Early Stopping
    - Noise Injection
    - Dataset Augmentation
    - Ensembling
    - Parameter Norm Penalties
        - L1 (sparsity)
        - L2 (smaller parameter values)
    - BatchNorm [[Paper](https://github.com/vlgiitr/DL_Topics/blob/master)]
        - Internal Covariate Shift
        - BatchNorm in CNNs [[Link](https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network)]
        - Backprop through BatchNorm Layer [[Explanation](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)]
    - Dropout Regularization [[Paper](https://github.com/vlgiitr/DL_Topics/blob/master)]

## Computer Vision

1. Convolution [[youtube](https://youtu.be/8rrHTtUzyZA)]
    - Cross-correlation
    - Pooling (Average, Max Pool)
    - Strides and Padding
    - Output volume dimension calculation
    - Deconvolution (Transposed Convolution), Upsampling, Reverse Pooling [[Visualization](https://github.com/vdumoulin/conv_arithmetic#readme)]
    - Types of convolution operation [[blog](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)]
    
2. ImageNet Classification
    - AlexNet [[paper](https://paperswithcode.com/model/alexnet)] [[blog](https://medium.com/p/b93598314160)]
    - ZFNet [[paper](https://paperswithcode.com/method/zfnet)] [[blog](https://medium.com/coinmonks/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification-d1a5a0c45103)]
    - VGGNet [[paper](https://paperswithcode.com/method/vgg)] [[blog](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)]
    - InceptionNet [[paper](https://paperswithcode.com/method/inception-v3)] [[blog](https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)]
    - ResNet [[paper](https://paperswithcode.com/lib/torchvision/resnet)] [[blog](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8)]
    - DenseNet [[paper](https://paperswithcode.com/lib/timm/densenet)] [[blog](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)]
    - SENet [[paper](https://paperswithcode.com/paper/squeeze-and-excitation-networks)] [[blog](https://towardsdatascience.com/review-senet-squeeze-and-excitation-network-winner-of-ilsvrc-2017-image-classification-a887b98b2883)]
    - ViT [[paper](https://arxiv.org/abs/2010.11929)] [[blog](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)]
    - Swin Transformer [[paper](https://arxiv.org/abs/2103.14030)] [[blog](https://sh-tsang.medium.com/review-swin-transformer-3438ea335585)]
    - BEiT [[paper](https://openreview.net/forum?id=p-BhZSz59o4)] [[blog](https://sh-tsang.medium.com/review-beit-bert-pre-training-of-image-transformers-c14a7ef7e295)]
    - ConvNext [[paper](https://arxiv.org/abs/2201.03545)] [[blog](https://medium.com/augmented-startups/convnext-the-return-of-convolution-networks-e70cbe8dabcc)]
    
3. Object Detection [[blog series](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904)]
    - RCNN [[paper](https://paperswithcode.com/method/r-cnn)]
    - Fast RCNN [[paper](https://paperswithcode.com/paper/fast-r-cnn)]
    - Faster RCNN [[paper](https://paperswithcode.com/paper/faster-r-cnn-towards-real-time-object)]
    - Mask RCNN [[paper](https://paperswithcode.com/paper/mask-r-cnn)]
    - YOLO (Real-time object recognition) [[blog](https://medium.com/deelvin-machine-learning/the-evolution-of-the-yolo-neural-networks-family-from-v1-to-v7-48dd98702a3d)]
    - SSD (Single Shot Detection) [[paper](https://paperswithcode.com/method/ssd)]
    - DETR [[project page](https://alcinos.github.io/detr_page/)] [[annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)]
    
4. Semantic Segmentation
    - UNet [[paper](https://paperswithcode.com/method/u-net)]
    - DeepLab [[paper](https://paperswithcode.com/method/deeplab)]
    - MaskFormer [[paper](https://paperswithcode.com/paper/per-pixel-classification-is-not-all-you-need)] [[project page](https://bowenc0221.github.io/maskformer/)]

## Natural Language Processing

1. Recurrent Neural Networks
    - Architectures (Limitations and inspiration behind every model)
        - Vanilla [[blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)]
        - GRU, LSTMs [[blog_1](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)] [[blog_2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)]
        - Bidirectional
    - Vanishing and Exploding Gradients
    
2. Word Embeddings [[blog_1](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)] [[blog_2](https://jalammar.github.io/illustrated-bert/)]
    - Word2Vec
    - CBOW
    - Glove
    - SkipGram, NGram
    - FastText
    - ELMO
    - BERT
    
3. Transformers [[blog posts](http://jalammar.github.io/)] [[youtube series](https://youtube.com/playlist?list=PLTx9yCaDlo1UlgZiSgEjq86Zvbo2yC87d)]
    - Attention is All You Need [[blog](https://jalammar.github.io/illustrated-transformer/)] [[paper](https://arxiv.org/abs/1706.03762)] [[annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)]
    - Query-Key-Value Attention Mechanism  (Quadratic Time)
    - Position Embeddings [[blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)]
    - BERT (Masked Language Modelling) [[blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)]
    - Longe Range Sequence Modelling [[blog](https://huggingface.co/blog/long-range-transformers)]
    - ELECTRA (Pretraining Transformers as Discriminators) [[blog](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html)]
    - GPT (Causal Language Modelling) [[blog](https://openai.com/blog/gpt-3-edit-insert/)]
    - OpenAI ChatGPT [[blog](https://openai.com/blog/chatgpt/)]

## Multimodal Learning

- Vision Language Models | AI Summer [[blog](https://theaisummer.com/vision-language-models/)]
- Open AI DALL-E [[blog](https://openai.com/blog/dall-e/)]
- OpenAI CLIP [[blog](https://openai.com/blog/clip/)]
- Flamingo [[blog](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)]
- Gato [[blog](https://www.deepmind.com/blog/a-generalist-agent)]
- data2vec [[blog](https://ai.facebook.com/blog/the-first-high-performance-self-supervised-algorithm-that-works-for-speech-vision-and-text/)]
- OpenAI Whisper [[blog](https://openai.com/blog/whisper/)]

## Generative Models

1. Generative Adversarial Networks (GANs) [[blog series](https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b)]
    - Basic Idea
    - Variants
        - Vanilla GAN [[paper](https://arxiv.org/abs/1406.2661)]
        - DCGAN [[paper](https://arxiv.org/abs/1511.06434v2)]
        - Wasserstein GAN [[paper](https://arxiv.org/abs/1701.07875)]
        - Conditional GAN [[paper](https://arxiv.org/abs/1411.1784)]
    - Mode Collapse
    - GAN Hacks [[link](https://github.com/soumith/ganhacks)]
2. Variational Autoencoders (VAEs)
    - Variational Inference [[tutorial paper](https://arxiv.org/abs/1606.05908)]
    - ELBO and Loss Function derivation
3. Normalizing Flows
    - Basic Idea and Applications [[link](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)]
    

## Stable Diffusion

- Demos
    - Lexica (Stable Diffusion search engine) [[link](https://lexica.art/)]
    - Stability AI | Huggingface Spaces [[link](https://huggingface.co/spaces/stabilityai/stable-diffusion)]
- Diffusion Models in general [[paper](https://ommer-lab.com/research/latent-diffusion-models/)]
    - What are Diffusion Models? | Lil'Log [[link](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)]

- Stable Diffusion | Stability AI [[blog](https://stability.ai/blog/stable-diffusion-v2-release)] [[annotated stable diffusion](https://huggingface.co/blog/annotated-diffusion)]
- Illustrated Stable DIffusion | Jay Alammar [[blog](https://jalammar.github.io/illustrated-stable-diffusion/)]
- Stable Diffusion in downstream Vision tasks
    - DiffusionDet [[paper](https://arxiv.org/abs/2211.09788)]
    

## Keeping up with the developments in Deep Learning

- **Youtube Channels**
    - Yannic Kilcher [[link](https://www.youtube.com/@YannicKilcher)]
    - Two Minute Papers [[link](https://www.youtube.com/@TwoMinutePapers)]
- **Blogs**
    - DeepMind Blog [[link](https://deepmind.com/blog)]
    - OpenAI Blog [[link](https://openai.com/blog/tags/research/)]
    - Google AI Blog [[link](https://ai.googleblog.com/)]
    - Meta AI Blog [[link](https://ai.facebook.com/blog/)]
    - Nvidia - Deep Learning Blog [[link](https://blogs.nvidia.com/blog/category/deep-learning/)]
    - Microsoft Research Blog [[link](https://www.microsoft.com/en-us/research/blog/)]
- **Trending Reseach Papers**
    - labml [[link](https://papers.labml.ai/papers/recent/)]
    - deep learning monitor [[link](https://deeplearn.org/)]

## Contributing
We welcome contributions to add resources such as notes, blogs, or papers for a topic. Feel free to open a pull request for the same!