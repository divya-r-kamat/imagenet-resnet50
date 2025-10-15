# ResNet-50 Training on ImageNet-1k Using AWS EC2

Train a ResNet-50 model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance

## Dataset : ImageNet-1k (ILSVRC 2012)

ImageNet (ILSVRC 2012) is a large-scale visual database designed for use in visual object recognition research.
It’s organized according to the WordNet hierarchy — each concept, described by a "synonym set" or synset, contains hundreds of labeled images.

Dataset Details:
- Name: ILSVRC 2012 (ImageNet-1k)
- Classes: 1000 object categories
- Training Images: 1,281,167
- Validation Images: 50,000
- Test Images: 100,000
- Source: Hugging Face Datasets - ImageNet-1k (https://huggingface.co/datasets/ILSVRC/imagenet-1k)

💡 This is the same dataset used in the original ResNet paper (“Deep Residual Learning for Image Recognition”, He et al., 2015).



## Environment Setup on AWS
