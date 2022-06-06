## ViT Without Flattening

Experiment of adding convolutional layers to replace the flattening operation of ViT.

The inputs to ViT are not 1-D vectors but are the 2-D feature maps. This is different to the
paper [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808.pdf).

My ViT got bad performance on my small dataset(3k train, 1k test). Inspired by the CNN's feature of remaining the 2-D structure of the image and it good performance on this small dataset, I want to remaining the 2-D structure of the input of ViT.


#### Current Progress: Fixing the gradient issue

