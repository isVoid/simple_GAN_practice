# Generative Adversarial Network

![Alt Text](https://github.com/isVoid/simple_GAN_practice/raw/master/gan.gif)

## To run the model:

    python model.py

## Require: see requirements.txt

## Folder structure:

* GAN/

  * log/--------Where tensorboard log is saved

  * mnist/------Where mnist dataset is placed

  * model/------Where model and checkpoint file is saved

  * output/-----Where generated images from each epoch is saved

  * model.py---defines GAN model and solver

  * ops.py-----defines ops used by GAN

  * utils.py---miscellaneous helper functions

## Noteworthy details

* Generator:

  * fc1024 bn lrelu -> fc128\*7\*7 bn lrelu -> conv_transpose 14\*14\*64 bn lrelu -> conv_transpose 28\*28\*1 tanh

* Discriminator:

  * convf64 bn lrelu -> convf128 bn lrelu -> fc1024 bn lrelu -> fc1 sigmoid

* Dataset:

  * MNIST input normalized to [-1, 1]

* Initialization:

  * Xavier Initialization (Makes a huge difference)

* Activation:
  * leaky relu 0.02

* Used batch_norm layer from tf.contrib.layers
