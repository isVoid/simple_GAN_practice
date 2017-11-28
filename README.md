to run the model:

    python model.py

require: see requirements.txt

folder structure:

* GAN

  * log--------Where tensorboard log is saved

  * mnist------Where mnist dataset is placed

  * model------Where model and checkpoint file is saved

  * output-----Where generated images from each epoch is saved

  * model.py---defines GAN model

  * ops.py-----defines ops used by GAN

  * utils.py---miscellaneous helper functions


generator:

fc1024 bn lrelu -> fc128\*7\*7 bn lrelu -> conv_transpose 14\*14\*64 bn lrelu -> conv_transpose 28\*28\*1 tanh

discriminator:

convf64 bn lrelu -> convf128 bn lrelu -> fc1024 bn lrelu -> fc1 sigmoid
