to run the model:
python model.py

generator:
fc1024 bn lrelu -> fc128*7*7 bn lrelu -> conv_transpose 14*14*64 bn lrelu -> conv_transpose 28*28*1 tanh

discriminator:
convf64 bn lrelu -> convf128 bn lrelu -> fc1024 bn lrelu -> fc1 sigmoid
