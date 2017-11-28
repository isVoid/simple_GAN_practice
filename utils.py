import numpy as np

def random_mini_batches(data, mini_batch_size=200, seed = 2):

    """Helper function to extract mini batches from the whole training set.

        Input:
            X_train: Noisy Patches training set
            mini_batch_size: Number of patches contained in a mini batch
            seed: We use different seeds each epoch to maintain randomness.

        Output:
            minibatches: python list of tuple (X_batch, y_batch)

    """

    np.random.seed(seed)
    """Number of batches and extra batch that does not fit into a whole batch"""
    num_batches = data.shape[0] // mini_batch_size
    res = data.shape[0] % mini_batch_size

    """Shuffle entire dataset"""
    sidx = np.arange(data.shape[0])
    np.random.shuffle(sidx)
    X = data[sidx]

    """Create an index array that refer to the multiples of mini_batch_size"""
    idx = np.arange(data.shape[0] - res)
    idx = idx.reshape(-1, mini_batch_size)

    """Create a view that refer X_train up to multiples of mini_batch_size"""
    X_train_mul = X[0:num_batches * mini_batch_size, :]

    """Extract the multiple parts into mini_batches"""
    X_train_minis = data[idx, :]

    minibatches = []
    for i in range(X_train_minis.shape[0]):
        minibatches.append(X_train_minis[i])

    return minibatches

def gen_to_img(gen):
    print ("gen shape", gen.shape)
    print ("gen type", gen.dtype)
    intgen = (gen*255).astype("uint8").reshape([-1, 28, 28])
    x = y = 0
    w = np.ceil(np.sqrt(intgen.shape[0])).astype(int)
    img = np.zeros((w * 28, w * 28))
    for i in range(gen.shape[0]):
        img[x:x+28, y:y+28] = intgen[i]
        if (x + 28 < img.shape[0]):
            x+=28
        else:
            y+=28
            x=0
    img = img.astype("uint8")
    return img

def norm_transform(img):
    return img * 2 - 1

def inv_norm_transform(img):
    return (img + 1) / 2.
