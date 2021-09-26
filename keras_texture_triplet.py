# source: https://github.com/Ao-Lee/Shares/blob/master/triplet%20inputs.py
# - changed to texture classification
# - changed to a small, simple network (not ResNet)
# - added generation of proper a, p, n triplets
# - increased alpha to try to prevent the embedding from winning trivially
# - removed the L2 normalisation (pushing all digits to surface of hypersphere)
# - added a 2D layout visualisation of our embedding
# - changed from a Lambda layer to a custom layer with self.add_loss()

from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Input, Layer
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

############## DATA ###########################

X_train = np.load("../data/Brodatz_Normalised/BrNoRo_X_encoder.npy")
y_train = np.load("../data/Brodatz_Normalised/BrNoRo_y_encoder.npy")
X_train = X_train.astype('float32')
X_train /= 255

img_size = (64, 64, 1)

# our code expects the channels to be a dimension, even if = 1
X_train = X_train.reshape(-1, *img_size)

    
def get_triplet(X, y):
    """Choose a triplet (anchor, positive, negative) of images
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    classes = sorted(list(set(y)))
    # choose two labels
    a, n = np.random.choice(classes, 2, replace=False)
    a_idxs = np.nonzero(y == a)[0] # indices where y == a
    n_idxs = np.nonzero(y == n)[0] # indices where y == n
    # choose two indices for (a, p) and one for n
    (a, p) = np.random.choice(a_idxs, 2, replace=False)
    n = np.random.choice(n_idxs, 1)
    return X[a], X[p], X[n] # get the images

def generate_triplets(X, y, batch_size):
    """Generate an un-ending stream of triplets for training or test."""
    while True:
        A = np.zeros((batch_size, *img_size), dtype="float32")
        P = np.zeros((batch_size, *img_size), dtype="float32")
        N = np.zeros((batch_size, *img_size), dtype="float32")

        for i in range(batch_size):
            a, p, n = get_triplet(X, y)
            A[i] = a
            P[i] = p
            N[i] = n
        yield [A, P, N]


        
############## Loss ###########################

# A layer that creates a triplet loss
class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2):
        super(TripletLossLayer, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        # a batch of (anchor, positive, negative), *after* encoding
        a, p, n = inputs

        # Euclidean distance in the embedding space. axis=-1 means sum
        # over the dimensions of the embedding, but don't sum over the
        # items of the batch. keepdims=True means the result eg d_ap
        # is of shape (batch_size, 1), not just (batch_size,).
        d_ap = K.sqrt(K.sum(K.square(a - p), axis=-1, keepdims=True))
        d_an = K.sqrt(K.sum(K.square(a - n), axis=-1, keepdims=True))

        # exactly as in the formula
        loss = K.maximum(0.0, d_ap - d_an + self.alpha)
        # loss is a tensor of shape (batch_size, 1), one loss per
        # triplet in the batch. This is the "expected" shape for Keras
        # losses, even though just a scalar, or just (batch_size,)
        # would also work.

        # this is the crucial line, allowing our calculation to be
        # used in the model
        self.add_loss(loss)
        # we won't use the return value, but let's return *something*
        return a

############## Model ###########################

def embedding_model():
    """A small convolutional model. Its input is an image and output is an
    embedding, ie a vector. We don't compile or add a loss since this
    model will become a component in the complete model below."""
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',
                            input_shape=img_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # tanh => output is in [-1, 1]^embedding_dim
    model.add(Dense(embedding_dim, activation='tanh'))
    return model


def complete_model(encoder):
    """This part of the model is quite tricky. Rather than a Sequential
    model, we declare a Model and say which are its inputs and
    outputs, and we declare how the outputs are calculated from the
    inputs. In particular, there are no layers in this model, *other
    than* the layers in the embedding model discussed above.

    A further complication is that our triplet loss can't be
    calculated as a function of y_true and y_predicted as
    usual. Instead we calculate the triplet loss in a custom Layer,
    which runs add_loss().

    """
    input_a = Input(img_size)
    input_p = Input(img_size)
    input_n = Input(img_size)
    
    # call the encoder three times to get embeddings
    a = encoder(input_a)
    p = encoder(input_p)
    n = encoder(input_n)

    # the return value from our TripletLossLayer is irrelevant
    # (NB return value is not the loss)
    dummy = TripletLossLayer(alpha=alpha)([a, p, n]) 
    model = Model(inputs=[input_a, input_p, input_n], outputs=dummy)
    # compile with no loss, because TripletLossLayer has added the loss
    model.compile(optimizer=Adam(LR)) 
    return model


def visualise():
    # create a visualisation

    # create a canvas to draw on
    # we add an extra 64 pixels to allow for images whose bottom-left is
    # at the top or right border
    canvas_size = 1000
    imside = img_size[0]
    canvas = np.zeros((canvas_size+imside, canvas_size+imside), dtype=float)

    def loc2pix(x, size):

        """All values in x are in [-1, 1], we want it in [0, size]."""
        # add 1 to make it non-negative, squeeze to remove trivial
        # dimension and transform.
        x = (((1.0 + x.squeeze()) / 2) * size).astype(int)
        assert np.all(x >= 0) and np.all(x < size)
        return x

    X = X_train[:30]
    # get the embedding for x
    Xloc = encoder.predict(X.reshape((-1, *img_size)))
    for xloc, Xi in zip(Xloc, X):
        x = loc2pix(xloc, canvas_size)
        # paint the image of each digit onto the canvas
        canvas[x[0]:x[0]+imside, x[1]:x[1]+imside] = Xi.reshape(img_size[:-1])

    f = "keras_texture_triplet_layout.png"

    dpi = 80.0
    xpixels, ypixels = canvas_size+imside, canvas_size+imside

    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    fig.figimage(canvas, cmap="gray")

    # plt.imshow(canvas, cmap="gray", interpolation=None,
    #            aspect="equal")
    plt.axis("off")
    plt.savefig(f)
    plt.close()

    

############## Settings ###########################
batch_size = 32
# 2 is interesting for visualisation, but higher allows more "space"
# to achieve accuracy in complex domains, eg 128 is common for
# faces. but our data is simpler, so maybe 2 is enough.
embedding_dim = 2 

LR = 0.001 # be careful: too large will be unstable for our data
EPOCHS = 100
alpha = 0.5 # interesting to think about different values

############## Main ###############################

# create the data generators
train_generator = generate_triplets(X_train, y_train, batch_size)

# instantiate the model and take a look
encoder = embedding_model()
encoder.summary()
model = complete_model(encoder)
model.summary()

# fit
model.fit(train_generator, 
          epochs=EPOCHS,
          steps_per_epoch=10,
          validation_steps=1,
          verbose=1
         )

encoder.save("texture_encoder.saved_model")

# encoder = load_model("texture_encoder.saved_model")
visualise()
