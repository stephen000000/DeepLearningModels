from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.models import load_model


encoder = load_model("texture_encoder.saved_model")


# idea: take the 11 images of 10 unseen classes

# for eeach class, take the first image for "training"
# and the rest as test

X_train  = np.load("../data/Brodatz_Normalised/BrNoRo_X_fewshot_train.npy")
y_train  = np.load("../data/Brodatz_Normalised/BrNoRo_y_fewshot_train.npy")
X_train = X_train.astype('float32')
X_train = X_train.reshape(-1, 64, 64, 1)
X_train /= 255

X_test  = np.load("../data/Brodatz_Normalised/BrNoRo_X_fewshot_test.npy")
y_test  = np.load("../data/Brodatz_Normalised/BrNoRo_y_fewshot_test.npy")
X_test = X_test.astype('float32')
X_test = X_test.reshape(-1, 64, 64, 1)
X_test /= 255

X_train_embed = encoder.predict(X_train)
X_test_embed = encoder.predict(X_test)
print(X_train.shape)
print(X_train_embed.shape)

knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_embed, y_train)
print("accuracy:", knn.score(X_test_embed, y_test))
