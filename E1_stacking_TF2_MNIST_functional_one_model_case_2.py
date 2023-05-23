'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

'''
D2. Load MNIST data / Only for Toy Project
'''

# print(tf.__version__)
## MNIST Dataset #########################################################
mnist = tf.keras.datasets.mnist
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
##########################################################################

## Fashion MNIST Dataset #################################################
# mnist = tf.keras.datasets.fashion_mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Change data type as float. If it is int type, it might cause error
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

'''
D5. Build dataset
'''
batch_size = 100
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
# if it is RGB type, type shall be [image_size, image_size, 3]
# For MNIST or Fashion MNIST, it need to reshape

import numpy as np
X_train_new_axis = np.expand_dims(X_train, axis=-1)
X_test_new_axis = np.expand_dims(X_test, axis=-1)

# It fills data as much as the input buffer_size and randomly samples and replaces it with new data.
# Perfect shuffling requires a buffer size greater than or equal to the total size of the data set.
# If you use a buffer_size smaller than the small number of data, 
# random shuffling occurs within the data as much as the initially set buffer_size.

shuffle_size = 100000

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_new_axis, Y_train)).shuffle(shuffle_size).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test_new_axis, Y_test)).batch(batch_size)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras import Input
import numpy as np

'''
M2. Set Hyperparameters
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 10
learning_rate = 0.001

'''
M3. Build NN model
'''
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
def create_model():
    inputs = Input(shape=(28, 28, 1))
    conv1      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1      = MaxPool2D(padding='SAME')(conv1)
    conv2      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(pool1)
    pool2      = MaxPool2D(padding='SAME')(conv2)
    pool3_flat = Flatten()(pool2)
    dense4     = Dense(units=128, activation=tf.nn.relu)(pool3_flat)
    drop4      = Dropout(rate=0.4)(dense4)
    
    conv1_a      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1_a      = MaxPool2D(padding='SAME')(conv1_a)
    conv2_a      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(pool1_a)
    pool2_a      = MaxPool2D(padding='SAME')(conv2_a)
    pool3_flat_a = Flatten()(pool2_a)
    dense4_a     = Dense(units=128, activation=tf.nn.relu)(pool3_flat_a)
    drop4_a      = Dropout(rate=0.4)(dense4_a)
    
    conv1_b      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1_b      = MaxPool2D(padding='SAME')(conv1_b)
    conv2_b      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(pool1_b)
    pool2_b      = MaxPool2D(padding='SAME')(conv2_b)
    pool3_flat_b = Flatten()(pool2_b)
    dense4_b     = Dense(units=128, activation=tf.nn.relu)(pool3_flat_b)
    drop4_b      = Dropout(rate=0.4)(dense4_b)
    
	# There are concatenate layer for n-models 
    merged = concatenate([drop4, drop4_a, drop4_b],axis=1)
    
    dense5 = Dense(units=128, activation=tf.nn.relu)(merged)
    drop5 = Dropout(rate=0.4)(dense5)
    
    logits = Dense(10, activation='softmax')(drop5)
    
    model = Model(inputs=inputs, outputs=logits)

    return model

model = create_model()

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_shapes.png', show_shapes=True)

'''
M4. Optimizer
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

'''
M5. Define Loss Function
'''
# Forward Pass included
@tf.function
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        y_pred=logits, y_true=labels, from_logits=True))    
    return loss   

'''
M6. Define train loop
'''

@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

@tf.function
def train_step(model, images, labels):
    gradients = grad(model, images, labels)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

'''
M7. Metrics - Accuracy
'''

@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# checkpoint was not used in this implementation
checkpoint = tf.train.Checkpoint(cnn=model)

'''
M8. Define Episode / each step process
'''

from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for images, labels in train_ds:
            
            train_step(model, images, labels)
            
            loss_val = loss_fn(model, images, labels)
            acc = evaluate(model, images, labels) * 100
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
'''
M9. Model evaluation
'''
with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
    test_losses = []
    test_accuracies = []
    for test_images, test_labels in test_ds:
        loss_val = loss_fn(model, test_images, test_labels)
        
        acc = evaluate(model, test_images, test_labels) * 100

        test_losses.append(loss_val)
        test_accuracies.append(acc)

        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
