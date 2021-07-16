import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from account import MomentsAccountant
from noisy_gradient import NoisySGD
from numpy import linalg as LA
import tensorflow.keras.backend as K

def classification_net_features(train_dataset,model,train_x_list):
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = layers.Dense(64, activation="relu")(inputs)
    x2 = layers.Dense(64, activation="relu")(x1)
    outputs = layers.Dense(10, name="predictions")(x2)
    #model = keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer.
    #optimizer = keras.optimizers.SGD()
    optimizer = NoisySGD()
    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    C =1
    # Prepare the training dataset.
    batch_size = 32
    leak_threshold = 10
    #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #x_train = np.reshape(x_train, (-1, 784))
    #x_test = np.reshape(x_test, (-1, 784))
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1).batch(batch_size)

    epochs = 5
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        eps = [1.0] * len(train_x_list)
        delta = [1e-5] * len(train_x_list)
        total_privacy_spending=[0.0]* len(train_x_list)

        accountant = MomentsAccountant()
        #accountant.accumulate_privacy_spending()




        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            log_mo = accountant._compute_log_moment()
            eps_min, order = accountant._compute_eps(log_mo)
            #print('log_moment_eps_min', log_mo[order])
            # initialize eps and delta

            # Log every 50 batches.
            #if step % 50 == 0:
            #    print(
            #        "Training loss (for one batch) at step %d: %.4f"
            #        % (step, float(loss_value))
            #    )
            #    print("Seen so far:%s to the %s samples" % (32 * step, 32 * step + 31))
            for i in range(32):
                if 32 * step + i >= 5039:
                    None
                else:
                    eps[32 * step + i] = eps_min
                    total_privacy_spending[32 * step + i] += log_mo[order]
            if total_privacy_spending[32 * step] > leak_threshold:
                continue


            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            summed_squares = [K.sum(K.square(g)) for g in grads]
            norm = K.sqrt(sum(summed_squares))
            #grads = tf.math.divide(grads,max(1,norm/C))
            #norms = tf.tile(norm, 512)
            #c = grads / tf.reshape(norm, (-1, 1))
            d=tf.reshape(norm, (1))
            print(d)
            print('cliped gradients', norm)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))






if __name__ == '__main__':
	classification_net_features()