'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import os
import sys
from Utils import read_file, to_output_form
import Models
tf.set_random_seed(777)  # reproducibility

seq_length = 7
data_dim = 5
hidden_dim = 25
output_dim = 1
learning_rate = 0.01
iterations = 500

normalize_data  = Models.normalize_data_meanstd
unormalize_data = Models.unormalize_data_meanstd

def predict_func(filename):
    data_filename = os.path.join("data", filename + '.csv')
    output_filename = os.path.join("out", filename + '.txt')

    length, list_open, list_high, list_low, list_close, list_volume = read_file(data_filename)

    length, input_data, label_data = normalize_data(length, list_open, list_high, list_low, list_close, list_volume)

    data_input, data_close = [], []
    for j in range(length - seq_length):
        _x = input_data[j:j + seq_length]
        _y = label_data[j + seq_length]
        data_input.append(_x)
        data_close.append(_y)

    train_size = int(length * 0.7) - 1

    #start index 1 for use prev_data for costing
    train_input, test_input = np.array(data_input[1:train_size]), np.array(data_input[train_size:])
    train_close, test_close = np.array(data_close[1:train_size]), np.array(data_close[train_size:])
    train_close_prev = np.array(data_close[0:train_size-1])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    Y_prev = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
    cost = (Y - Y_pred) * (
    tf.cast((Y_pred - Y_prev) * (Y - Y_prev) < 0, tf.float32) * 4 + tf.cast((Y_pred - Y_prev) * (Y - Y_prev) >= 0,
                                                                            tf.float32))
    # cost/loss
    loss = tf.reduce_mean(tf.square(cost))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None,1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for k in range(iterations):
            _, step_loss = sess.run([training, loss], feed_dict={X: train_input, Y: train_close, Y_prev: train_close_prev})

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: test_input})
        rmse = sess.run(rmse, feed_dict={targets: test_close, predictions: test_predict})

    test_close = unormalize_data(test_close, list_close)
    test_predict = unormalize_data(test_predict, list_close)

    with open(output_filename, 'w') as f:
        f.write(to_output_form(test_close, test_predict))

def main():
    predict_func(sys.argv[1])

if __name__ == "__main__":
    main()