import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import savefig
import sys

tf.set_random_seed(777)  # reproducibility

timesteps = seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

file_list = []
filenames = []

bbb = int(sys.argv[1])

for (path, dir, files) in os.walk(os.getcwd()):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if (ext == '.csv'):
            file_list.append(path + '\\' + filename)
            filenames.append(str(filename))

input_len = len(file_list)

for i in range(bbb, bbb + 1):
    input_data = []
    close_data = []
    dataInput = []
    dataClose = []
    temp1, temp2, temp3, temp4, temp5 = [], [], [], [], []

    f = open("%s" % (file_list[i]))

    lines = f.readlines()

    for a in lines:
        temp = a.split(',')
        temp = [float(i) for i in temp[:]]

        temp[4], temp[5] = temp[5], temp[4]
        temp1.append(temp[1])
        temp2.append(temp[2])
        temp3.append(temp[3])
        temp4.append(temp[4])
        temp5.append(temp[5])

    f.close()

    openMean = []
    highMean = []
    lowMean = []
    volumeMean = []
    closeMean = []

    for k in range(len(temp1) - 2):
        _open = (temp1[k] + temp1[k + 1] + temp1[k + 2]) / 3
        _high = (temp2[k] + temp2[k + 1] + temp2[k + 2]) / 3
        _low = (temp3[k] + temp3[k + 1] + temp3[k + 2]) / 3
        _volume = (temp4[k] + temp4[k + 1] + temp4[k + 2]) / 3
        _close = (temp5[k] + temp5[k + 1] + temp5[k + 2]) / 3

        openMean.append(_open)
        highMean.append(_high)
        lowMean.append(_low)
        volumeMean.append(_volume)
        closeMean.append(_close)

    maxOpen, minOpen = np.max(openMean), np.min(openMean)
    maxHigh, minHigh = np.max(highMean), np.min(highMean)
    maxLow, minLow = np.max(lowMean), np.min(lowMean)
    maxVolume, minVolume = np.max(volumeMean), np.min(volumeMean)
    maxClose, minClose = np.max(closeMean), np.min(closeMean)

    for line in lines:
        x = line.split(',')
        x = [float(i) for i in x[:]]
        x[4], x[5] = x[5], x[4]
        x[1] = (x[1] - minOpen) / (maxOpen - minOpen + 1e-7)
        x[2] = (x[2] - minHigh) / (maxHigh - minHigh + 1e-7)
        x[3] = (x[3] - minLow) / (maxLow - minLow + 1e-7)
        x[4] = (x[4] - minVolume) / (maxVolume - minVolume + 1e-7)
        x[5] = (x[5] - minClose) / (maxClose - minClose + 1e-7)

        input_data.append(x[1:])
        close_data.append(x[5])

    for j in range(0, len(lines) - seq_length):
        _x = input_data[j:j + seq_length]
        _y = close_data[j + seq_length]
        dataInput.append(_x)
        dataClose.append([_y])

    train_size = int(len(dataClose) * 0.7)
    test_size = len(dataClose) - train_size
    trainInput, testInput = np.array(dataInput[0:train_size]), np.array(dataInput[train_size:len(dataInput)])
    trainClose, testClose = np.array(dataClose[0:train_size]), np.array(dataClose[train_size:len(dataClose)])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dim, activation_fn=None)

    # cost/loss
    loss = tf.reduce_mean(tf.square(Y_pred - Y))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for k in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainInput, Y: trainClose})

            # if k % 100 is 0:
            # print("[step: {}] loss: {}".format(k, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testInput})
        rmse = sess.run(rmse, feed_dict={
            targets: testClose, predictions: test_predict})
        # print("RMSE: {}".format(rmse))
        # print(testClose)
    print(test_predict[-1])

    f = open("C:\\Users\\Ryu\\PycharmProjects\\savebystock\\1.txt", 'a')
    f.write(str(filenames[i]) + " ")
    f.write(str([temp5[-2]]) + " ")
    f.write(str((3 * (test_predict[-2] * (maxClose - minClose + 1e-7) + minClose)) - temp5[-3] - temp5[-2]) + " ")
    f.write(str([temp5[-1]]) + " ")
    f.write(str((3 * (test_predict[-1] * (maxClose - minClose + 1e-7) + minClose)) - temp5[-2] - temp5[-1]) + '\n')
    f.close()

    # plt.plot(testClose)
    # plt.plot(test_predict)
    # fig = plt.gcf()
    # savefig(r'C:\Users\Ryu\Desktop\kosdaq_sbs\%s.jpg' % filenames[i].split('.')[0])
    # plt.show()