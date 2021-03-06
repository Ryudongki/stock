''' 
This script shows how to predict stock prices using a basic RNN 
'''
import tensorflow as tf
import numpy as np
import os
import sys

tf.set_random_seed(777)  # reproducibility 

timesteps = seq_length = 7
data_dim = 5
hidden_dim = 20
output_dim = 1
learning_rate = 0.01
iterations = 500
cnt = 0
file_list = []
filenames = []

# aaa = int(sys.argv[1]) 
ccc = int(sys.argv[1])

for (path, dir, files) in os.walk(os.getcwd()):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if (ext == '.csv'):
            file_list.append(path + '\\' + filename)
            filenames.append(str(filename))

input_len = len(file_list)

for i in range(ccc, ccc + 1):
    input_data = []
    close_data = []
    dataInput = []
    dataClose = []
    temp1, temp2, temp3, temp4, temp5 = [], [], [], [], []

    f = open("%s" % (file_list[i]))

    lines = f.readlines()

    for a in lines:
        temp = a.split(',')
        temp = [float(t) for t in temp[:]]
        temp[4], temp[5] = temp[5], temp[4]
        temp1.append(temp[1])
        temp2.append(temp[2])
        temp3.append(temp[3])
        temp4.append(temp[4])
        temp5.append(temp[5])

    f.close()

    meanOpen, stdOpen = np.mean(temp1), np.std(temp1)
    meanHigh, stdHigh = np.mean(temp2), np.std(temp2)
    meanLow, stdLow = np.mean(temp3), np.std(temp3)
    meanVolume, stdVolume = np.mean(temp4), np.std(temp4)
    meanClose, stdClose = np.mean(temp5), np.std(temp5)

    for line in lines:
        x = line.split(',')
        x = [float(t) for t in x[:]]
        x[4], x[5] = x[5], x[4]
        x[1] = (x[1] - meanOpen) / (stdOpen)
        x[2] = (x[2] - meanHigh) / (stdHigh)
        x[3] = (x[3] - meanLow) / (stdLow)
        x[4] = (x[4] - meanVolume) / (stdVolume)
        x[5] = (x[5] - meanClose) / (stdClose)

        input_data.append(x[1:])
        close_data.append(x[5])

    for j in range(0, len(lines) - seq_length):
        _x = input_data[j:j + seq_length]
        _y = close_data[j + seq_length]
        dataInput.append(_x)
        dataClose.append([_y])

    train_size = int(len(dataClose) * 0.7) - 1
    test_size = len(dataClose) - train_size
    trainInput, testInput = np.array(dataInput[1:train_size]), np.array(dataInput[train_size:len(dataInput)])
    trainClose, testClose = np.array(dataClose[1:train_size]), np.array(dataClose[train_size:len(dataClose)])
    trainClosePrev = np.array(dataClose[0:train_size - 1])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    Y_prev = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network 
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
    cost = (Y - Y_pred) * (
        tf.cast((Y_pred - Y_prev) * (Y - Y_prev) < 0, tf.float32) * 2 + tf.cast((Y_pred - Y_prev) * (Y - Y_prev) >= 0,
                                                                                tf.float32))

    # cost/loss 
    loss = tf.reduce_mean(tf.square(cost))  # sum of the squares 
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
            _, step_loss = sess.run([train, loss], feed_dict={X: trainInput, Y: trainClose, Y_prev: trainClosePrev})

            # if k % 100 is 0: 
            # print("[step: {}] loss: {}".format(k, step_loss)) 

        # Test step 
        test_predict = sess.run(Y_pred, feed_dict={X: testInput})
        rmse = sess.run(rmse, feed_dict={
            targets: testClose, predictions: test_predict})
        # print("RMSE: {}".format(rmse)) 
        # print(testClose) 
    print(test_predict[-1])

    f = open("C:\\Users\\Ryu\\PycharmProjects\\savebystock\\5.txt", 'a')
    f.write(str(filenames[i]) + " ")
    f.write(str(testClose[-2] * (stdClose) + meanClose) + " ")
    f.write(str(test_predict[-2] * (stdClose) + meanClose) + " ")
    f.write(str(testClose[-1] * (stdClose) + meanClose) + " ")
    f.write(str(test_predict[-1] * (stdClose) + meanClose) + '\n')
    f.close()

    num1 = testClose[-2] * (stdClose) + meanClose
    num2 = test_predict[-2] * (stdClose) + meanClose
    num3 = testClose[-1] * (stdClose) + meanClose

    if (num1 - num3 < 0 and num1 - num2 > 0):
        cnt += 1
    elif (num1 - num3 > 0 and num1 - num2 < 0):
        cnt += 1

    f.close()

f = open("C:\\Users\\Ryu\\PycharmProjects\\savebystock\\5.txt", 'a')

f.write(str(cnt) + "\n")

f.close()