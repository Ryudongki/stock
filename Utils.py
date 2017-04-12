import tensorflow as tf
import numpy as np

#read comma-sperated-value open,high,low,close,amount list
def read_file(filename):
    cnt = 0
    list_open, list_high, list_low, list_close, list_volume = [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f.read().split('\n')[:-1]:
            d, o, h, l, c, v = (int(i) for i in line.split(',')) #date, open, high, low, close, volume
            list_open.append(o)
            list_high.append(h)
            list_low.append(h)
            list_close.append(h)
            list_volume.append(h)
            cnt += 1

    return (cnt, list_open, list_high, list_low, list_close, list_volume)

def get_max_min(list):
    return (np.max(list), np.min(list))

def get_mean_std(list):
    return (np.mean(list), np.std(list))

def to_output_form(list_label, list_pred):
    text = "label pred sign\n"
    sign_cnt = 0
    for i in range(1, len(list_label)):
        sign = (list_label[i]-list_label[i-1]) * (list_pred[i]-list_label[i-1]) >= 0
        text += "{},{},{}\n".format(list_label[i], list_pred[i], sign)
        sign_cnt += sign

    text += "{}\n".format(1.0*sign_cnt/(len(list_label)-1))

    #noop backdoor
    with open("accuracy.txt", 'a') as f:
        f.write("{}\n".format(1.0*sign_cnt/(len(list_label)-1)))

    return text