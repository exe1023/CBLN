import matplotlib.pyplot as plt
import numpy as np

def smoothing(x, y, param=5):
    x_new, y_new = [], []
    for step in range(0, len(x), param):
        y_new.append(np.mean(y[step:step+param]))
        x_new.append(step)
    return x_new, y_new

if __name__ == '__main__':
    config = ['baseline.json', 'ln.json', 'cbln.json']

    for c in config:
        train_log = c + '.log.train'
        valid_log = c + '.log.valid'

        train_loss = []
        train_idx = []
        with open(train_log, 'r') as f:
            for line in f:
                idx, loss = line.strip().split('\t')
                train_loss.append(float(loss))
                train_idx.append(int(idx))
    
        valid_loss, valid_acc = [], []
        valid_idx = []
        with open(valid_log, 'r') as f:
            for line in f:
                idx, loss, acc = line.strip().split('\t')
                valid_loss.append(float(loss))
                valid_acc.append(float(acc))
                valid_idx.append(int(idx))
    
        train_idx, train_loss = smoothing(train_idx, train_loss, param=50)
        valid_idx, valid_acc = smoothing(valid_idx, valid_acc)
        #plt.plot(train_idx, train_loss)
        plt.plot(valid_idx, valid_acc, label=c)
    plt.show()