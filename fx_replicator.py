import os
import datetime
import wave
import yaml
import numpy as np
from numpy.lib.stride_tricks import as_strided
import nnabla as nn
#import nnabla_ext.cudnn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

def load_wave(wave_file):
    with wave.open(wave_file, "r") as w:
        buf = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int32)
    return (buf / 0x7fffffff).astype(np.float32)

def save_wave(buf, wave_file):
    _buf = (buf * 0x7fffffff).astype(np.int32)
    with wave.open(wave_file, "w") as w:
        w.setparams((1, 4, 48000, len(_buf), "NONE", "not compressed"))
        w.writeframes(_buf)

def flow(dataset, timesteps, batch_size):
    n_data = len(dataset)
    while True:
        i = np.random.randint(n_data)
        x, y = dataset[i]
        yield random_clop(x, y, timesteps, batch_size)

def random_clop(x, y, timesteps, batch_size):
    max_offset = len(x) - timesteps
    offsets = np.random.randint(max_offset, size=batch_size)
    batch_x = np.stack((x[offset:offset+timesteps] for offset in offsets))
    batch_y = np.stack((y[offset:offset+timesteps] for offset in offsets))
    return batch_x, batch_y

def LSTMCell(x , h2, h1):

    units = h1.shape[1]

    #first stack  h2=hidden, h1= cell
    h2 = F.concatenate(h2, x, axis=1)

    h3 = PF.affine(h2, ( units), name='Affine')

    h4 = PF.affine(h2, ( units), name='InputGate')

    h5 = PF.affine(h2, ( units), name='ForgetGate')

    h6 = PF.affine(h2, ( units), name='OutputGate')

    h3 = F.tanh(h3)

    h4 = F.sigmoid(h4)

    h5 = F.sigmoid(h5)

    h6 = F.sigmoid(h6)

    h4 = F.mul2(h4, h3)

    h5 = F.mul2(h5, h1)

    h4 = F.add2(h4, h5, True)

    h7 = F.tanh(h4)

    h6 = F.mul2(h6, h7)

    return h6 , h4 # hidden, cell


def LSTM(inputs, units, initial_state=None, return_sequences=False, return_state=False, name='lstm'):
    
    batch_size = inputs.shape[0]

    if initial_state is None:

        c0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)), need_grad=True)
        h0 = nn.Variable.from_numpy_array(np.zeros((batch_size, units)), need_grad=True)
    else:
        assert type(initial_state) is tuple or type(initial_state) is list, \
               'initial_state must be a typle or a list.'
        assert len(initial_state) == 2, \
               'initial_state must have only two states.'

        c0, h0 = initial_state

        assert c0.shape == h0.shape, 'shapes of initial_state must be same.'
        assert c0.shape[0] == batch_size, \
               'batch size of initial_state ({0}) is different from that of inputs ({1}).'.format(c0.shape[0], batch_size)
        assert c0.shape[1] == units, \
               'units size of initial_state ({0}) is different from that of units of args ({1}).'.format(c0.shape[1], units)

    cell = c0
    hidden = h0

    hs = []

    for x in F.split(inputs, axis=1):
        with nn.parameter_scope(name):
            cell, hidden = LSTMCell(x, cell, hidden)
        hs.append(hidden)

    if return_sequences:
        ret = F.stack(*hs, axis=1)
    else:
        ret = hs[-1]

    if return_state:
        return ret, cell, hidden
    else:
        return ret


def build_model(x):
    t = LSTM(x, 16,  return_sequences=True, name='LSTM1')
    t1 = LSTM(t, 1,  return_sequences=True, name='LSTM_OUT')
    return t1


def train(model, train_dataflow, val_dataflow, max_epochs, patience):
    timestamp = datetime.datetime.now()

    cp_dir = "checkpoint\{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    cp_filepath = os.path.join(cp_dir, "model_{epoch:06d}.h5")
    cb_mc = ModelCheckpoint(filepath=cp_filepath, monitor="val_loss", period=1, save_best_only=True)

    cb_es = EarlyStopping(monitor="val_loss", patience=patience)

    tb_log_dir = "tensorboard\{:%Y%m%d_%H%M%S}".format(timestamp)
    cb_tb = TensorBoard(log_dir=tb_log_dir)

    model.fit_generator(
        generator=train_dataflow,
        steps_per_epoch=100,
        validation_data=val_dataflow,
        validation_steps=10,
        epochs=max_epochs,
        callbacks=[cb_mc, cb_es, cb_tb])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
            print("bad_epoch:", self.num_bad_epochs)
            print("patience:", self.patience)

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

def sliding_window(x, window, slide):
    n_slide = (len(x) - window) // slide
    remain = (len(x) - window) % slide
    clopped = x[:-remain]
    return as_strided(clopped, shape=(n_slide + 1, window), strides=(slide * 4, 4))
