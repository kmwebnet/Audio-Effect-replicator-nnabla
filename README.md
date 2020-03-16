# Audio-effect-replicator-nnabla
replicate audio effects by LSTM. based on [coz-a/Audio-Effect-Replicator](https://github.com/coz-a/Audio-Effect-Replicator).

# what is it  
Train audio effects by setting dry sound as X, effected sound as Y and predict audio effect for new given sound file.  

# Contents  
train.py  -- training main program
config.yml -- define parameters and the list of sound file.  
predict.py -- predict by using trained model and out .nnp file.
fx_replicator.py -- helper program for train.py and predict.py

# Environment
you need to setup [nnabla-tensorboard](https://github.com/naibo-code/nnabla_tensorboard).  
sudo pip3 install -r requirements.txt

# Usage  
python3 train.py

python3 predict.py  -i ./data/testdata1-2.wav

you need to prepare WAV file as monoural 32bit signed int(uncompessed)format.
