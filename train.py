import os
from argparse import ArgumentParser
import datetime
import yaml
import numpy as np
from fx_replicator import (
    load_wave, flow, build_model , train, AverageMeter, EarlyStopping
)
import nnabla as nn
#import nnabla_ext.cudnn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save
import tqdm
from nnabla_tensorboard import SummaryWriter

def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]

    train_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1))
        for _ in config["train_data"]]
    train_dataflow = flow(train_dataset, input_timesteps, batch_size)

    val_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1))
        for _ in config["val_data"]]
    val_dataflow = flow(val_dataset, input_timesteps, batch_size)

    """
    from nnabla.ext_utils import get_extension_context
    cuda_device_id = 0
    ctx = get_extension_context('cudnn', device_id=cuda_device_id)
    print("Context: {}".format(ctx))
    nn.set_default_context(ctx)  # Set CUDA as a default context.
    """


    timestamp = datetime.datetime.now()

    # For tensorboard
    tb_log_dir = "tensorboard/{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    tb_writer = SummaryWriter(tb_log_dir)


    x = nn.Variable((batch_size, input_timesteps, 1), need_grad=True)
    y = nn.Variable((batch_size, input_timesteps, 1), need_grad=True)

    vx = nn.Variable((batch_size, input_timesteps, 1), need_grad=True)
    vy = nn.Variable((batch_size, input_timesteps, 1), need_grad=True)

    t = build_model(x)
    vt = build_model(vx)

    loss = F.mean(F.squared_error(t[:, -output_timesteps:, :], y[:, -output_timesteps:, :]))
    solver = S.AdaBound()
    solver.set_parameters(nn.get_parameters(grad_only=False))

    tloop = tqdm.trange(1, max_epochs + 1)
    es = EarlyStopping(patience=patience)

    steps = 100
    validation_time = 10


    cp_dir = "checkpoint/{:%Y%m%d_%H%M%S}".format(timestamp)
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)

    idx = 0

    for i in tloop:

        # TRAINING

        tloop.set_description("Training Epoch")
        st = tqdm.trange( 0, steps )
        losses = AverageMeter()
        for j in st:

            x.d ,y.d  = train_dataflow.__next__()
            st.set_description("Training Steps")
            loss.forward()
            solver.zero_grad()
            loss.backward()
            solver.update()
            losses.update(loss.d.copy().mean())
            st.set_postfix(
                train_loss=losses.avg
            )

            # write train graph
            tb_writer.add_scalar('train/loss', losses.avg, global_step=idx)
            idx += 1

        # VALIDATION
        vlosses = AverageMeter()
        for j in range(validation_time):
            vx.d ,vy.d  = val_dataflow.__next__()

            vloss = F.mean(F.squared_error(vt[:, -output_timesteps:, :], vy[:, -output_timesteps:, :]))
            vloss.forward(clear_buffer=True)
            vlosses.update(vloss.d.copy().mean())

        tloop.set_postfix(
            train_loss=losses.avg, val_loss=vlosses.avg
        )

        tb_writer.add_scalar('test/loss', vlosses.avg, global_step=i)

        stop = es.step(vlosses.avg)
        is_best = vlosses.avg == es.best

        # save current model
        nn.save_parameters(os.path.join(
            cp_dir ,'checkpoint_{}.h5'.format(i) ))

        if is_best:
            nn.save_parameters(os.path.join(
                cp_dir ,'best_result.h5'))
            """
            contents = {
                'networks': [
                    {'name': 'MyChain',
                    'batch_size': batch_size,
                    'outputs': {'t': t },
                    'names': {'x': x}}],
                'executors': [
                    {'name': 'runtime',
                    'network': 'MyChain',
                    'data': ['x'],
                    'output': ['t']}]}
            nnabla.utils.save.save(os.path.join(cp_dir ,'MyChain.nnp'), contents)
            """
        if stop:
            print("Apply Early Stopping")
            break




def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
