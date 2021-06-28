# coding=utf-8
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from util import *


def train(config, dataloaders, model, optimizers, evaluation, val_plot=None):
    # configure train
    log_name = config['log_name']
    tqdm_bar = '{desc} {n_fmt}/{total_fmt}-{percentage:3.0f}%|{rate_fmt}'
    print(config)
    set_random_seed(config['seed'])

    # data
    train_data_loader, val_data_loader = dataloaders

    # configure model
    cp_config = model.load_checkpoint(config['cp_path'])
    model.to_device(config['device'])

    # optimizer and scheduler
    optimizer, scheduler = optimizers

    # checkpoint
    if config['save_interval'] != 0:
        cp_dir_path = os.path.normcase(os.path.join('checkpoints', log_name))
        os.mkdir(cp_dir_path)
        with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
            if config['cp_path']:
                info = str(config) + '#' * 30 + '\npre_config:\n' + str(
                    cp_config) + '#' * 30 + '\n'
            else:
                info = str(config) + '#' * 30 + '\n'
            f.write(info)

    # visble
    if config['visible']:
        log_writer = SummaryWriter(os.path.join("log", log_name))
        log_writer.add_text('cur_config', str(config))
        if config['cp_path']:
            log_writer.add_text('pre_config', cp_config.__str__())

    # Start!
    print("Start training!\n")
    cur_epoch = 0
    error_num = 0
    while cur_epoch < config['max_epochs']:
        cur_epoch += 1
        # runtime checkpoints
        model_checkpoint = model.state_dict()
        optimizer_checkpoint = optimizer.state_dict()
        # train
        model.train()
        epoch_loss = 0
        try:
            for train_data in tqdm(train_data_loader,
                                   desc='[{}] Epoch'.format(cur_epoch),
                                   bar_format=tqdm_bar):

                loss = 0
                optimizer.zero_grad()
                loss = model.train_epoch(train_data)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                # Don't waste time
                assert not torch.isnan(loss)
        except AssertionError as err:
            error_num += 1
            model.load_state_dict(model_checkpoint)
            optimizer.load_state_dict(optimizer_checkpoint)
            err_info = '[{}] {} {}, back to last epoch'.format(
                cur_epoch, get_timestamp(), err)
            print(err_info)
            if config['save_interval'] != 0:
                with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
                    f.write(err_info)
            cur_epoch -= 1
            assert error_num < config['log_interval']
            continue
        error_num = 0

        train_loss = epoch_loss / len(train_data_loader)
        scheduler.step(train_loss)

        print("[{}] {} Training - loss: {:.4e}".format(cur_epoch,
                                                       get_timestamp(),
                                                       train_loss))
        if config['save_interval'] != 0:
            with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
                f.write("[{}] {} Training - loss: {:.4e}\n".format(
                    cur_epoch, get_timestamp(), train_loss))
        if config['visible']:
            cur_lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Train/Loss', train_loss, cur_epoch)
            log_writer.add_scalar('Train/lr', cur_lr, cur_epoch)

        # val
        if cur_epoch % config['log_interval'] == 0 or config['debug']:
            # torch.cuda.empty_cache()
            model.eval()
            model.init_val()
            with torch.no_grad():
                for val_data in tqdm(val_data_loader,
                                     desc='[{}] Val'.format(cur_epoch),
                                     bar_format=tqdm_bar):
                    val_save = model.val_epoch(val_data)

            val_result = evaluation(val_save, len(val_data_loader))

            val_info = "[{}] {} Val -".format(cur_epoch, get_timestamp())
            for k, v in val_result.items():
                if 'loss' in k.lower():
                    val_info += " {}: {:.4e},".format(k, v)
                else:
                    val_info += " {}: {:.4f},".format(k, v)
            print(val_info)

            if config['save_interval'] != 0:
                with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
                    val_info += '\n'
                    f.write(val_info)
            if config['visible']:
                for k, v in val_result.items():
                    log_writer.add_scalar('Val/{}'.format(k), v, cur_epoch)
                if val_plot is not None:
                    val_plot(log_writer, cur_epoch, val_save)

        # checkpoint save
        if config['save_interval'] != 0 and cur_epoch % config[
                'save_interval'] == 0:
            save_path = os.path.join(
                cp_dir_path, '{}_{:.4e}.pth'.format(cur_epoch, train_loss))
            model.save_checkpoint(save_path, info=str(config))
        elif config['save_interval'] == -1:
            save_path = os.path.join(cp_dir_path, '{}.pth'.format(log_name))
            model.save_checkpoint(save_path, info=str(config))
