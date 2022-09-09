from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.backends.cudnn as cudnn
from torch_geometric import data
from torch_geometric.loader import DataLoader, DataListLoader

import numpy as np

from config_dist import get_parse_args
from models.model import create_model, load_model, save_model
import utils.misc as misc
from utils.logger import Logger
from utils.random_seed import set_seed
from utils.circuit_utils import check_difference
from trains.train_factory import train_factory
from datasets.dataset_factory import dataset_factory


def main(args):
    misc.init_distributed_mode(args)
    print('==> Using settings {}'.format(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.random_seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    print('==> Loading dataset from: ', args.data_dir)
    dataset = dataset_factory[args.dataset](args.data_dir, args)
    data_len = len(dataset)
    print("Size: ", data_len)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank,shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0:
        log_writer = Logger(args)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


    print('==> Creating model...')
    model = create_model(args)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best = 1e10
    if args.load_model != '':
        model, optimizer, start_epoch, best = load_model(
        model, args.load_model, optimizer, args.resume, args.lr, args.lr_step, best)

    Trainer = train_factory[args.arch]
    trainer = Trainer(args, model, optimizer)
    trainer.set_device(device, args.gpus)


    print('==> Starting training...')
    # best = 1e10
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        mark = epoch if args.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, data_loader_train)
        if misc.is_main_process() and log_writer is not None:
            log_writer.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                log_writer.scalar_summary('train_{}'.format(k), v, epoch)
                log_writer.write('{} {:8f} | '.format(k, v))
        if args.save_intervals > 0 and epoch % args.save_intervals == 0:
            save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(mark)), 
                    epoch, model, optimizer)
    
        save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)), 
                    epoch, model, optimizer)
        if epoch in args.lr_step:
            # save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)), 
            #         epoch, model, optimizer)
            lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    if misc.is_main_process() and log_writer is not None:            
        log_writer.close()


if __name__ == '__main__':
    args = get_parse_args()

    main(args)
