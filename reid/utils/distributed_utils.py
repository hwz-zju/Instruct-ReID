import os

import torch
import torch.distributed as dist
# import linklink as link
# try:
#     import spring.linklink as link
# except:
#     import linklink as link

def dist_linklink_init():
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()

    return rank, world_size

def dist_init(args):
    try:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)

        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        addr = node_list[8:].replace('-', '.')
        print(addr)

        os.environ['MASTER_PORT'] = args.port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
    except BaseException:
        print("For debug....")
        num_gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.port)
        torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend='nccl')
    print("Rank {}, World_size {}".format(dist.get_rank(), dist.get_world_size()))
    return num_gpus > 1
