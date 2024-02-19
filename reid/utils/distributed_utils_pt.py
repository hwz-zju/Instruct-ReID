import os

import torch
import torch.distributed as dist
import subprocess
import socket
import time
from . import comm_
import torch.cuda.comm


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]

def dist_init():
    
    hostname = socket.gethostname()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if int(os.environ["RANK"]) == 0:
            print('this task is not running on cluster!')
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        addr = socket.gethostname()

    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id == 0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url_" + jobid + ".txt"
        if proc_id == 0:
            tcp_port = str(find_free_port())
            print('write port {} to file: {} '.format(tcp_port, hostfile))
            with open(hostfile, "w") as f:
                f.write(tcp_port)
        else:
            print('read port from file: {}'.format(hostfile))
            while not os.path.exists(hostfile):
                time.sleep(1)
            time.sleep(2)
            with open(hostfile, "r") as f:
                tcp_port = f.read()

        os.environ['MASTER_PORT'] = str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        dist_url = 'env://'
        world_size = ntasks
        rank = proc_id
        gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        distributed = False
        return
    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    print('rank: {} addr: {}  port: {}'.format(rank, addr, os.environ['MASTER_PORT']))
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    if 'SLURM_PROCID' in os.environ and rank == 0:
        if os.path.isfile(hostfile):
            os.remove(hostfile)
    if world_size >= 1:
        # Setup the local process group (which contains ranks within the same machine)
        assert comm_._LOCAL_PROCESS_GROUP is None
        num_gpus = torch.cuda.device_count()
        num_machines = world_size // num_gpus
        for i in range(num_machines):
            ranks_on_i = list(range(i * num_gpus, (i + 1) * num_gpus))
            print('new_group: {}'.format(ranks_on_i))
            pg = torch.distributed.new_group(ranks_on_i)
            if rank in ranks_on_i:
                # if i == os.environ['SLURM_NODEID']:
                comm_._LOCAL_PROCESS_GROUP = pg
    return rank, world_size

def dist_init_singletask(args):
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
