import numpy as np
from easydict import EasyDict as edict
# import linklink as link
# try:
#     import spring.linklink as link
# except:
#     import linklink as link

__all__ = ['get_taskinfo']

def specific_task_split(task_spec, world_size, rank, tasks):
    ## sanity check
    assert type(task_spec) is list
    assert all(map(lambda x: type(x) is int, task_spec))

    num_tasks = len(task_spec)
    splits = np.sum(task_spec)

    assert world_size % splits == 0
    unit = int(world_size / splits)
    rank = link.get_rank()
    if rank==0:
        print("processing unit num : {0}".format(unit))
    ## split
    Ltask_sizes = [x*unit for x in task_spec]
    Ltasks = []
    Lroots = []
    last = 0
    thistask_info = edict()
    alltask_info = edict()

    for i,gs in enumerate(Ltask_sizes):
        ranks = list(map(int, np.arange(last, last+gs)))
        Ltasks.append(link.new_group(ranks=ranks))    ## The handle for each task
        Lroots.append(last)

        if rank in ranks:
            thistask_info.task_handle = Ltasks[-1]
            thistask_info.task_size = gs
            thistask_info.task_id = i
            thistask_info.task_rank = rank - last
            thistask_info.task_root_rank = last
        last += gs

    alltask_info.root_handles = link.new_group(ranks=Lroots)
    alltask_info.task_sizes = Ltask_sizes
    alltask_info.task_root_ranks = Lroots
    alltask_info.task_num = num_tasks

    return thistask_info, alltask_info


def get_taskinfo(args, world_size, rank):
    # config = args.config
    # tasks = config['tasks']
    tasks = args
    num_tasks = len(tasks)
    task_spec = [tasks[i].get('gres_ratio',1) for i in range(num_tasks)]
    thistask_info, alltask_info = specific_task_split(task_spec, world_size, rank, tasks)

    loss_weight_sum = float(np.sum(np.array([task['loss_weight'] for task in tasks.values()])))

    thistask_info.task_name = tasks[thistask_info.task_id]['task_name']
    thistask_info.task_weight = float(tasks[thistask_info.task_id]['loss_weight']) / loss_weight_sum
    thistask_info.train_file_path = tasks[thistask_info.task_id].get('train_file_path','')
    thistask_info.root_path = tasks[thistask_info.task_id].get('root_path', '')
    thistask_info.task_spec = tasks[thistask_info.task_id].get('task_spec', '')
    alltask_info.task_names = [tasks[i]['task_name'] for i in range(alltask_info.task_num)]
    return thistask_info, alltask_info
