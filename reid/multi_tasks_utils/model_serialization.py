import os
from reid.utils.serialization import save_checkpoint, load_checkpoint
import torch.distributed as dist

class ModelSerialization():
    def __init__(self, taskinfo, args):
        super(ModelSerialization, self).__init__()
        self.taskinfo = taskinfo
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.args = args

    def save(self, model, cur_iter):
        dirs = os.path.join(self.args.logs_dir, 'checkpoint_{}'.format(cur_iter))
        base_fpath = os.path.join(dirs, "base_checkpoint.pth.tar")
        task_specific_fpath = os.path.join(dirs, self.taskinfo.task_name+"_checkpoint.pth.tar")

        #  save base
        if self.rank == 0:
            base_params = dict()
            for key, value in model.base.state_dict().items():
                base_params['base.'+key] = value
            save_checkpoint({
                "state_dict": base_params,
                "start_iter": cur_iter
            }, fpath=base_fpath)
            print("Stage 1: Base Checkpoint has saved...")

        # save task-specific parameters
        if self.taskinfo.task_rank == 0:
            task_specific_params = dict()
            for key, value in model.state_dict().items():
                if key.startswith('base'):
                    continue
                task_specific_params[key] = value
            save_checkpoint(task_specific_params, fpath=task_specific_fpath)
            print("Stage 2: [Task-{}] Task-Specific Checkpoint has saved...".format(self.taskinfo.task_name))

    def load(self, checkpoint_path, load_task_specific=True):
        base_checkpoint_path = os.path.join(checkpoint_path, "base_checkpoint.pth.tar")
        base_state_dict = load_checkpoint(base_checkpoint_path)
        params_state_dict = base_state_dict["state_dict"]
        start_iter = base_state_dict['start_iter']

        if not load_task_specific:
            return params_state_dict, start_iter
        task_specific_checkpoint_path = os.path.join(checkpoint_path, self.taskinfo.task_name+"_checkpoint.pth.tar")
        task_specific_state_dict = load_checkpoint(task_specific_checkpoint_path)
        params_state_dict.update(task_specific_state_dict)
        return params_state_dict, start_iter







