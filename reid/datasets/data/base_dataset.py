
class BaseDataset(object):
    """
    Base class of reid dataset
    """

    @staticmethod
    def get_imagedata_info(data):
        pids, cids, cams = [], [], []
        for _, _, pid, cid, camid in data:
            pids += [pid]
            cids += [cid]
            cams += [camid]
        pids = set(pids)
        cids = set(cids)
        cams = set(cams)
        num_pids = len(pids)
        num_cids = len(cids)
        num_imgs = len(data)
        num_cams = len(cams)
        return num_pids, num_imgs, num_cids, num_cams


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, dataset, dataset_type):
        num_train_pids, num_train_imgs, num_train_cids, num_train_cams = self.get_imagedata_info(dataset)

        print("Dataset statistics:")
        print("  ------------------------------------------")
        print("  {:<9s}| {:^5s} | {:^8s} | {:^8s} | {:^9s}".format('subset', '# ids', '# images', '# clothes', '# cameras'))
        print("  ------------------------------------------")
        print("  {:<9s}| {:^5d} | {:^8d} | {:^8d} | {:^9d}".format(dataset_type, num_train_pids, num_train_imgs, num_train_cids, num_train_cams))
        print("  ------------------------------------------")

    @staticmethod
    def _relabel(label_list):
        sorted_pids = sorted(list(set(label_list)))
        label_dict = dict()
        for idx, pid in enumerate(sorted_pids):
            if pid in label_dict.keys():
                continue
            label_dict[pid] = idx

        relabeled_list = [label_dict[pid] for pid in label_list]
        return relabeled_list
