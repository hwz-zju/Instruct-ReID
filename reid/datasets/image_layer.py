from __future__ import print_function, absolute_import

from reid.datasets.data.base_dataset import BaseImageDataset


class Image_Layer(BaseImageDataset):
    def __init__(self, image_list, image_list_additional=None, is_train=False, is_query=False, is_gallery=False, verbose=True):
        super(Image_Layer, self).__init__()
        imgs, clothes, pids, cids, cams =[], [], [], [], []
        with open(image_list) as f:
            for line in f.readlines():
                info = line.strip('\n').split(" ")
                imgs.append(info[0])
                clothes.append(info[1])
                pids.append(int(info[2]))
                cids.append(int(info[3]))
                if len(info) >4:
                    cams.append(int(info[4]))
                elif is_train:
                    cams.append(0)
                elif is_query:
                    cams.append(-1)
                else:
                    cams.append(-2)
        if image_list_additional is not None:
            with open(image_list_additional) as f:
                for line in f.readlines():
                    info = line.strip('\n').split(" ")
                    imgs.append(info[0])
                    clothes.append(info[1])
                    pids.append(int(info[2]))
                    cids.append(int(info[3]))
                    if len(info) >4:
                        cams.append(int(info[4]))
                    elif is_train:
                        cams.append(0)
                    elif is_query:
                        cams.append(-1)
                    else:
                        cams.append(-2)

        if is_train:
            pids = self._relabel(pids)

        self.data = list(zip(imgs, clothes, pids, cids, cams))
        self.num_classes, self.num_imgs, self.num_cids, self.num_cams = self.get_imagedata_info(self.data)

        if verbose:
            print("=> {} Dataset information has been loaded.".format(image_list))
            if is_train:
                self.print_dataset_statistics(self.data, 'train')
            if is_gallery:
                self.print_dataset_statistics(self.data, 'gallery')
            if is_query:
                self.print_dataset_statistics(self.data, 'query')



