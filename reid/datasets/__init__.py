from .data_builder_attr import DataBuilder_attr
from .data_builder_sc_mnt import DataBuilder_sc
from .data_builder_cc import DataBuilder_cc
from .data_builder_ctcc import DataBuilder_ctcc
from .data_builder_attr import DataBuilder_attr
from .data_builder_t2i import DataBuilder_t2i
from .data_builder_cross import DataBuilder_cross


def dataset_entry(this_task_info):
    return globals()[this_task_info.task_name]