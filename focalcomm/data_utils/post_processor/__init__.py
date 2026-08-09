# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from focalcomm.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from focalcomm.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from focalcomm.data_utils.post_processor.focalcomm_postprocesser import FocalCommPostprocessor
from focalcomm.data_utils.post_processor.anchor_free_postprocessor import AnchorFreePostprocessor
__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'BevPostprocessor': BevPostprocessor,
    'FocalCommPostprocessor': FocalCommPostprocessor,
    'AnchorFreePostprocessor': AnchorFreePostprocessor,
}


def build_postprocessor(params, class_names, train):
    process_method_name = params['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'BevPostprocessor', 'FocalCommPostprocessor', 'AnchorFreePostprocessor']
    # print(process_method_name)
    anchor_generator = __all__[process_method_name](
        # anchor_params=anchor_cfg,
        params,
        class_names,
        train
    )

    return anchor_generator
