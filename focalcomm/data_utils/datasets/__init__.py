# -*- coding: utf-8 -*-


# Dereje Shenkut <derejeshenkut@gmail.com>

# Adapted from OpenCOOD: Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# from focalcomm.data_utils.datasets.late_fusion_dataset import LateFusionDataset
# from focalcomm.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from focalcomm.data_utils.datasets.v2xreal_intermediate_fusion_dataset import V2XRealIntermediateDataset
from focalcomm.data_utils.datasets.dairv2x_intermediate_fusion_dataset import DAIRV2XIntermediateFusionDataset
# from focalcomm.data_utils.datasets.opv2v_intermediate_fusion_dataset import OPV2VIntermediateFusionDataset

__all__ = {
    # 'LateFusionDataset': LateFusionDataset,
    # 'EarlyFusionDataset': EarlyFusionDataset,
    'V2XRealIntermediateDataset': V2XRealIntermediateDataset,
    'DAIRV2XIntermediateFusionDataset': DAIRV2XIntermediateFusionDataset,
    # 'OPV2VIntermediateFusionDataset': OPV2VIntermediateFusionDataset,
}

from focalcomm.utils.constants import GT_RANGE, COM_RANGE


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in focalcomm/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in __all__.keys(), error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
