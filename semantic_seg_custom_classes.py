# import packages
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.splits import random_bbox_assignment
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
import torch

# sub-class raster datasets to create the two datasets we need
# NAIP for images and Chesapeake for labels


class NAIPImages(RasterDataset):
    filename_glob = "m_*.tif"
    is_image = True
    separate_files = False


class ChesapeakeLabels(RasterDataset):
    filename_glob = "m_*.tif"
    is_image = False
    separate_files = False

# create a custom geodatamodule


# class CustomGeoDataModule(GeoDataModule):
#     def __init__(self, dataset_class, patch_size, batch_size, length, dataset, ** kwargs):
#         """
#         Args:
#             dataset_class: The class or function used to load the dataset.
#             patch_size: Size of the patches for the sampler.
#             batch_size: Batch size for the sampler.
#             length: Length of the dataset.
#             **kwargs: Additional arguments to pass to the dataset_class.
#         """
#         super().__init__()
#         self.dataset_class = dataset_class
#         self.patch_size = patch_size
#         self.batch_size = batch_size
#         self.length = length
#         self.dataset = dataset
#         self.kwargs = kwargs

#     def setup(self, stage: str) -> None:
#         """Set up datasets and samplers.

#         Args:
#             stage: Either 'fit', 'validate', 'test', or 'predict'.
#         """
#         # Instantiate the dataset
#         # self.dataset = self.dataset_class(**self.kwargs)

#         # Seed generator for deterministic splitting
#         generator = torch.Generator().manual_seed(0)

#         # Split the dataset into training, validation, and test datasets
#         self.train_dataset, self.val_dataset, self.test_dataset = random_bbox_assignment(
#             self.dataset, [0.6, 0.2, 0.2], generator
#         )

#         # Set up samplers based on the stage
#         if stage in ["fit"]:
#             self.train_batch_sampler = RandomBatchGeoSampler(
#                 self.train_dataset, self.patch_size, self.batch_size, self.length
#             )

#         if stage in ["fit", "validate"]:
#             self.val_sampler = GridGeoSampler(
#                 self.val_dataset, self.patch_size, self.patch_size
#             )

#         if stage == "test":
#             self.test_sampler = GridGeoSampler(
#                 self.test_dataset, self.patch_size, self.patch_size
#             )
