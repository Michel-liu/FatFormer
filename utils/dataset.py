import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset


class Dataset_Creator:
    def __init__(self, dataset_path, batch_size, num_workers=0, img_resolution=256, crop_resolution=224):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.all_dataset = {
            "val": ["airplane", "bird", "bottle", "car", "chair", "diningtable", "horse", "person", "sheep", "train", "bicycle", "boat", "bus", "cat", "cow", "dog", "motorbike", "pottedplant", "sofa", "tvmonitor"],
            "test": ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal", "DDPM", "DMBGIS", "IDDPM", "LDM", "PDS", "PNDM", "STSP4"] + ['dalle', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg']
        }

        self.transforms = {
            "val": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "test": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        }
        
    
    def build_dataset(self, spilt_dataset, selected_subsets="all"):
        assert spilt_dataset in self.all_dataset.keys()

        if selected_subsets == "all":
            selected_subsets = self.all_dataset[spilt_dataset]
        else:
            assert isinstance(selected_subsets, list)
            for subset in selected_subsets:
                # assert subset in self.all_dataset[spilt_dataset]
                pass

        
        sub_datasets = []
        for subset in selected_subsets:
            subset_path = os.path.join(self.dataset_path, spilt_dataset, subset)
            # identify multi-classes subset
            
            if "0_real" in os.listdir(subset_path) and "1_fake" in os.listdir(subset_path):
                sub_datasets.append(ImageFolder(
                    subset_path,
                    self.transforms[spilt_dataset]
                ))
            elif spilt_dataset == "test":
                tmp_datasets = []
                for sub_class in os.listdir(subset_path):
                    tmp_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
                sub_datasets.append(ConcatDataset(tmp_datasets))
            else:
                for sub_class in os.listdir(subset_path):
                    sub_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
        
        if spilt_dataset == "test":
            return sub_datasets, selected_subsets

        return ConcatDataset(sub_datasets)
