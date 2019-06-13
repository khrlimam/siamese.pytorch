import os
import random
import pathlib
from bisect import insort_right
from collections import defaultdict

import torch.utils.data as data
import PIL
import PIL.Image
import torch
from typing import List, Any
from tqdm import tqdm


class SiamesePairDataset(data.Dataset):
    def __init__(self, root, ext: str = 'jpg', glob_pattern: str = "*/*.",
                 similar_factor: float = 1., different_factor: float = 0.38,
                 micro_factor: float = 0.38,
                 transform = None,
                 pair_transform = None,
                 target_transform = None):
        super(SiamesePairDataset, self).__init__()
        self._init_seed()
        self.similar_factor = similar_factor
        self.micro_factor = micro_factor
        self.different_factor = different_factor

        self.transform = transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.root: str = root

        # print(f"Files Mapping from {self.root}, please wait...")
        self.base_path = pathlib.Path(root)
        self.files = sorted(list(self.base_path.glob(glob_pattern + ext)))
        self.files_map = self._files_mapping()
        self.classes = list(self.files_map.keys())
        self.similar_pair = self._similar_pair()
        self.different_pair = self._different_pair()
        self.pair_files = self._pair_files()

    @staticmethod
    def _init_seed():
        random.seed(1261)

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        (imp1, imp2), sim = self.pair_files[idx]
        im1 = PIL.Image.open(imp1)
        im2 = PIL.Image.open(imp2)

        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        if self.pair_transform:
            im1, im2 = self.pair_transform(im1, im2)

        if self.target_transform:
            sim = self.target_transform(sim)
        return (im1, im2), sim

    def _files_mapping(self):
        dct = {}
        for f in self.files:
            spl = str(f).split('/')
            dirname = spl[-2]
            filename = spl[-1]
            if dirname not in dct.keys():
                dct.update({dirname: [filename]})
            else:
                dct[dirname].append(filename)
                dct[dirname] = sorted(dct[dirname])
        return dct

    def _similar_pair(self) -> List:
        # print("Generating Similar Pair, please wait...")
        fmap = self.files_map
        # atp = {}
        similar = []
        text = f'generating similar pair from\t {self.root}'
        bar = tqdm(fmap.keys(), desc=text)
        for key in bar:
            n = len(fmap[key])
            for (idz, idj, sim) in self._similar_sampling_generator(n):
                fz = os.path.join(self.base_path, key, fmap[key][idz])
                fj = os.path.join(self.base_path, key, fmap[key][idj])
                # atp[key].append(((fz, fj), 0))
                similar.append(((fz, fj), 0))
        num_sample = int(len(similar) * self.similar_factor)
        similar = random.sample(similar, num_sample)
        return similar

    def _similar_sampling_generator(self, n):
        num_sample = int(n * n * self.micro_factor)
        list_similar = [(i, j, 0) for i in range(n) for j in range(n)]
        sampled = random.sample(list_similar, num_sample)
        return sampled

    def _len_similar_pair(self):
        dct = {}
        for key in self.files_map.keys():
            dd = {key: len(self.similar_pair[key])}
            dct.update(dd)
        return dct

    def _pair_class_to_other(self) -> List:
        # print("Generating pair class to other class, please wait...")
        num = len(self.classes)
        list_idx = [i for i in range(num)]
        pair = []
        for idz in range(num):
            other_list = list_idx.copy()
            other_list.pop(idz)
            for idj in other_list:
                pair.append((idz, idj))
        num_sample = int(len(pair) * self.different_factor)
        pair = random.sample(pair, num_sample)
        # print("Generating pair class to other class, finished...")
        return pair

    def _diff_sampling_generator(self):
        # print("Creating diff sampling generator, please wait...")
        list_sampled: List[Any] = []
        for idx, (cidx, oidx) in enumerate(self._pair_class_to_other()):
            cname, cother = self.classes[cidx], self.classes[oidx]
            num_cname, num_cother = len(self.files_map[cname]), len(self.files_map[cother])
            num_sample = int(num_cname * num_cother * self.micro_factor)
            list_diff = [((cidx, i), (oidx, j), 1) for i in range(num_cname) for j in range(num_cother)]
            sampled = random.sample(list_diff, num_sample)
            list_sampled += sampled
        # print("Creating diff sampling generator, finished...")
        return list_sampled

    def _different_pair(self):
        # print("Generating Different Pair, please wait...")
        diff = []
        text = f'generating different pair from\t {self.root}'
        bar = tqdm(self._diff_sampling_generator(), desc=text)
        for z, j, sim in bar:
            zname, idz = self.classes[z[0]], z[1]
            jname, idj = self.classes[j[0]], j[1]
            zfile = self.files_map[zname][idz]
            jfile = self.files_map[jname][idj]
            fz = os.path.join(self.base_path, zname, zfile)
            fj = os.path.join(self.base_path, jname, jfile)
            diff.append(((fz, fj), 1))

        sim_len = len(self.similar_pair)
        if len(diff) > sim_len:
            diff = random.sample(diff, sim_len)
        # print("Generating Different Pair, finished...")
        return diff

    def _pair_files(self):
        sim_pair = self.similar_pair
        diff_pair = self.different_pair
        all_pair = sim_pair + diff_pair
        return all_pair
