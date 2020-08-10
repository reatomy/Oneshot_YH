import sys, os
import random
import torch
from torch.utils.data import Dataset, DataLoader
# from skimage import io
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

import pdb

class DataManager(object):
    def __init__(self, bg_dir, eval_dir, seed, n_way = 20, train_sample_size = 30000):
        self.bg_dir = Path(bg_dir)
        self.eval_dir = Path(eval_dir)
        self.seed = seed
        self.train_sample_size = train_sample_size
        self.n_way = n_way

        self.bg_paths = self.load_path(self.bg_dir)
        print("ALPHABET IN BACKGROUND", ":", len(list(self.bg_paths.keys())))
        print("\t", list(self.bg_paths.keys())[:5])

        eval_paths = self.load_path(self.eval_dir)
        self.valid_paths = {alp: eval_paths[alp] for alp in random.sample(list(eval_paths.keys()), 10)}
        self.eval_paths = {alp: eval_paths[alp] for alp in eval_paths.keys() if alp not in self.valid_paths.keys()}
        print("ALPHABET IN VALIDATION", ":", len(list(self.valid_paths.keys())))
        print("\t", list(self.valid_paths.keys())[:5])
        print("ALPHABET IN EVALUATION", ":", len(list(self.eval_paths.keys())))
        print("\t", list(self.eval_paths.keys())[:5])


        self.sample_drawers()

        self.verification_dataset = VerificationDataset(path_dict = self.bg_paths, drawers = self.train_drawers, sample_size=self.train_sample_size)
        self.validation_dataset = OneshotDataset(path_dict = self.valid_paths, drawers = self.valid_drawers, n_way = self.n_way)
        self.evaluation_dataset = OneshotDataset(path_dict = self.eval_paths, drawers = self.test_drawers, n_way = self.n_way)

    def load_path(self, _dirpath):
        path_dict = {}
        for alphabet_name in os.listdir(_dirpath):
            alphabet_dir = os.path.join(_dirpath, alphabet_name)
            if os.path.isdir(alphabet_dir):
                if alphabet_name not in path_dict.keys():
                    path_dict[alphabet_name] = {}
                for character_name in os.listdir(alphabet_dir):
                    character_dir = os.path.join(alphabet_dir, character_name)
                    if os.path.isdir(character_dir):
                        if character_name not in path_dict[alphabet_name].keys():
                            path_dict[alphabet_name][character_name] = []
                        for char_img_name in os.listdir(character_dir):
                            char_img_path = os.path.join(character_dir, char_img_name)
                            path_dict[alphabet_name][character_name].append(char_img_path)

        return path_dict

    def sample_drawers(self, train_cnt = 12, valid_cnt = 4):
        random.seed(self.seed)
        t = random.sample(range(1, 21), train_cnt)
        t = sorted(t)
        _t = [i for i in range(1, 21) if i not in t]
        v = random.sample(_t, valid_cnt)
        v = sorted(v)
        tt = [i for i in _t if i not in v]

        self.train_drawers = int_list_to_strs(t)
        self.valid_drawers = int_list_to_strs(v)
        self.test_drawers = int_list_to_strs(tt)

        print("TRAIN DRAWERS:", self.train_drawers)
        print("VALID DRAWERS:", self.valid_drawers)
        print("TEST DRAWERS:", self.test_drawers)

class VerificationDataset(Dataset):
    # 30K, 90K, 150K training examples by sampling random same and different pairs
    # We fixed a uniform number of training examples per alphabet
    # 30 alphabets out of 50 and 12 drawers out of 20
    def __init__(self, path_dict, drawers, sample_size):
        # path_dict: path dictionary about alphabet, character and image
        # drawers: drawer list
        self.path_dict = path_dict
        self.drawers = drawers
        self.sample_size = sample_size
        self.data_pairs = self.sample_pairs()
        self.pair_statistics()
        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        if type(pair) == tuple:
            # Single sample

            images = [self.trans(Image.open(p)) for p in pair[:2]]
            label = torch.from_numpy(np.array([int(pair[2])], dtype=np.float32))

            return (images[0], images[1], label)

        else:
            res = []

            for p in pair:
                images = [self.trans(Image.open(_p)) for _p in p[:2]]
                label = torch.from_numpy(np.array([int(p[2])], dtype=np.float32))

                res.append((images[0], images[1], label))

            return res


        # images = [io.imread(p) for p in pair[:2]]
        # images = [self.trans(Image.open(p)) for p in pair[:2]]
        # label = torch.from_numpy(np.array([int(pair[-1])], dtype=np.float32))
        # return images[0], images[1], label

    def generate_pairs(self, total_examples):
        # Training example(alphabet pair) generator
        # Generates unique pair
        seen = set()
        alp_cnts = {k: 0 for k in self.path_dict.keys()}
        alp_max = int(self.sample_size/len(list(self.path_dict.keys()))) * 2
        total_example_size = len(total_examples)

        pair_indices = random.sample(range(total_example_size), 2)
        x = pair_indices[0]
        y = pair_indices[1]

        x_alp = total_examples[x][1]
        y_alp = total_examples[y][1]
        while True:
            seen.add((x, y))
            alp_cnts[x_alp] += 1
            alp_cnts[y_alp] += 1
            yield (x, y)
            pair_indices = random.sample(range(total_example_size), 2)
            x = pair_indices[0]
            y = pair_indices[1]
            x_alp = total_examples[x][1]
            y_alp = total_examples[y][1]
            while ((x, y) in seen) or (alp_cnts[x_alp] >= alp_max) or (alp_cnts[y_alp] >= alp_max):
                # Check already seen example / uniform number of alphabet -> create new pair
                pair_indices = random.sample(range(total_example_size), 2)
                x = pair_indices[0]
                y = pair_indices[1]

                x_alp = total_examples[x][1]
                y_alp = total_examples[y][1]

    def sample_pairs(self):
        res = []

        total_examples = []
        for alp in self.path_dict.keys():
            for c in self.path_dict[alp].keys():
                for i_path in self.path_dict[alp][c]:
                    if i_path.split(".")[-2][-2:] in self.drawers:
                        total_examples.append((i_path, alp, "_".join((alp, c))))

        total_classes = []
        for alp in self.path_dict.keys():
            for c in self.path_dict[alp].keys():
                total_classes.append("//".join((alp, c)))

        # g = gencoordinates(0, len(total_examples)-1)
        g = self.generate_pairs(total_examples)

        for idx in range(self.sample_size):

            label = idx % 2

            # TODO: UNIFORM NUMBER PROBLEM

            if label == 1:
                # Same class example
                c = random.choice(total_classes)
                i_path_list = self.path_dict[c.split("//")[0]][c.split("//")[1]]
                i_path_list = [i for i in i_path_list if i.split(".")[-2][-2:] in self.drawers]
                i_sample = random.sample(i_path_list, 2)
                i_sample.append(label)

                res.append(tuple(i_sample))

            else:
                # different class example

                cs = random.sample(total_classes, 2)
                i_sample = []
                for c in cs:
                    i_path_list = self.path_dict[c.split("//")[0]][c.split("//")[1]]
                    i_path_list = [i for i in i_path_list if i.split(".")[-2][-2:] in self.drawers]
                    i_item = random.choice(i_path_list)
                    i_sample.append(i_item)
                i_sample.append(label)

                res.append(tuple(i_sample))


            # idx_pair = next(g)
            # print(idx_pair)
            # print(total_examples[idx_pair[0]][0], total_examples[idx_pair[0]][1], total_examples[idx_pair[0]][2])
            # print(total_examples[idx_pair[1]][0], total_examples[idx_pair[1]][1], total_examples[idx_pair[1]][2])
            # print(total_examples[idx_pair[0]][2] == total_examples[idx_pair[1]][2])
            # raise

            # res.append((total_examples[idx_pair[0]][0], total_examples[idx_pair[1]][0], total_examples[idx_pair[0]][2] == total_examples[idx_pair[1]][2]))

            if (idx+1) % 10000 == 0:
                print("\t", (idx+1), "th pair was generated:")
                print("\t", res[idx][0], res[idx][1], res[idx][2])
                # print("\t", total_examples[idx_pair[0]][0], total_examples[idx_pair[0]][1], total_examples[idx_pair[0]][2])
                # print("\t", total_examples[idx_pair[1]][0], total_examples[idx_pair[1]][1], total_examples[idx_pair[1]][2])

        return res

    def pair_statistics(self):
        alp_cnts = {k: 0 for k in self.path_dict.keys()}
        same_cnt = 0

        for item in self.data_pairs:

            # x_alp = item[0].split('\\')[0].split('/')[-1]
            x_alp = Path(item[0]).parent.parent.name
            alp_cnts[x_alp] += 1
            # y_alp = item[1].split('\\')[0].split('/')[-1]
            y_alp = Path(item[1]).parent.parent.name
            alp_cnts[y_alp] += 1
            if item[2]:
                same_cnt += 1

        print("=================================")
        print("STATISTICS OF GENERATED EXAMPLES:")
        print("\tTOTAL EXAMPLE:", len(self.data_pairs))
        for alp in alp_cnts.keys():
            print('\t', alp, ":", alp_cnts[alp])
        print('\t', "CNT OF SAME CLASS:", same_cnt)

class OneshotDataset(Dataset):
    def __init__(self, path_dict, drawers, n_way):
        self.path_dict = path_dict
        self.drawers = drawers
        self.n_way = n_way
        self.data = self.make_trials()
        self.trans = transforms.ToTensor()

    def make_trials(self):
        res = []
        for alp in self.path_dict.keys():
            for trial_cnt in range(2):
                chars = random.sample(list(self.path_dict[alp].keys()), self.n_way)
                t_drawers = random.sample(self.drawers, 2)

                a_images = [self.path_dict[alp][c][int(t_drawers[0])-1] for c in chars]
                b_images = [self.path_dict[alp][c][int(t_drawers[1])-1] for c in chars]

                for a_idx, a_image in enumerate(a_images):
                    res.append((a_image, b_images, a_idx))

        return res


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trial = self.data[idx]

        if(type(trial) == tuple):
            # Single sample

            a_image = self.trans(Image.open(trial[0]))
            b_images = [self.trans(Image.open(b)) for b in trial[1]]
            label = torch.from_numpy(np.array([trial[2]], dtype=np.float32))

            return (a_image, b_images, label)

        else:
            res = []
            for t in trial:
                a_image = self.trans(Image.open(t[0]))
                b_images = [self.trans(Image.open(b)) for b in t[1]]
                label = torch.from_numpy(np.array([t[2]], dtype=np.float32))

                res.append((a_image, b_images, label))

            return res
        #
        #
        # pdb.set_trace()
        #
        # # a_image = io.imread(trial[0])
        # a_image = self.trans(Image.open(trial[0]))
        # # b_images = [io.imread(b) for b in trial[1]]
        # b_images = [self.trans(Image.open(b)) for b in trial[1]]
        # label = torch.from_numpy(np.array([trial[2]], dtype=np.float32))
        #
        # return a_image, b_images, label


def int_list_to_strs(l):
    # Drawer indices to list of string
    l = [format(i, '02d') for i in l]
    return l


if __name__ == "__main__":
    bg_dir = "../data/python/images_background/"
    eval_dir = "../data/python/images_evaluation/"
    seed = 10
    dm = DataManager(bg_dir = bg_dir, eval_dir = eval_dir, seed = seed)

    # train_vd = VerificationDataset(path_dict = dm.bg_paths, drawers = dm.train_drawers, sample_size = 30000)
    # v_dl = DataLoader(train_vd, batch_size=4, shuffle=True, num_workers=4)

    # od = OneshotDataset(path_dict = dm.eval_paths, drawers = dm.test_drawers, n_way = 20)
    # o_dl = DataLoader(od, batch_size=4, shuffle=True, num_workers=4)
    # print(od[:2])
