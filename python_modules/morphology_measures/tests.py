import sys
sys.path.insert(0,"../")

import unittest
from pdb import set_trace

import torch

from measures import get_morphology_measures, get_morphology_measures_set
from statistics import evaluate_generator
from dataset import MakeDataLoader

make_data_loader = MakeDataLoader(N_sample=32)


class MyTestCase(unittest.TestCase):
    def test_get_morphology_measures(self):
        batch_size = 1
        data_loader = make_data_loader.get_data_loader_train(batch_size=batch_size, shuffle=False)  # , num_workers=4)
        for image, _ in data_loader:
            morph = get_morphology_measures(image)
            break
        self.assertTrue(morph.ellipticity_asymmetry)

    def test_get_morphology_measures_set(self):
        batch_size = 16
        data_loader = make_data_loader.get_data_loader_valid(batch_size=batch_size)  # , num_workers=4)
        for images, _ in data_loader:
#            set_trace()
            measures_target = get_morphology_measures_set(images)
            break
        self.assertIsInstance(measures_target.torch(), torch.Tensor)


    def test_evaluate_generator(self):
        batch_size = 16
        # first, compute distance of training data to itself
        data_loader = make_data_loader.get_data_loader_train(batch_size=batch_size)  # , num_workers=4)
        generator = lambda latent, label: next(iter(data_loader))[0]
        distances_same = evaluate_generator(data_loader, generator)
#        print(distances_same)
        # compare to distance of different datasets (train vs valid)
        data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size)  # , num_workers=4)
        distances_diff = evaluate_generator(data_loader_valid, generator)
#        print(distances_diff)
        self.assertTrue(distances_same["total"] < distances_diff["total"])  # add assertion here


if __name__ == '__main__':
    unittest.main()
