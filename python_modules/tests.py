import unittest

import torch

from image_classifier import ImageClassifier

class MyTestCase(unittest.TestCase):
    def test_augmentation_handling(self):
        """ test whether multiple views of input image are recombined correctly"""
        classifier = ImageClassifier()

        images_test = torch.zeros(2,3,64,64)
        images_test[0,0,20] = 1
        images_test[0,1,:,20] = 1
        images_augmented = classifier.augment(images_test)
        self.assertEqual(images_augmented.shape, (32,3,45,45))

        classifier.N_conv_outputs = 1
        results = torch.tensor([[i] for img in images_augmented for i, color in enumerate(img) if torch.any(color)])
        recombined = classifier.recombine_augmentation(results)
        results = [torch.all(rec == i) for i, rec in enumerate(recombined)]
        self.assertTrue(results[0] and results[1])

if __name__ == '__main__':
    unittest.main()
