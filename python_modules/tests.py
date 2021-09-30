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

    def test_vanilla_discriminator(self):
        from vanilla_gan.gan import Discriminator
        discriminator = Discriminator()
        batch_size = 4
        images = torch.rand(batch_size, 3, 64, 64)
        valid = discriminator(images, 0)
        self.assertTrue(valid.shape == (batch_size,1))

    def test_vanilla_generator(self):
        from vanilla_gan.gan import Generator
        latent_dim = 128
        labels_dim = 37
        batch_size = 4
        latent = torch.randn(batch_size, latent_dim)
        labels = torch.rand(batch_size, labels_dim)
        generator = Generator(latent_dim=latent_dim, labels_dim=labels_dim)
        images = generator(latent, labels)
        print(images.shape)
        self.assertTrue(images.shape == (batch_size, 3, 64, 64))

if __name__ == '__main__':
    unittest.main()
