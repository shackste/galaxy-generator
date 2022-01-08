from tqdm import trange
import numpy as np
from sklearn.manifold import TSNE
from scipy import linalg

import torch
from torch.utils.data import DataLoader, ConcatDataset
from chamferdist import ChamferDistance

from python_modules.evaluation.resnet_simclr import ResNetSimCLR
from python_modules.evaluation.fid import get_fid_fn, load_patched_inception_v3
from python_modules.evaluation.inception_score import inception_score
from python_modules.evaluation.vgg16 import vgg16
from python_modules.evaluation.dataset import MakeDataLoader, GANDataset, infinite_loader
from python_modules.evaluation.utils import get_device, get_config


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the features
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


class Evaluator:
    """Class aimed at evaluation of the trained model"""

    def __init__(self, config_path: str):
        self._config = get_config(config_path)
        self._device = get_device()
        self._generator, self._encoder = self._load_model()

    def evaluate(self):
        # compute FID score
        fid_score = self._compute_fid_score()
        print(f'{fid_score=}')

        # compute inception score (IS)
        i_score = self._compute_inception_score()
        print(f'Inception score: {i_score}')

        # compute Chamfer distance
        chamfer_dist = self._chamfer_distance()
        print(f'{chamfer_dist=}')

        # compute SSL FID score
        ssl_fid = self._compute_ssl_fid()
        print(f'{ssl_fid=}')

        # compute SSL PPL score
        ssl_ppl = self._compute_ppl('simclr')
        print(f'{ssl_ppl=}')

        # compute VGG PPL score
        vgg_ppl = self._compute_ppl('vgg')
        print(f'{vgg_ppl=}')

        # compute KID Inception
        kid_inception = self._compute_kid('inception')
        print(f'{kid_inception=}')

        # compute KID SSL
        kid_ssl = self._compute_kid('simclr')
        print(f'{kid_ssl=}')
        # compute morphological features

    def _get_dl(self) -> DataLoader:
        """Creates infinite dataloader from valid and test sets of images

        Returns:
            DataLoader: created dataloader
        """

        bs = self._config['batch_size']
        n_workers = self._config['n_workers']

        # load dataset
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']
        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=True)
        ds_valid = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        ds = ConcatDataset([ds_valid, ds_test])
        dl = infinite_loader(DataLoader(ds, bs, True, num_workers=n_workers, drop_last=True))
        return dl

    @torch.no_grad()
    def _compute_kid(self, encoder_type: str = 'simclr') -> float:
        """Computes KID score

        Args:
            encoder_type: type of encoder to use. Choices: simclr, inception

        Returns:
            float: KID score
        """

        if encoder_type not in ['simclr', 'inception']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = load_patched_inception_v3().to(self._device).eval()

        n_samples = 50_000
        bs = self._config['batch_size']
        n_batches = int(n_samples / bs) + 1

        dl = self._get_dl()

        features_real = []
        features_gen = []

        for _ in trange(n_batches):
            img, lbl = next(dl)
            img = img.to(self._device)

            lbl = lbl.to(self._device)
            latent = torch.randn((bs, self._generator.dim_z)).to(self._device)

            with torch.no_grad():
                img_gen = self._generator(latent, lbl)

                if encoder_type == 'inception':
                    if img.shape[2] != 299 or img.shape[3] != 299:
                        img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bicubic')

                    img_gen = torch.nn.functional.interpolate(img_gen, size=(299, 299), mode='bicubic')

                    h = encoder(img)[0].flatten(start_dim=1)
                    h_gen = encoder(img_gen)[0].flatten(start_dim=1)
                else:
                    h, _ = encoder(img)
                    h_gen, _ = encoder(img_gen)
            features_real.extend(h.cpu().numpy())
            features_gen.extend(h_gen.cpu().numpy())

        features_real = np.array(features_real)
        features_gen = np.array(features_gen)
        m = 1000  # max subset size
        num_subsets = 100

        n = features_real.shape[1]
        t = 0
        for _ in range(num_subsets):
            x = features_gen[np.random.choice(features_gen.shape[0], m, replace=False)]
            y = features_real[np.random.choice(features_real.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def _compute_ppl(self, encoder_type: str = 'simclr') -> float:
        """Computes perceptual path length (PPL)

        Args:
            encoder_type: type of encoder to use. Choices: simclr, vgg

        Returns:
            float: perceptual path length (smaller better)
        """

        if encoder_type not in ['simclr', 'vgg']:
            raise ValueError('Incorrect encoder')

        if encoder_type == 'simclr':
            encoder = self._encoder
        else:
            encoder = vgg16().to(self._device).eval()

        n_samples = 50_000
        eps = 1e-4
        bs = self._config['batch_size']
        n_batches = int(n_samples / bs) + 1

        dl = self._get_dl()

        dist = []
        for _ in trange(n_batches):
            img, label = next(dl)
            label = label.to(self._device)

            labels_cat = torch.cat([label, label])
            t = torch.rand([label.shape[0]], device=label.device)
            z0 = torch.randn((bs, self._generator.dim_z)).to(self._device)
            z1 = torch.randn((bs, self._generator.dim_z)).to(self._device)

            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + eps)

            with torch.no_grad():
                img = self._generator(torch.cat([zt0, zt1]), labels_cat)

                if encoder_type == 'simclr':
                    h, _ = encoder(img)
                else:
                    if img.shape[2] != 256 or img.shape[3] != 256:
                        img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bicubic')
                    h = encoder(img)

            h0, h1 = h.chunk(2)
            d = (h0 - h1).square().sum(1) / eps ** 2
            dist.extend(d.cpu().numpy())

        dist = np.array(dist)
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return ppl

    @torch.no_grad()
    def _compute_ssl_fid(self) -> float:
        """Computes FID on SSL features

        Returns:
            float: FID
        """
        n_samples = 50_000
        bs = self._config['batch_size']
        n_batches = int(n_samples / bs) + 1

        dl = self._get_dl()

        # compute activations
        activations_real = []
        activations_fake = []

        for _ in trange(n_batches):
            img, lbl = next(dl)
            img = img.to(self._device)
            img = (img - 0.5) / 0.5
            lbl = lbl.to(self._device)
            latent = torch.randn((bs, self._generator.dim_z)).to(self._device)

            with torch.no_grad():
                img_gen = self._generator(latent, lbl)
                img_gen = (img_gen - 0.5) / 0.5

                h, _ = self._encoder(img)
                h_gen, _ = self._encoder(img_gen)

            activations_real.extend(h.cpu().numpy())
            activations_fake.extend(h_gen.cpu().numpy())

        activations_real = np.array(activations_real)
        activations_fake = np.array(activations_fake)

        mu_real = np.mean(activations_real, axis=0)
        sigma_real = np.cov(activations_real, rowvar=False)

        mu_fake = np.mean(activations_fake, axis=0)
        sigma_fake = np.cov(activations_fake, rowvar=False)
        fletcher_distance = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
        return fletcher_distance

    @torch.no_grad()
    def _chamfer_distance(self) -> float:
        """Computes Chamfer distance between real and generated samples

        Returns:
            float: Chamfer distance
        """

        n_samples = 50_000
        bs = self._config['batch_size']
        n_workers = self._config['n_workers']
        n_batches = int(n_samples / bs) + 1

        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1)
        ds_val = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        ds = ConcatDataset([ds_val, ds_test])

        dl = infinite_loader(DataLoader(ds, bs, True, num_workers=n_workers))

        embeddings_real = []
        embeddings_gen = []
        # real data embeddings
        for _ in trange(n_batches):
            img, lbl = next(dl)
            img = img.to(self._device)
            img = (img - 0.5) / 0.5  # renormalize

            lbl = lbl.to(self._device)
            latent = torch.randn((bs, self._generator.dim_z)).to(self._device)

            with torch.no_grad():
                img_gen = self._generator(latent, lbl)
                img_gen = (img_gen - 0.5) / 0.5  # renormalize

                h, _ = self._encoder(img)
                h_gen, _ = self._encoder(img_gen)

            embeddings_real.extend(h.cpu().numpy())
            embeddings_gen.extend(h_gen.cpu().numpy())

        embeddings_real = np.array(embeddings_real, dtype=np.float32)
        embeddings_gen = np.array(embeddings_gen, dtype=np.float32)
        embeddings = np.concatenate((embeddings_real, embeddings_gen))
        tsne_emb = TSNE(n_components=3, n_jobs=16).fit_transform(embeddings)

        n = len(tsne_emb)
        tsne_real = np.array(tsne_emb[:n//2, ], dtype=np.float32)
        tsne_fake = np.array(tsne_emb[n//2:, ], dtype=np.float32)

        tsne_real = torch.from_numpy(tsne_real).unsqueeze(0)
        tsne_fake = torch.from_numpy(tsne_fake).unsqueeze(0)

        chamfer_dist = ChamferDistance()
        return chamfer_dist(tsne_real, tsne_fake).detach().item()

    @torch.no_grad()
    def _compute_inception_score(self) -> float:
        """Computes inception score (IS) for the model

        Returns:
            float: inception score (IS)
        """

        n_samples = 50_000
        batch_size = self._config['batch_size']

        # load dataset
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']
        size = self._config['dataset']['size']

        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=True)
        ds_valid = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        ds = ConcatDataset([ds_valid, ds_test])

        dataset = GANDataset(self._generator, ds, self._device, n_samples)
        score = inception_score(dataset, batch_size=batch_size, resize=True)[0]
        return score

    @torch.no_grad()
    def _compute_fid_score(self) -> float:
        """Computes FID score for the dataset

        Returns:
            float: FID score
        """

        # load dataset
        n_samples = 50_000
        path = self._config['dataset']['path']
        anno = self._config['dataset']['anno']

        # all images should be resized to 299 for the network, used for to calculate FID
        size = 299

        # updated version of MakeDataLoader is used, where size of the image can be changes
        # and custom paths to data and annotations can be passed directly
        make_dl = MakeDataLoader(path, anno, size, N_sample=-1, augmented=True)
        ds_valid = make_dl.dataset_valid
        ds_test = make_dl.dataset_test
        ds = ConcatDataset([ds_valid, ds_test])

        fid_func = get_fid_fn(ds, self._device, n_samples)

        with torch.no_grad():
            fid_score = fid_func(self._generator)
        return fid_score

    def _load_model(self):
        # load generator
        path_model = self._config['model_path']
        generator = torch.load(path_model)
        generator = generator.to(self._device).eval()

        # load encoder
        encoder_path = self._config['encoder']['path']
        base_model = self._config['encoder']['base_model']
        out_dim = self._config['encoder']['out_dim']
        n_channels = self._config['dataset']['n_channels']  # number of channels in the images (input and generated)

        encoder = ResNetSimCLR(base_model, n_channels, out_dim)
        ckpt = torch.load(encoder_path, map_location='cpu')
        encoder.load_state_dict(ckpt)
        encoder = encoder.to(self._device).eval()

        return generator, encoder


if __name__ == '__main__':

    config_path = './configs/biggan_eval.yaml'
    evaluator = Evaluator(config_path)
    evaluator.evaluate()
