import numpy as np

from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1).squeeze()
    mu2 = np.atleast_1d(mu2).squeeze()

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
    val = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return val


def calculate_mmd(x, y, kernel='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-9):
    def gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = source.shape[0] + target.shape[0]
        total = np.concatenate([source, target], axis=0)
        total0 = np.expand_dims(total, 0)
        total1 = np.expand_dims(total, 1)
        L2_distance = np.sum((total0 - total1) ** 2, axis=2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples + eps)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [np.exp(-L2_distance / (bandwidth_temp + eps)) for bandwidth_temp in bandwidth_list]
        return np.sum(kernel_val, axis=0)

    def linear_mmd2(f_of_X, f_of_Y):
        delta = f_of_X.mean(axis=0) - f_of_Y.mean(axis=0)
        loss = np.dot(delta, delta.T)
        return loss

    if kernel == 'linear':
        return linear_mmd2(x, y)
    elif kernel == 'rbf':
        batch_size = x.shape[0]
        kernels = gaussian_kernel(x, y, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        xx = np.mean(kernels[:batch_size, :batch_size])
        yy = np.mean(kernels[batch_size:, batch_size:])
        xy = np.mean(kernels[:batch_size, batch_size:])
        yx = np.mean(kernels[batch_size:, :batch_size])
        loss = xx + yy - xy - yx
        return loss
