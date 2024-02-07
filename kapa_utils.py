import numpy as np

## #######
## to compare images
## code inspired https://github.com/scikit-image/scikit-image/blob/v0.16.1/skimage/metrics/simple_metrics.py#L108
def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
        BUT for non regular images it is based on true min,max values
    Returns
    -------
    psnr : float
        The PSNR metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    if not image_true.shape == image_test.shape:
        raise ValueError('Input images must have the same dimensions.')
        return
    #protection against true ~ test images
    eps=0.
    if np.allclose(image_true, image_test):
        eps=1.0e-12

    # usualy data_range is 1 or 255 for regular Gray-scaled or RGB image
    if data_range is None:
        # Do not use the data-type-min-max but the min/max of the true image
        # as the pixel values are positive ans negative wo a priori any bounds
        true_min, true_max = np.min(image_true), np.max(image_true)
        data_range = true_max - true_min

    image_true, image_test = _as_floats(image_true, image_test)
    err = np.mean((image_true - image_test) ** 2, dtype=np.float64)+eps
    return 10 * np.log10((data_range ** 2) / err)


def batch_psnr(img, imclean, data_range=None):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                                                data_range=data_range)
	return psnr/img_cpu.shape[0]
