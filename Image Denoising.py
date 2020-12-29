import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import pickle
import time
from skimage.util import view_as_windows as viewW


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def run_denoising(image, model, denoise_function,
                  noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    denoise_mse = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        # print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        # print(np.mean((crop_image(noisy_images[:, :, i],
        #                           patch_size) - cropped_original) ** 2))
        # print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        denoise_mse.append(np.mean((cropped_original - denoised_images[i]) ** 2))
        # print(denoise_mse[-1])

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.gca().set_title(f'noise std = {noise_range[i]}')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
        plt.gca().set_title(f'MSE = {"{:.2e}".format(denoise_mse[i])}')
    plt.suptitle('Top row - noisy images\nBottom row - denoised images\nUsing MVN model')
    plt.show()
    return denoise_mse


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """

    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeX number_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return np.sum(multivariate_normal.logpdf(X.T, mean=model.mean, cov=model.cov))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    pis = np.log(model.mix)
    d, N = X.shape
    mean = np.zeros(d)
    logpdf_X = np.array([multivariate_normal.logpdf(X.T, mean, c) for c in model.cov])

    return np.sum([logsumexp(logpdf_X[:, i] + pis) for i in range(N)])


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    mean = np.mean(X, axis=1)
    cov = np.cov(X)
    return MVN_Model(mean, cov)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    # pi in shape (k,)
    # c in shape (k, N)
    # r in shape (k,)
    # initialization
    pi = np.ones(k) / k
    d, N = X.shape
    model_cov = np.cov(X)
    r = np.random.random_sample(k)
    covs_array = np.squeeze(np.split(r * np.stack([model_cov] * k, axis=2), k, axis=2))

    # define likelihood
    model_GSM = GSM_Model(covs_array, pi)
    likelihood = GSM_log_likelihood(X, model_GSM)
    prev_likelihood = -np.inf
    iter = 1
    likelihoods = [likelihood]
    # check for convergence
    while likelihood - prev_likelihood > 0.001:
        # E-step
        # distribution in shape (k, N)
        distribution = np.array([multivariate_normal(mean=np.zeros(d), cov=cov).pdf(X.T) for cov in covs_array])
        c = pi * np.eye(k) @ distribution / np.sum(pi * np.eye(k) @ distribution, axis=0)
        c[np.isnan(c)] = 0
        # M-step
        pi = np.average(c, axis=1)
        for i in range(k):
            c_i = c[i]
            r[i] = c_i @ np.diagonal(X.T @ np.linalg.inv(model_cov) @ X) / (d * np.sum(c_i))
        # print(f"iteration {iter}, likelihood is {likelihood}")
        iter += 1
        covs_array = np.squeeze(np.split(r * np.stack([model_cov] * k, axis=2), k, axis=2))
        prev_likelihood = likelihood
        model_GSM = GSM_Model(covs_array, pi)
        likelihood = GSM_log_likelihood(X, model_GSM)
        likelihoods.append(likelihood)
    return model_GSM, likelihoods


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """

    # TODO: YOUR CODE HERE


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    D = len(Y)
    cov = mvn_model.cov
    mean = mvn_model.mean
    var = noise_std ** 2
    # weiner's filter
    return np.linalg.inv(np.linalg.inv(cov) + np.eye(D) / var).dot(
        np.linalg.inv(cov).dot(mean).reshape((-1, 1)) + Y / var)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    D = len(Y)
    var = noise_std ** 2
    weiner_vector = []
    # weiner's filter when mean = 0 - depend only on Y linearly
    for i, cov in enumerate(gsm_model.cov):
        weiner_vector.append(gsm_model.mix[i] * np.linalg.inv(np.linalg.inv(cov) + np.eye(D) / var) @ (Y / var))
    return np.sum(weiner_vector, axis=0)


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    # TODO: YOUR CODE HERE


def main(train_path, test_path):
    """
    Function to run all models and create all graphs.
    """
    with open(train_path, 'rb') as f:
        train_imgs = pickle.load(f)
    times = []
    X = sample_patches(train_imgs)
    start = time.time()
    mvn_model = learn_MVN(X)
    times.append(time.time() - start)
    mvn_likelihood = MVN_log_likelihood(X, mvn_model)
    mvn_likelihood = float("{:.3f}".format(mvn_likelihood))
    start = time.time()
    gsm_model_3, likelihood = learn_GSM(X, 3)
    times.append(time.time() - start)
    plt.plot(likelihood)
    start = time.time()
    gsm_model_5, likelihood = learn_GSM(X, 5)
    times.append(time.time() - start)
    plt.plot(likelihood)
    start = time.time()
    gsm_model_10, likelihood = learn_GSM(X, 10)
    times.append(time.time() - start)
    plt.plot(likelihood)
    plt.legend(['GSM k=3', 'GSM k=5', 'GSM k=10'], title="Model")
    plt.ylabel('Likelihood')
    plt.xlabel('iterations')
    plt.title(f'Likelihood as a function of iterations\nMVN likelihood is {mvn_likelihood}')
    plt.show()
    plt.bar([1, 2, 3, 4], times, color='black')
    plt.xticks([1, 2, 3, 4], ['MVN', 'GSM k=3', 'GSM k=5', 'GSM k=10'])
    plt.title('Runtime for learning a model on train set')
    plt.ylabel('runtime[sec]')
    plt.show()
    # print('done creating the models')
    with open(test_path, 'rb') as f:
        test_imgs = pickle.load(f)
    gray_test_imgs = grayscale_and_standardize(test_imgs, remove_mean=False)
    # sorting to take the 6 smallest images for feasible runtime
    gray_test_imgs = sorted(gray_test_imgs, key=len)[:6]
    denoised_gsm_3 = []
    denoised_gsm_5 = []
    denoised_gsm_10 = []
    denoised_mvn = []
    for im in gray_test_imgs:
        mse = run_denoising(im, gsm_model_3, GSM_Denoise)
        denoised_gsm_3.append(mse)
        mse = run_denoising(im, gsm_model_5, GSM_Denoise)
        denoised_gsm_5.append(mse)
        mse = run_denoising(im, gsm_model_10, GSM_Denoise)
        denoised_gsm_10.append(mse)
        mse = run_denoising(im, mvn_model, MVN_Denoise)
        denoised_mvn.append(mse)

    denoised_gsm_3, denoised_gsm_5, denoised_gsm_10 = np.array(denoised_gsm_3), np.array(denoised_gsm_5), \
                                                      np.array(denoised_gsm_10)
    denoised_mvn = np.array(denoised_mvn)
    denoised_gsm_3 = np.mean(denoised_gsm_3, axis=0)
    denoised_gsm_5 = np.mean(denoised_gsm_5, axis=0)
    denoised_gsm_10 = np.mean(denoised_gsm_10, axis=0)
    denoised_mvn = np.mean(denoised_mvn, axis=0)
    noise = np.array([0.01, 0.05, 0.1, 0.2])
    plt.plot(noise, denoised_gsm_3)
    plt.plot(noise, denoised_gsm_5)
    plt.plot(noise, denoised_gsm_10)
    plt.plot(noise, denoised_mvn)
    plt.xlabel('noise')
    plt.ylabel('MSE')
    plt.legend(['GSM k=3', 'GSM k=5', 'GSM k=10', 'MVN'], title="Model")
    plt.title(f"Average over test set, MSE  as a function of noise std")
    plt.show()


if __name__ == '__main__':
    main('train_images.pickle', 'test_images.pickle')
