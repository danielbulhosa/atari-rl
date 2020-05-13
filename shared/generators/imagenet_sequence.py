from keras.utils import Sequence
import numpy as np
import albumentations
import cv2
import math
from shared.generators.augmentation_list import AugmentationList


class AlexNetSequence(Sequence):
    def __init__(self, x_paths, y_labels, batch_size,
                 images_dir, eigenvectors, eigenvalues,
                 pixel_avg, stdev, scale_shift,
                 aug_list, mode):
        """
        We initialize the AlexNetSequence class by
        passing it all of the paths to the images we
        use for training along with a matching array
        with the class labels.

        The sequence class itself takes care of taking
        the paths in a mini-batch and converting them
        to the correspoding image arrays.
        """
        self.x_paths, self.y_labels = x_paths, y_labels
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.pixel_avg = pixel_avg
        self.stdev = stdev
        self.scale_shift = scale_shift
        self.aug_list = aug_list
        self.mode = mode

        assert isinstance(aug_list, AugmentationList), "`aug_list` must be of class AugmetationList"

    def __len__(self):
        """
        We appear to define the length of a sequence
        to be (approximately) the number of batches in
        said sequence.

        This is likely the case so that there is a clear
        number of iterations defined for each epoch by
        the Sequence itself.

        The docs say this is one of the methods that MUST
        be implemented by any custom Sequence subclass.

        Note number of batches is same for test, just that
        batch size is always bigger (by 10x)
        """
        return int(np.ceil(len(self.x_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        The second method required to be implemented by
        the Sequence interface. This is the method used to
        generate minibatches.

        In this method we take a minibatch of image paths,
        convert them to arrays, apply our augmentations, and
        then return the augmented image array batch.
        """

        # Have programmatically checked that paths and labels match as expected
        x_batch_paths = self.x_paths[idx * self.batch_size:(idx + 1) *
                                     self.batch_size]

        if self.mode in ['train', 'validate']:

            x_batch = np.stack([self.aug(cv2.imread(self.images_dir + x_path))
                               for x_path in x_batch_paths], axis=0)

            y_batch = self.y_labels[idx * self.batch_size:(idx + 1) *
                                                          self.batch_size]

        else:
            # Flatten 10 augmentations and stack all together
            x_batch = np.stack([image for x_path in x_batch_paths
                                for image in self.aug(cv2.imread(self.images_dir + x_path))], axis=0)
            # Repeat each label 10 times
            y_batch = np.array([label for label in self.y_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
                                for repeat in range(10)])

        return x_batch, y_batch

    @staticmethod
    def pad_image(image, x_pad_to=224, y_pad_to=224):
        pad_vals = ((0, 0), (0, 0), (0, 0))
        width, height = image.shape[0], image.shape[1]

        pad_x, pad_y = max(x_pad_to - width, 0), max(y_pad_to - height, 0)
        pad_x_upper, pad_x_lower = int(math.ceil(pad_x / 2)), int(math.floor(pad_x / 2))
        pad_y_upper, pad_y_lower = int(math.ceil(pad_y / 2)), int(math.floor(pad_y / 2))

        assert pad_x_upper + pad_x_lower == pad_x, "Error with horizontal padding"
        assert pad_y_upper + pad_y_lower == pad_y, "Error with vertical padding"

        pad_dims = ((pad_x_upper, pad_x_lower), (pad_y_upper, pad_y_lower), (0, 0))

        return np.pad(image, pad_dims, 'constant', constant_values=pad_vals)

    @staticmethod
    def train_augment(image, eigenvectors, eigenvalues, pixel_avg, stdev, shift_scale, aug_list):
        """
        Takes an image from the training set, applies a random
        224 x 224 crop and a horizontal flip with 50% probability.
        Additionally it sums a random linear combination of the
        eigenvectors of all the pixels in the training set to all
        of the pixels in the training image. We also normalize the
        pixel values.

        Since we are not generating all of the possible transformation
        combinations (but only randomly returning a single one), do
        we need more epochs or do we sample more times each epoch?
        """

        # Note height comes first since the 0-axis is the y-axis! Made a mistake here before
        height, width = image.shape[0], image.shape[1]

        crop_width = min(224, width)
        crop_height = min(224, height)

        augmenter = albumentations.Compose(
            aug_list.get() +  # We add custom augmentations separately
            [
            # Note height comes first in crop transformations, made a mistake here before
            albumentations.RandomCrop(crop_height, crop_width)
        ])

        alphas = np.random.normal(scale=shift_scale, size=3)
        shift = np.zeros(3)

        for i in range(3):
            shift += alphas[i] * eigenvalues[i] * eigenvectors[i]

        # Shift was normalized but not standardized, hence need to divide it by standard deviation
        # Since standard deviation is less than 1 this will increase the shift and thus increase regularization...
        output_image = ((augmenter(image=image)["image"] - pixel_avg) / 255 + shift)/stdev

        return AlexNetSequence.pad_image(output_image)

    @staticmethod
    def validate_augment(image, pixel_avg, stdev):
        """
        Simply normalize image values and do a crop.
        """
        height, width = image.shape[0], image.shape[1]

        crop_width = min(224, width)
        crop_height = min(224, height)

        augmenter = albumentations.Compose([
            albumentations.CenterCrop(crop_height, crop_width)])

        output_image = ((augmenter(image=image)["image"] - pixel_avg) / 255)/stdev

        return AlexNetSequence.pad_image(output_image)

    @staticmethod
    def test_augment(image, eigenvectors, eigenvalues, pixel_avg, stdev):
        """
        Create ten different images per input image, all
        the different augmentations specified on the paper
        for test time.
        """
        height, width = image.shape[0], image.shape[1]

        crop_width = min(224, width)
        crop_height = min(224, height)

        crops = [albumentations.CenterCrop(crop_height, crop_width)]
        flipper = albumentations.HorizontalFlip()

        for corner in ['ur', 'lr', 'll', 'ul']:
            if corner[0] == 'u':
                y_max = height
                y_min = height - crop_height
            else:
                y_max = crop_height
                y_min = 0

            if corner[1] == 'r':
                x_max = width
                x_min = width - crop_width
            else:
                x_max = crop_width
                x_min = 0

            crops.append(albumentations.Crop(x_min, y_min, x_max, y_max))

        output_images = []
        for cropper in crops:
            for is_reflected in [True, False]:

                augmentations = [cropper, flipper] if is_reflected else [cropper]
                augmenter = albumentations.Compose(augmentations)
                output_image = ((augmenter(image=image)["image"] - pixel_avg) / 255)/stdev
                output_images.append(AlexNetSequence.pad_image(output_image))

        return output_images

    def aug(self, image):
        """
        A succinct method for convenience. The static one
        is useful for playing around with data in the notebook.

        Since our goal is not to try out different augmentations
        but just to reproduce those in paper, we have hardcoded
        our augmentation choices into this class. We could parametrize
        them in the constructor instead to accomplish the former
        goal.
        """
        if self.mode == "train":
            return AlexNetSequence.train_augment(image,
                                                 self.eigenvectors,
                                                 self.eigenvalues,
                                                 self.pixel_avg,
                                                 self.stdev,
                                                 self.scale_shift,
                                                 self.aug_list)
        elif self.mode == "validate":
            return AlexNetSequence.validate_augment(image,
                                                    self.pixel_avg,
                                                    self.stdev)

        elif self.mode == "test":
            return AlexNetSequence.test_augment(image,
                                                self.eigenvectors,
                                                self.eigenvalues,
                                                self.pixel_avg,
                                                self.stdev)

        else:
            raise ValueError("The `mode` parameter passed to the "
                             "constructor must be one of [`train`, `validate`, `test`]")