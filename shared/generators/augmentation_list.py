import albumentations
import random


class AugmentationList():

    def __init__(self, *list_of_augs, shuffle=False):

        self.aug_list = list(list_of_augs)
        self.shuffle = shuffle

        valid_augmentations = all([isinstance(aug, albumentations.BasicTransform) for aug in self.aug_list])

        if not valid_augmentations:
            raise ValueError("Elements of `list_of_augs` must be valid albumentations augmentations")

    def get(self):

        if not self.shuffle:
            return self.aug_list

        else:
            shuffled_list = self.aug_list[:]
            random.shuffle(shuffled_list)
            return shuffled_list
