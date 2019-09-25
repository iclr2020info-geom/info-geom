def CIFAR10(h0):
    from torchvision import datasets, transforms
    from torchvision.transforms.functional import rotate
    from PIL import Image
    import numpy as np
    from math import sqrt
    CustomRotation = lambda img: rotate(img, (np.random.vonmises(0.0, 1.0) / (4 * np.pi)) * 180, resample=False,
                                         expand=False, center=None)
    CustomTranslation = \
        lambda img: img.transform(img.size, Image.AFFINE,
                                  (1, 0, np.random.randint(-4, 4), 0, 1, np.random.randint(-4, 4)), Image.NEAREST)
    IdentityTransform = lambda id: id
    average_augmented = 1.1509890384860935
    average_UNaugmented = 0.99985631921269
    scale_augmented = sqrt(h0 / average_augmented)
    scale_UNaugmented = sqrt(h0 / average_UNaugmented)

    augs = transforms.RandomChoice([
                                        IdentityTransform,
                                        CustomTranslation,
                                        CustomRotation,
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                                        transforms.RandomHorizontalFlip(),
                                    ])
    transform_train = transforms.Compose([augs,
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.49137255, 0.48235294, 0.44666667),
                                                                 (0.24705882, 0.24352941, 0.26156863)),
                                            lambda x: x * scale_augmented,
                                            ])
    transform_eval = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.49137255, 0.48235294, 0.44666667),
                                                                (0.24705882, 0.24352941, 0.26156863)),
                                           lambda x: x * scale_UNaugmented, ])

    return transform_train, transform_eval

def SVHN(h0):
    from torchvision import datasets, transforms
    from torchvision.transforms.functional import rotate
    from PIL import Image
    import numpy as np
    from math import sqrt
    CustomRotation = lambda img: rotate(img, (np.random.vonmises(0.0, 1.0) / (4 * np.pi)) * 180, resample=False,
                                         expand=False, center=None)
    CustomTranslation = \
        lambda img: img.transform(img.size, Image.AFFINE,
                                  (1, 0, np.random.randint(-4, 4), 0, 1, np.random.randint(-4, 4)), Image.NEAREST)
    IdentityTransform = lambda id: id
    average_augmented = 1.2600871324539185
    average_UNaugmented = 1.282157063484192
    scale_augmented = sqrt(h0 / average_augmented)
    scale_UNaugmented = sqrt(h0 / average_UNaugmented)

    augs = transforms.RandomChoice([
                                        IdentityTransform,
                                        CustomTranslation,
                                        CustomRotation,
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                                        transforms.RandomHorizontalFlip(),
                                    ])
    transform_train = transforms.Compose([ augs,
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4380, 0.4442, 0.4733),
                                                                 (0.19810377, 0.20114554, 0.19715817)),
                                            lambda x: x * scale_augmented,
                                            ])
    transform_eval = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4380, 0.4442, 0.4733),
                                                              (0.19810377, 0.20114554, 0.19715817)),
                                           lambda x: x * scale_UNaugmented,
                                         ])

    return transform_train, transform_eval














