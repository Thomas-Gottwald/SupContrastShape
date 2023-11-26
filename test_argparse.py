import argparse

def parse_option():
    parser = argparse.ArgumentParser('arguments')

    parser.add_argument('--diff_folder', type=str, default=None, help='path to diffused dataset')



    parser.add_argument('--aug', nargs='*', default=['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale'],
                        choices=['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale', 'sameResizedCrop', 'sameHorizontalFlip', 'sameColorJitter', 'sameGrayscale'],
                        type=str, help='list of the used image augmentations')
    defaultResizedCrop = [0.2, 1.0, 3/4, 4/3]
    parser.add_argument('--resizedCrop', nargs='+', default=defaultResizedCrop,
                        type=float, help='crop scale lower and upper bound and resize lower and upper bound for aspect ratio')
    parser.add_argument('--horizontalFlip', default=0.5, type=float, help='probability for horizontal flip')
    defaultColorJitter = [0.08, 0.4, 0.4, 0.4, 0.4]
    parser.add_argument('--colorJitter', nargs='+', default=defaultColorJitter,
                        type=float, help='probability to apply colorJitter and how much to jitter brightness, contrast, saturation and hue')

    parser.add_argument('--grayscale', default=0.2, type=float, help='probability for random grayscale')


    opt = parser.parse_args()

    # check that of each augmentation type only one of the independent or identical is passed
    assert not('resizedCrop' in opt.aug and 'sameResizedCrop' in opt.aug)\
        and not('horizontalFlip' in opt.aug and 'sameHorizontalFlip' in opt.aug)\
        and not('colorJitter' in opt.aug and 'sameColorJitter' in opt.aug)\
        and not('grayscale' in opt.aug and 'sameGrayscale' in opt.aug)

    if len(opt.resizedCrop) < len(defaultResizedCrop):
        opt.resizedCrop.extend(defaultResizedCrop[len(opt.resizedCrop):])
    elif len(opt.resizedCrop) > len(defaultResizedCrop):
        opt.resizedCrop = opt.resizedCrop[:len(defaultResizedCrop)]

    if len(opt.colorJitter) < len(defaultColorJitter):
        opt.colorJitter.extend(defaultColorJitter[len(opt.colorJitter):])
    elif len(opt.colorJitter) > len(defaultColorJitter):
        opt.colorJitter = opt.colorJitter[:len(defaultColorJitter)]

    return opt


def main():
    opt = parse_option()

    print(opt)


if __name__ == '__main__':
    main()