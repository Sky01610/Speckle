from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def crop_image(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def convert_to_gray(image):
    return image.convert('L')

def convert_to_color(image):
    return image.convert('RGB')

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, color_mode='L'):
        super(TrainDatasetFromFolder, self).__init__()
        self.color_mode = color_mode
        phase_dataset_dir = join(dataset_dir, 'phase')
        amplitude_dataset_dir = join(dataset_dir, 'amplitude')
        noisy_dataset_dir = join(dataset_dir, 'noisy')
        self.phase_img_filenames = [join(phase_dataset_dir, x) for x in listdir(phase_dataset_dir) if is_image_file(x)]
        self.amplitude_img_filenames = [join(amplitude_dataset_dir, x) for x in listdir(amplitude_dataset_dir) if is_image_file(x)]
        self.noisy_img_filenames = [join(noisy_dataset_dir, x) for x in listdir(noisy_dataset_dir) if is_image_file(x)]
        self.crop_transform = crop_image(crop_size)

    def __getitem__(self, index):
        phase_img = Image.open(self.phase_img_filenames[index])
        amplitude_img = Image.open(self.amplitude_img_filenames[index])
        noisy_img = Image.open(self.noisy_img_filenames[index])

        if self.color_mode == 'L':
            phase_img = convert_to_gray(phase_img)
            amplitude_img = convert_to_gray(amplitude_img)
            noisy_img = convert_to_gray(noisy_img)
        elif self.color_mode == 'RGB':
            phase_img = convert_to_color(phase_img)
            amplitude_img = convert_to_color(amplitude_img)
            noisy_img = convert_to_color(noisy_img)

        phase_img = self.crop_transform(phase_img)
        amplitude_img = self.crop_transform(amplitude_img)
        noisy_img = self.crop_transform(noisy_img)

        return phase_img, amplitude_img, noisy_img

    def __len__(self):
        return len(self.phase_img_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, color_mode='L'):
        super(ValDatasetFromFolder, self).__init__()
        self.color_mode = color_mode
        phase_dataset_dir = join(dataset_dir, 'phase')
        amplitude_dataset_dir = join(dataset_dir, 'amplitude')
        noisy_dataset_dir = join(dataset_dir, 'noisy')
        self.phase_img_filenames = [join(phase_dataset_dir, x) for x in listdir(phase_dataset_dir) if is_image_file(x)]
        self.amplitude_img_filenames = [join(amplitude_dataset_dir, x) for x in listdir(amplitude_dataset_dir) if is_image_file(x)]
        self.noisy_img_filenames = [join(noisy_dataset_dir, x) for x in listdir(noisy_dataset_dir) if is_image_file(x)]
        self.crop_transform = crop_image(crop_size)

    def __getitem__(self, index):
        phase_img = Image.open(self.phase_img_filenames[index])
        amplitude_img = Image.open(self.amplitude_img_filenames[index])
        noisy_img = Image.open(self.noisy_img_filenames[index])

        if self.color_mode == 'L':
            phase_img = convert_to_gray(phase_img)
            amplitude_img = convert_to_gray(amplitude_img)
            noisy_img = convert_to_gray(noisy_img)
        elif self.color_mode == 'RGB':
            phase_img = convert_to_color(phase_img)
            amplitude_img = convert_to_color(amplitude_img)
            noisy_img = convert_to_color(noisy_img)

        phase_img = self.crop_transform(phase_img)
        amplitude_img = self.crop_transform(amplitude_img)
        noisy_img = self.crop_transform(noisy_img)

        return phase_img, amplitude_img, noisy_img

    def __len__(self):
        return len(self.phase_img_filenames)