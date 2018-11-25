import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import ImgDataset
import torchvision.utils as vutils
from torch.autograd import Variable



def load_data(batch_size, training_set_feat, training_set_labels, validation_set_feat, validation_set_labels):
    train_dataset = ImgDataset(training_set_feat, training_set_labels)
    validation_dataset = ImgDataset(validation_set_feat, validation_set_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader



Generator = torch.load('Generator.pt', map_location='cpu')


'''
highres_images = np.load('./imagedataset/highresimages.npy')
lowres_images = np.load('./imagedataset/lowresimages.npy')


index = 213
real = torch.from_numpy(highres_images[index,:,:])
inp_orig = torch.from_numpy(lowres_images[index,:,:])
input = (torch.from_numpy(lowres_images[index,:,:])).unsqueeze(dim = 0)
input = input.unsqueeze(dim = 0)

gen_output = Generator(input.float())
gen_output = gen_output.squeeze()
gen_output = gen_output.squeeze()
gen_output = np.array(gen_output.detach().numpy())

print(np.shape(np.array(gen_output)))
print(type(gen_output))




plt.subplot(1, 3, 1)
plt.title("LowRes Image")
plt.imshow(inp_orig, interpolation='nearest', cmap='gray')
plt.subplot(1, 3, 2)
plt.title("HighRes GAN")
plt.imshow(gen_output, interpolation='nearest', cmap='gray')
plt.subplot(1, 3, 3)
plt.title("HighRes Original")
plt.imshow(real, interpolation='nearest', cmap='gray')


plt.show()

'''


torch.manual_seed(0)
np.random.seed(0)

batchSize = 64
labels = np.load('./imagedataset/highresimages.npy')
features = np.load('./imagedataset/lowresimages.npy')

seed = 0
training_set_feat, validation_set_feat, training_set_labels, validation_set_labels \
    = train_test_split(features, labels, test_size=0.2, random_state=seed)

training_loader, validation_loader = load_data(batchSize, training_set_feat, training_set_labels,
                                               validation_set_feat, validation_set_labels)

for epoch in range(25):
    for i, batch in enumerate(training_loader):
        lowres, real = batch




        lowres = np.array(lowres)
        real = np.array(real)

        print(np.shape(real))
        print(np.shape(lowres))


    #vutils.save_image(real[0], '%s/real_samples_epoch_%03d.png' % ("./temp_results_2", epoch), normalize=True)
    #fake = Generator(noise.float())
    #vutils.save_image(fake[0], '%s/fake_samples_epoch_%03d.png' % ("./temp_results", epoch), normalize=True)


