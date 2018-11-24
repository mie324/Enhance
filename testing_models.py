import torch
import numpy as np
from matplotlib import pyplot as plt



Generator = torch.load('Generator.pt', map_location='cpu')



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
