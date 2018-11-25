import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_loss_generator', type=str, default='training_loss_g.npy')
    parser.add_argument('--validation_loss_generator', type=str, default='validation_loss_g.npy')
    parser.add_argument('--interpolated_psnr', type=str, default='psnrint.npy')
    parser.add_argument('--generated_psnr', type=str, default='psnrgen.npy')

    args = parser.parse_args()

    tl_g = np.load(args.training_loss_generator)
    vl_g = np.load(args.validation_loss_generator)

    #tl_d = np.load(args.training_loss_discriminator)
    #vl_d = np.load(args.training_loss_discriminator)

    psnrint = np.load(args.interpolated_psnr)
    psnrgen = np.load(args.generated_psnr)

    epochs = psnrint.shape[0]

    fig1 = plt.figure()
    plt.plot(np.array([i+1 for i in range(epochs)]), psnrint)
    plt.plot(np.array([i+1 for i in range(epochs)]), psnrgen)
    plt.legend(['PSNR of Interpolated', 'PSNR of Generated'])
    plt.title('PSNR Value per Epoch for Select Sample')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (High is better image)')
    plt.show()
    fig1.savefig('PSNROverEpochs.png', dpi = fig1.dpi)

    fig2 = plt.figure()
    plt.plot(np.array([i+1 for i in range(args.images)]), tl_g)
    plt.plot(np.array([i+1 for i in range(args.images)]), vl_g)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    fig2.savefig('GeneratorLossOverEpochs.png', dpi = fig2.dpi)