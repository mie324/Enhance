testGAN2.py has our training loop
model_test_2.py is our model
buildnewyalefaces.py and pngtonpy.py build our datasets (latter makes the npy file needed to run the training loop)
visualize.py is used to create results from our testing set, input a string for the picture of your choice
psnrgen.npy and psnrint.npy have psnr values of psnr for our psnrgen.npy
evaluating.py has a function used to calculates psnr for our interpolated and generated images
training_loss_g.npy and validation_loss_g.npy store our loss over the various epochs
resultsPSNR.py if run will evaluate PSNR on the generated images in the results file as well as produce bicubically interpolated images and calculate their PSNR
