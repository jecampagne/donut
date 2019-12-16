import types
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# the Network Model Classes
from kapa_network import *
# Data set & augmentation
from kapa_image_manip import *


# ####################### Utils ###########################
def display(img_orig, img_transf, img_transf_noisy, figsize=(10,10)):
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    ax0 = fig.add_subplot(gs[0, :])
    g0 = ax0.imshow(np.squeeze(img_orig))
    fig.colorbar(g0)
    ax0.set_title('Orig.')
    
    ax1 = fig.add_subplot(gs[1, 0])
    g1 = ax1.imshow(np.squeeze(img_transf))
    fig.colorbar(g1)
    ax1.set_title('Transf.')

    ax2 = fig.add_subplot(gs[1, 1])
    g2 = ax2.imshow(np.squeeze(img_transf_noisy))
    fig.colorbar(g2)
    ax1.set_title('Transf. + noise')

    plt.tight_layout()
    plt.show()

# ####################### TEST ###########################
def test(args, model, device, test_loader, transforms, epoch):
    # switch network layers to Testing mode
    model.eval()

    transf_size = len(transforms)

    test_loss = 0
    # turn off the computation of the gradient for all tensors
    with torch.no_grad():
        for i_batch, img_batch in enumerate(test_loader):
            batch_size = len(img_batch)

            new_img_batch = torch.zeros(batch_size,1,args.crop_size,args.crop_size)

            for i in range(batch_size):
                # transform the images
                img = img_batch[i].numpy()

                for it in range(transf_size):
                    img = transforms[it](img)

                new_img_batch[i] = img


            # add gaussian noise
            new_img_batch_noisy = new_img_batch \
                                  + args.sigma_noise*torch.randn(*new_img_batch.shape)

            # Clip the images to be between 0 and 1
            # new_img_batch_noisy.clamp_(0.,1.)

            # alternative way for noise & cliping
            scaler = MinMaxScaler()
            for i in range(batch_size):
                new_img_batch[i] = scaler(new_img_batch[i])
                new_img_batch_noisy[i] = scaler(new_img_batch_noisy[i])
            

            

            # send the inputs and target to the device
            new_img_batch,  new_img_batch_noisy = new_img_batch.to(device), \
                                                  new_img_batch_noisy.to(device)


            # denoise
            outputs = model(new_img_batch_noisy)
            # get the loss
            loss = F.mse_loss(outputs, new_img_batch)
            test_loss += loss.item() * batch_size

            # debug using display
            if i_batch <= 5:
                imgs       = new_img_batch.detach().cpu().numpy()                
                imgs_noisy = new_img_batch_noisy.detach().cpu().numpy()
                outputs    = outputs.detach().cpu().numpy()
##                 print('input: ',imgs.shape)
##                 print('noisy: ',imgs_noisy.shape)
##                 print('output ',outputs.shape)

##                 print("input")
##                 print(np.squeeze(imgs[0])[:10,:10])
                
##                 print("output")
##                 print(np.squeeze(outputs[0])[:10,:10])

                # plot the first ten input images and then reconstructed images
                fig, axes = plt.subplots(nrows=3, ncols=5,
                                         sharex=True, sharey=True, figsize=(10,4))
                
                # input images on top row, noisy middle, reconstructions on bottom
                for imgs, row in zip([imgs,imgs_noisy,outputs], axes):
                    for img, ax in zip(imgs, row):
                        ax.imshow(np.squeeze(img))
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                plt.show()

    # return stat
    return test_loss/len(test_loader.dataset)


# ####################### MAIN ###########################
def main():
    args = types.SimpleNamespace(
        batch_size = 10,
        no_cuda = False,
        crop_size=512,      # max = 512 = no crop, the smaller the more data augmentation
        sigma_noise =0.02,    # to be defined
        scale = 1.0, 
        dir_path="/sps/lsst/data/campagne/convergence/",
        redshift=0.5,
        file_tag = "WLconv_z$0_",  # $ is a placeholder
        file_range = (9001,9050), # 1 to 9999 max
        root_file = "/sps/lsst/users/campagne/kapaconv/",
        checkpoint_file = "model.pth"
        )

    #Define the test path
    strz = str(args.redshift).replace('.','')
    test_path = args.dir_path + 'Maps'+strz+'/'
    file_tag = args.file_tag.replace('$',str(args.redshift))

    #Checkpoint to load the trained model
    args.checkpoint_file = args.root_file + args.checkpoint_file

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device....: ",device)

    # Pin_memory speed up the CPU->GPU transfert for NVidia GPU
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    # Test data
    test_dataset = DatasetKapa(dir_path=test_path,
                                file_tag=file_tag,
                                file_range=args.file_range)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,drop_last=True,
        shuffle=False,**kwargs)

    print("test_loader length=",len(test_loader))

    # Allow random crop to the correct size
    test_transforms = [RandomCrop(size=args.crop_size),
                       ToTensorKapa()]
    
    # The Network
##    model = ConvDenoiserUp()
##    model = ConvDenoiserUpV1()
    model = REDNet30()
    # put model to device before loading scheduler/optimizer parameters
    model.to(device)
    # load trained state
    if os.path.isfile(args.checkpoint_file):
        print("=> loading checkpoint '{}'".format(args.checkpoint_file))
        checkpoint = torch.load(args.checkpoint_file)
        # model update state
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
        return


    epoch=0
    test_loss = test(args, model, device, test_loader, test_transforms, epoch)
    print('Epoch {}, Test Loss: {:.6f}'.format(epoch,test_loss))
    

################################
if __name__ == '__main__':
  main()
