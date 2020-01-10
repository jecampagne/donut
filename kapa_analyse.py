import random
import types
import os


# the Network Model Classes
from kapa_network import *
# Data set & augmentation
from kapa_image_manip import *
#utility
from kapa_utils import *


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ####################### Utils ###########################
def extractPatches(img,*,patch_size_x=64,patch_size_y=64,type='numpy'):
    # HWC for numpy And torch tensor !
    if type == 'torch':
        data = img.numpy()
    else:
        data = img
        
    h,w,c = data.shape

    assert w % patch_size_x == 0, \
        f"patch_size={patch_size_x} should devide W={w}"    
    assert h % patch_size_y == 0, \
        f"patch_size={patch_size_y} should devide H={h}"    

    list_of_patches = []
    n_x = w//patch_size_x
    n_y = h//patch_size_y
    print(n_x,n_y)
    for j in range(n_y):
        idy =j*patch_size_y
        for i in range(n_x):
            idx = i*patch_size_x
            list_of_patches.append(data[idy:idy+patch_size_y,idx:idx+patch_size_x,:])

    # attention au reshaping
    lop = np.array(list_of_patches).reshape(n_y,n_x,patch_size_y,patch_size_x,c)
    if type == 'torch':
        return torch.from_numpy(lop)
    else:
        lop

## #########
def gluePatches(lop,type='numpy'):
    # HWC for numpy and torch tensors
    ny,nx,psy,psx,c = lop.shape
    img = np.zeros((ny*psy,nx*psx,c)) # numpy
    for j in range(ny):
        idy =j*psy
        for i in range(nx):
            idx = i*psx
            img[idy:idy+psy,idx:idx+psx,:] = lop[j,i]
    if type == 'torch':
        return torch.from_numpy(img)
    else:
        return img

## #########

def display(img_orig, img_noisy, img_denoised, figsize=(10,10)):

    n = img_orig.shape[0]
    
    fig = plt.figure(figsize=(3,n))
    gs = gridspec.GridSpec(nrows=n, ncols=3,wspace=0.0, hspace=0.0)

    for i in range(n):
        img0 = np.squeeze(img_orig[i])
        img1 = np.squeeze(img_noisy[i])
        img2 = np.squeeze(img_denoised[i])

        vmin = np.min(img0)
        vmax = np.max(img0)
        print(vmin, vmax)

        ax0 = fig.add_subplot(gs[i, 0])
        g0 = ax0.imshow(img0,vmin=vmin,vmax=vmax)
        ax0.set(xticks=[], yticks=[])
#        fig.colorbar(g0)
#        ax0.set_title('Orig.')

        ax1 = fig.add_subplot(gs[i, 1])
        g1 = ax1.imshow(img1,vmin=vmin,vmax=vmax)
        ax1.set(xticks=[], yticks=[])
        pnsr_noisy = peak_signal_noise_ratio(img0,img1)
        ax1.text(0.1,0.9,f'{pnsr_noisy:0.2f}dB',transform=ax1.transAxes,
                 fontsize=10,bbox={'facecolor':'white', 'alpha':0.5, 'pad':2})
#        fig.colorbar(g1)
#        ax1.set_title(f'Noisy: psnr = {pnsr_noisy:0.2f}dB')

        ax2 = fig.add_subplot(gs[i, 2])
        g2 = ax2.imshow(img2,vmin=vmin,vmax=vmax)
        ax2.set(xticks=[], yticks=[])
        pnsr_denoised = peak_signal_noise_ratio(img0,img2)
        ax2.text(0.1,0.9,f'{pnsr_denoised:0.2f}dB',transform=ax2.transAxes,
                 fontsize=10,bbox={'facecolor':'white', 'alpha':0.5, 'pad':2})
#        fig.colorbar(g2)
#        ax2.set_title(f'Denoised: psnr = {pnsr_denoised:0.2f}dB')

    plt.tight_layout()
    plt.savefig('denoising.png')
    plt.show()

## ####################
def test(args, model, device, test_loader, transforms):
    # switch network layers to Testing mode
    model.eval()

    transf_size = len(transforms)

    
    test_loss = 0 
    test_psnr = 0
    # turn off the computation of the gradient for all tensors
    with torch.no_grad():
        for i_batch, img_batch in enumerate(test_loader):
            
            batch_size = len(img_batch)

            new_img_batch = torch.zeros(batch_size,1,args.test_crop_size,args.test_crop_size)

            for i in range(batch_size):
                # transform the images
                img = img_batch[i].numpy()
                for it in range(transf_size):
                    img = transforms[it](img)   # last transform HWC numpy to CHW tensor
                new_img_batch[i] = img
                
                        

            # add gaussian noise
            new_img_batch_noisy = new_img_batch \
                                  + args.test_sigma_noise*torch.randn(*new_img_batch.shape)

            
            
            # Clip the images to be between 0 and 1
            # new_img_batch_noisy.clamp_(0.,1.)

            # alternative way for noise & cliping using MinMaxScaler (min,max) -> [0,1]
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
            test_loss += loss.item()

            #psnr
            test_psnr += batch_psnr(outputs, new_img_batch)

            # display
            if i_batch == 0:
                display(new_img_batch[:5].data.cpu().numpy(),
                        new_img_batch_noisy[:5].data.cpu().numpy(),
                        outputs[:5].data.cpu().numpy(),
                        figsize=(10,15))


    # return stat
    test_loss /= len(test_loader)
    test_psnr /= len(test_loader)
    return {'loss': test_loss, 'psnr': test_psnr}

## ####################
def test_multipatches(args, model, device, test_loader, transforms):
    # switch network layers to Testing mode
    model.eval()

    transf_size = len(transforms)
    
    test_loss = 0 
    test_psnr = 0
    # turn off the computation of the gradient for all tensors
    with torch.no_grad():
        for i_batch, img_batch in enumerate(test_loader):
            
            batch_size = len(img_batch)

#            new_img_batch = torch.zeros(batch_size,1,args.test_crop_size,args.test_crop_size)

            for i in range(batch_size):
                # transform the images
                scaler = MinMaxScaler()
                img  = img_batch[i].numpy()
                img0 = img # original
                imgn = img # clone for noise
                img  = scaler(img) # scaling
                imgn = imgn + args.test_sigma_noise*np.random.randn(*imgn.shape)
                imgn0 = imgn # for dbg
                scalern = MinMaxScaler()
                imgn = scalern(imgn)
                assert np.isclose(img0,scaler.inverse_transform(img)), "gros (1) pb de re-scaling" 
                assert np.isclose(imgn0,scalern.inverse_transform(imgn)), "gros (2) pb de re-scaling"
                # At this stage img & imgn pixel values are in [0,1]
                # mini batch composed by extraction of patches
                orig_patches = extracPatches(img,
                                             patch_size_x=args.test_crop_size,
                                             patch_size_y=args.test_crop_size
                                        )
                noisy_patches = extracPatches(imgn,
                                              patch_size_x=args.test_crop_size,
                                              patch_size_y=args.test_crop_size
                                              )
                
                # to tensors
                transf_order = ToTensorKapa()
                orig_patches = transf_order(torch.from_numpy(orig_patches)).to(torch.float32)
                noisy_patches = transf_order(torch.from_numpy(noisy_patches)).to(torch.float32)

                # to device
                orig_patches, noisy_patches = orig_patches.to(device), \
                                              noisy_patches.to(device)
            

            


                # denoise
                denoised_patches = model(noisy_patches)
                # get the loss on the patches
                loss = F.mse_loss(denoised_patches, orig_patches)
                test_loss += loss.item()

                # rebuild the images from the patches
                img_test = gluePatches(orig_patches) # for test
                assert np.isclose(img,img_test), "rebuild img original failed"
                img_denoised = gluePatches(denoised_patches)

                # rescaling
                img_test_rescaled = scaler.inverse_transform(img_test)
                assert np.isclose(img0,img_test_rescaled), "gros (3) pb de re-scaling"
                img_denoised_rescaled = scalern.inverse_transform(img_denoised)
                
                #psnr
                test_psnr += peak_signal_noise_ratio(img0,img_denoised_rescaled)

                # display
                if i_batch == 0 and i<5:
                    display(img0,
                            imgn0,
                            img_denoised_rescaled,
                            figsize=(10,10))


    # return stat
    test_loss /= len(test_loader)
    test_psnr /= len(test_loader)
    return {'loss': test_loss, 'psnr': test_psnr}

## ####################
    
def main():
    
    args = types.SimpleNamespace(
        no_cuda = False,
        #
        dir_path="/sps/lsst/data/campagne/convergence/",
        file_tag = "WLconv_z$0_",  # $ is a placeholder
        #
        test_batch_size = 100,
        test_crop_size = 64,
        test_sigma_noise = 0.02,
        test_redshift = 0.5,
        test_file_range = (9000,10000),
        #
        root_file = "/sps/lsst/users/campagne/kapaconv/",
#        checkpoint_file ="DnCNN_64x64_20depth.pth",
        checkpoint_file ="REDNet30_64x64.pth",
        )

    args.checkpoint_file = args.root_file + args.checkpoint_file


    print("\n### Training model ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device....: ",device)

    # Pin_memory speed up the CPU->GPU transfert for NVidia GPU
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Test data
    strz = str(args.test_redshift).replace('.','')
    test_path = args.dir_path + 'Maps'+strz+'/'
    file_tag = args.file_tag.replace('$',str(args.test_redshift))


    test_dataset = DatasetKapa(dir_path=test_path,
                                file_tag=file_tag,
                                file_range=args.test_file_range)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,drop_last=True,
        shuffle=False,**kwargs)

    print("test_loader length=",len(test_loader))


    test_transforms = [ToTensorKapa()]

##    model = DnCNN(depth=20)
    model = REDNet30()
    model.to(device)

    # load checkpoint of model/scheduler/optimizer
    if os.path.isfile(args.checkpoint_file):
        print("=> loading checkpoint '{}'".format(args.checkpoint_file))
        checkpoint = torch.load(args.checkpoint_file)
        # model update state
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
        return

    
    
    test_loss = test_multipatches(args, model, device, test_loader, test_transforms)
    print('test_loss: ',test_loss)

################################
if __name__ == '__main__':
  main()
