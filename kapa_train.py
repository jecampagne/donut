import random
import types
import os

import swats


# the Network Model Classes
from kapa_network import *
# Data set & augmentation
from kapa_image_manip import *


# ####################### Train ###########################

def train(args, model, device, train_loader, transforms, optimizer, epoch):

    assert transforms

    # switch network layers to Training mode
    model.train()

    transf_size = len(transforms)

    train_loss = 0 # to get the mean loss over the dataset

    
    for i_batch, img_batch in enumerate(train_loader):
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
        new_img_batch_noisy.clamp_(0.,1.)


        # send the inputs and target to the device
        new_img_batch,  new_img_batch_noisy = new_img_batch.to(device), \
                                              new_img_batch_noisy.to(device)

        # perform the optimizer loop
        optimizer.zero_grad()
        outputs = model(new_img_batch_noisy)

        print('MinMax In,Noisy,Out: ',
              new_img_batch.min(),'/',new_img_batch.max(),', ',
              new_img_batch_noisy.min(),'/',new_img_batch_noisy.max(),', ',
              outputs.min(),'/',outputs.max())
        

        loss = F.mse_loss(outputs, new_img_batch)
        train_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()

        # some debug
        if epoch < 5:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, i_batch * batch_size, len(train_loader.dataset),
                 100. * i_batch / len(train_loader), loss.item()))

    return train_loss/len(train_loader.dataset)
            
# ####################### MAIN ###########################

def main():

    args = types.SimpleNamespace(
        batch_size = 64,
        no_cuda = False,
        epochs = 100,
        crop_size = 64,      # 2^n, max = 512 = no crop, the smaller the more data augmentation
        sigma_noise =0.1,    # to be defined
        dir_path="/sps/lsst/data/campagne/convergence/",
        redshift=0.5,
        file_tag = "WLconv_z$0_",  # $ is a placeholder
        file_range = (1,5001), # 1 to 9999 max
        use_scheduler = True,
        resume = False,
        resume_scheduler = False,
        resume_optimizer = False,
        root_file = "/sps/lsst/users/campagne/kapaconv/",
        checkpoint_file ="",
        history_loss_cpt_file=""
        )

    start_epoch = 0


    #Define the train path
    strz = str(args.redshift).replace('.','')
    train_path = args.dir_path + 'Maps'+strz+'/'
    file_tag = args.file_tag.replace('$',str(args.redshift))
    

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device....: ",device)

    # Pin_memory speed up the CPU->GPU transfert for NVidia GPU
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Train data
    train_dataset = DatasetKapa(dir_path=train_path,
                                file_tag=file_tag,
                                file_range=args.file_range)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,drop_last=True,
        shuffle=True, **kwargs)

    print("train_loader length=",len(train_loader))
    n_images = len(train_dataset)
    args.log_interval = (n_images//args.batch_size)//10

    # Data augmentation 
    train_transforms = [RandomApplyKapa([[flipH,flipV,identity],
                                         [rot90,rot180,rot270,identity]
                                         ]),
                        RandomCrop(size=args.crop_size),
                        MinMaxScaler(),
                        ToTensorKapa()]


    # The Network
##    model = ConvDenoiser()
    model = ConvDenoiserUp()
##    model = ConvDenoiserUpV1()
##    model = REDNet30()
    # put model to device before loading scheduler/optimizer parameters
    model.to(device)

    # specify loss function

##     optimizer = swats.SWATS(model.parameters(),
##                             lr=0.01,
##                             weight_decay=0,  # default
##                             betas=(0.9, 0.999), # default
##                             eps=1e-8, #  1e-8 default but 
##                             amsgrad=False,
##                             nesterov=True,
##                             verbose=True
##                             )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
##     optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
##                                   weight_decay=0.01,
##                                   betas=(0.9, 0.999),
##                                   eps=1e-8, # 0.1, # was 1e-8 default but 
##                                   amsgrad=False
##                                   )


    print("Use ReduceLROnPlateau scheduler")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1, patience=5,
                                                     verbose=True)

    
    # set manual seeds per epoch JEC 2/11/19 fix seed 0 once for all
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)


    # Loop on the epochs
    train_loss_history = []

    for epoch in range(start_epoch, args.epochs + 1):

        print("process epoch[",epoch,"]: LR = ",end='')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        
        train_loss = train(args, model, device, train_loader, train_transforms, optimizer, epoch)

        print('Epoch {}, Train Loss: {:.6f}'.format(epoch,train_loss))
        # bookkeeping
        train_loss_history.append(train_loss)
        scheduler.step(train_loss)

        # save state
        if args.use_scheduler:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }
        else:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }
        torch.save(state,args.root_file+"/model.pth")
        
        # save intermediate history
        np.save(args.root_file+"/history.npy",
                np.array((train_loss_history))
                )




################################
if __name__ == '__main__':
  main()
