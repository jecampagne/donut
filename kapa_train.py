import random
import types
import os

#import swats


# the Network Model Classes
from kapa_network import *
# Data set & augmentation
from kapa_image_manip import *
#utility
from kapa_utils import *

# ####################### Training Generic  ###########################

def train(args, model, device, train_loader, transforms, optimizer, epoch, use_clipping=True):

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


        if use_clipping:
            # Clip the images to be between 0 and 1
            #        new_img_batch_noisy.clamp_(0.,1.)
            
            # alternative way for noise & cliping: rescale [min,max] to [0,1]
            scaler = MinMaxScaler()
            for i in range(batch_size):
                new_img_batch[i] = scaler(new_img_batch[i])
                new_img_batch_noisy[i] = scaler(new_img_batch_noisy[i])


        # send the inputs and target to the device
        new_img_batch,  new_img_batch_noisy = new_img_batch.to(device), \
                                              new_img_batch_noisy.to(device)

        # perform the optimizer loop
        optimizer.zero_grad()
        outputs = model(new_img_batch_noisy)        
        loss = F.mse_loss(outputs, new_img_batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # some debug
        if epoch < 5 and i_batch % args.log_interval == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, i_batch * batch_size, len(train_loader.dataset),
                 100. * i_batch / len(train_loader), loss.item()))

    return train_loss/len(train_loader)

# ####################### Test Generic  ###########################

def test(args, model, device, test_loader, transforms, epoch, use_clipping=True):
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
                    img = transforms[it](img)

                new_img_batch[i] = img


            # add gaussian noise
            new_img_batch_noisy = new_img_batch \
                                  + args.test_sigma_noise*torch.randn(*new_img_batch.shape)

            if use_clipping:
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


    # return stat
    test_loss /= len(test_loader)
    test_psnr /= len(test_loader)
    return {'loss': test_loss, 'psnr': test_psnr}

        
# ####################### MAIN ###########################

def main():

    args = types.SimpleNamespace(
        no_cuda = False,
        model = "DnCNN", # (ConvDenoiser, ConvDenoiserUp, ConvDenoiserUpV1, REDNet10,20,30), DnCNN
        use_clipping = False,  # for DnCNN
        batch_size = 64,
        epochs = 30,
        crop_size = 128,      # 2^n, max = 512 = no crop, the smaller the more data augmentation
        sigma_noise =0.02,    # to be defined  
        dir_path="/sps/lsst/data/campagne/convergence/",
        redshift=0.5,
        file_tag = "WLconv_z$0_",  # $ is a placeholder
        file_range = (1,5001), # 1 to 9999 max
        #
        test_batch_size = 64,
        test_crop_size = 128,
        test_sigma_noise = 0.02,
        test_redshift = 0.5,
        test_file_range = (5001,8001),
        #
        use_scheduler = True,
        resume = True,
        resume_scheduler = True,
        resume_optimizer = True,
        root_file = "/sps/lsst/users/campagne/kapaconv/",
        checkpoint_file ="model_DnCNN_128.pth.sav",
        history_loss_cpt_file="history_DnCNN_128.npy.sav"
        )

    # Verify arguments compatibility 
    if args.use_clipping == False and args.model != "DnCNN":
        print("model {args.model} use clipping, so we switch defacto")
        args.use_clipping = True


    print("\n### Training model ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))


    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use device....: ",device)

    # Pin_memory speed up the CPU->GPU transfert for NVidia GPU
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Train data
    strz = str(args.redshift).replace('.','')
    train_path = args.dir_path + 'Maps'+strz+'/'
    file_tag = args.file_tag.replace('$',str(args.redshift))
    
    train_dataset = DatasetKapa(dir_path=train_path,
                                file_tag=file_tag,
                                file_range=args.file_range)
        
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,drop_last=True,
        shuffle=True, **kwargs)

    print("train_loader length=",len(train_loader)," all: ", len(train_loader.dataset))
    n_images = len(train_dataset)
    args.log_interval = (n_images//args.batch_size)//10


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
    


    # Data augmentation for training
    train_transforms = [RandomApplyKapa([[flipH,flipV,identity],
                                         [rot90,rot180,rot270,identity]
                                         ]),
                        RandomCrop(size=args.crop_size),
                        ToTensorKapa()]

    test_transforms = [RandomCrop(size=args.test_crop_size),
                       ToTensorKapa()]


    # Some Networks clipping [0,1] is required: 
    if args.model == "ConvDenoiser":
        model = ConvDenoiser()
    elif args.model == "ConvDenoiserUp":
        model = ConvDenoiserUp()
    elif args.model == "ConvDenoiserUpV1":
        model = ConvDenoiserUpV1()
    elif args.model == "REDNet30":
        model = REDNet30()
    elif args.model == "DnCNN":
        model = DnCNN()
    else:
        assert False, "model unknown"
    
        
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
    test_loss_history  = []
    test_psnr_history  = []

    start_epoch = 0
    if args.resume :
        # load checkpoint of model/scheduler/optimizer
        if os.path.isfile(args.checkpoint_file):
            print("=> loading checkpoint '{}'".format(args.checkpoint_file))
            checkpoint = torch.load(args.checkpoint_file)
            # the first epoch for the new training
            start_epoch = checkpoint['epoch']
            # model update state
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.resume_scheduler:
                # scheduler update state
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print('=>>> scheduler not resumed')
            if args.resume_optimizer:
                # optizimer update state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print('=>>> optimizer not resumed')
            print("=> loaded checkpoint")
        else:
            print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
            return

        #load previous history of losses
        if os.path.isfile(args.history_loss_cpt_file):
            loss_history = np.load(args.history_loss_cpt_file)
            train_loss_history = loss_history[0].tolist()
            test_loss_history  = loss_history[1].tolist()
            test_psnr_history  = loss_history[2].tolist()
            
        else:
            print("=> FATAL no history loss checkpoint '{}'".format(args.history_loss_cpt_file))
            return
    else:
        print("=> no checkpoints then Go as fresh start")

    
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):

        print("process epoch[",epoch,"]: LR = ",end='')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        
        train_loss = train(args, model, device, train_loader, train_transforms, optimizer, epoch,
                           use_clipping = args.use_clipping)
        test_loss = test(args, model, device, test_loader, test_transforms, epoch,
                         use_clipping = args.use_clipping)


        print('Epoch {}, Train Loss: {:.6f}, Test Loss: {:.6f}, Test PSNR: {:.3f}'.format(epoch,train_loss,test_loss['loss'],test_loss['psnr']))
        # bookkeeping
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss['loss'])
        test_psnr_history.append(test_loss['psnr'])
        # scheduler update
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

        s_clip = "_clip" if args.use_clipping else ""
        f_name = args.model+"_"+str(args.crop_size)+s_clip
        torch.save(state,args.root_file+"/model_"+f_name+".pth")
        
        # save intermediate history
        np.save(args.root_file+"/history_"+f_name+".npy",
                np.array((train_loss_history,test_loss_history,test_psnr_history))
                )




################################
if __name__ == '__main__':
  main()
