import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Non-Up solution
class ConvDenoiser(nn.Module):
    '''
    Work for images HxW= 32, 64, 128, 256, 512, ...
    '''
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2,padding=1,output_padding=1)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied to get number in [0,1]
        x = torch.sigmoid(self.conv_out(x))
                
        return x
#####################
# version with interpolation : upsample
class ConvDenoiserUp(nn.Module):
    def __init__(self):
        super(ConvDenoiserUp, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.conv3i = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2i = nn.Conv2d(16, 32, 3, padding=1)
        self.conv1i = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3i(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv2i(x))

        #last use a sigmoid to get numbers in [0,1]    
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv1i(x))
        
        return x

#####################
# More layers than ConvDenoiserUp
class ConvDenoiserUpV1(nn.Module):
    def __init__(self):
        super(ConvDenoiserUpV1, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)  
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.conv4i = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3i = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2i = nn.Conv2d(32, 64, 3, padding=1)
        self.conv1i = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # add fouth hidden layer
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4i(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3i(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv2i(x))

        #last use a sigmoid to get numbers in [0,1]    
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv1i(x))
        
        return x

###################
# from https://github.com/yjn870/REDNet-pytorch/blob/master/model.py
# cite: https://arxiv.org/pdf/1603.09056.pdf
class REDNet10(nn.Module):
    def __init__(self, num_layers=5, num_features=64, num_channels=1):
        super(REDNet10, self).__init__()
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += residual
##        out = self.relu(out)
        out = torch.sigmoid(out)
        return out

class REDNet20(nn.Module):
    def __init__(self, num_layers=10, num_features=64, num_channels = 1):
        super(REDNet20, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
##        x = self.relu(x)
        x = torch.sigmoid(x)
                

        return x


class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=64, num_channels = 1):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
##        x = self.relu(x)
        x = torch.sigmoid(x)
        return x
## ###############################
## https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py
## https://arxiv.org/pdf/1608.03981.pdf
    
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## ###########
## http://www.ipol.im/pub/art/2019/231/article.pdf
## https://arxiv.org/pdf/1710.04026.pdf
## source  http://www.ipol.im/pub/art/2019/231/?utm_source=doi

class UpSampleFeatures(nn.Module):
	r"""Implements the last layer of FFDNet
	"""
	def __init__(self):
		super(UpSampleFeatures, self).__init__()
	def forward(self, x):
		return functions.upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
	r"""Implements the middel part of the FFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):
		super(IntermediateDnCNN, self).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = input_features
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			raise Exception('Invalid number of input features')

		layers = []
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)
	def forward(self, x):
		out = self.itermediate_dncnn(x)
		return out

class FFDNet(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels):
		super(FFDNet, self).__init__()
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()

	def forward(self, x, noise_sigma):
		concat_noise_x = functions.concatenate_input_noise_map(\
				x.data, noise_sigma.data)
		concat_noise_x = Variable(concat_noise_x)
		h_dncnn = self.intermediate_dncnn(concat_noise_x)
		pred_noise = self.upsamplefeatures(h_dncnn)
		return pred_noise
