import torch
import torch.nn as nn
torch.set_printoptions(threshold=10000)


from PIL import Image
import os

import numpy as np

from torchvision import transforms as T


def saveFeatureMaps(model, layer=1):
    root = './feature_maps'
    if not os.path.exists(root):
        os.mkdir(root)
    for idx, m in enumerate(model.linear):
        print(m)
        if isinstance(m, nn.Linear):
            print('Linear ', idx)
            weight = m.weight.view(18,-1,512)
            print(weight.size())
            for s in range(weight.size(1)):
                w1 = weight[:,s,:]
                for w in w1:
                    w = sorted(w.detach().cpu().numpy(), reverse=True)
                    print(w[0:10])
                print('\n')

            # for w in range(weight.size(1)):
            #     print(weight[:,w,:], end='\n')
            #     break
            break


            # img = np.moveaxis(weight.cpu().detach().numpy(), 0, -1)
            # img_resized = np.moveaxis(img_resized[0].cpu().detach().numpy().squeeze(), 0, -1)

            # img_array = img*255
            # Image.fromarray(img_array.astype(np.uint8)).resize((400, 400)).save(current_dir + '/img' + str(0) + '.png')

            # img_array = img_resized*255
            # Image.fromarray(img_array.astype(np.uint8)).resize((400, 400)).save(current_dir + '/img_resized' + str(0) + '.png')

            # f_map = np.moveaxis(feature_maps_ref[0].cpu().detach().numpy().squeeze(), 0, -1)

            # if not os.path.exists(current_dir):
            #     os.mkdir(current_dir)

            # for f in range(f_map.shape[-1]):
            #     img_array = f_map[:,:,f]
            #     img_array = ((img_array - np.min(f_map))/np.max(f_map))*255
            #     f_map_img = Image.fromarray(img_array.astype(np.uint8)).convert('L').resize((400, 400))
                
            #     img_array = img_resized*255
            #     img_resized_img = Image.fromarray(img_array.astype(np.uint8)).resize((400, 400))
                
            #     img = np.expand_dims(np.asarray(f_map_img), 2).repeat(3,2)/255 * np.asarray(img_resized_img)
            #     Image.fromarray(img.astype(np.uint8)).save(current_dir + '/feature_map_' + str(0) + '_' + str(f) + '.png')


class GetFeatureMaps(nn.Module):
    """ 
        GetFeatureMaps: model that returns the flattened feature maps in a certain layer for a certain model
        
         """
    def __init__(self, model = None, layer = 45):
         
        super(GetFeatureMaps, self).__init__()

        # The Sequential element is the first dictionary of the model and it's where we can find the inception modules.
        self.feature_map = nn.Sequential(*model[0:layer+1])

    def forward(self, imgs):
        return self.feature_map(imgs).view(imgs.size(0), -1)


def transform_img(img, img_size):
    transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std of the imageNet dataset
    ]
    transform = T.Compose(transforms)
    
    return transform(img).unsqueeze(0)


def compute_loss(f_maps, f_maps_ref, lambdas):
    assert len(f_maps) == len(f_maps_ref)

    num_f_maps = len(f_maps)
    loss = 0
    for i in range(num_f_maps):
        loss += (nn.functional.l1_loss(f_maps[i], f_maps_ref[i])/num_f_maps)*lambdas[i]
        # print("LOSSS " + str(i) + " -->",  nn.functional.l1_loss(f_map_norm, f_map_ref_norm))
    return loss