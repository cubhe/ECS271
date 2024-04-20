# Created by Renzhi He, UCDavis, 2024


import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 64),  # Input layer to hidden layer 1
            nn.ReLU(),
            nn.Linear(64, 64),  # Hidden layer 1 to hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 10)  # Hidden layer 2 to output layer
        )

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x=self.layers(x)
        out=self.softmax(x)
        return out


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
#import skimage
#from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
# from absl import flags
import logging


# from optics import PhaseObject3D, TomographySolver

#import contexttimer
#
# FLAGS = flags.FLAGS
#
# NUM_Z = "nz"
# INPUT_CHANNEL = "ic"
# OUTPUT_CHANNEL = "oc"
# MODEL_SCOPE = "infer_y"
# NET_SCOPE = "MLP"
# DNCNN_SCOPE = "DnCNN"
#
#
#
# # get total number of visible gpus
# NUM_GPUS = torch.cuda.device_count()
#
#
#
# def record_summary(writer, name, value, step):
#     writer.add_scalar(name, value, step)
#     writer.flush()
#
#
# def reshape_image(image):
#     if len(image.shape) == 2:
#         image_reshaped = image.unsqueeze(0).unsqueeze(-1)
#     elif len(image.shape) == 3:
#         image_reshaped = image.unsqueeze(-1)
#     else:
#         image_reshaped = image
#     return image_reshaped
#
# def remove_data(array, proportion_to_remove):
#     N = array.shape[0]
#     num_to_remove = int(np.round(N * proportion_to_remove))
#     shuffle_index=np.arange(N)
#     np.random.shuffle(shuffle_index)
#     indices_to_remove = shuffle_index[0:num_to_remove]
#     indices_to_remove = np.sort(indices_to_remove)
#     remaining_data = np.delete(array, indices_to_remove, axis=0)
#     org_index=np.arange(N)
#     indices_to_remain = np.delete(org_index, indices_to_remove, axis=0)
#     return remaining_data, indices_to_remove,indices_to_remain
#
# def insert_data_torch(original_tensor, data, indices_to_remove,indices_to_remain):
#     new_tensor = torch.zeros(data.shape).cuda().float()
#     new_tensor[indices_to_remain] = original_tensor.float()
#     new_tensor[indices_to_remove] = torch.tensor(data[indices_to_remove]).float()
#     return new_tensor
#
# #################################################
# # ***      CLASS OF NEURAL REPRESENTATION     ****
# #################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# class NPRF(nn.Module):
#     def __init__(self, FLAGS,RI=None, locations=None, name="model_summary"):
#         super(NPRF, self).__init__()
#         args=FLAGS
#         # Setup parameters
#         self.name = name
#         self.wavelength = args.wavelength
#         self.NA = args.NA
#         self.n_measure=args.n_measure
#         self.n_b=args.n_b
#         self.factor = args.factor
#
#         self.dz = args.dz / self.factor
#         self.layer = args.layers#int((layer+0.1) * 1)
#
#         self.dx = args.dx / self.factor
#         #
#         grid_x = args.H / self.dx
#         self.grid_x_org = int(grid_x * 1)
#         self.grid_x = int(grid_x * 1)
#         #self.dx = 2
#
#         self.dy = args.dy / self.factor
#         grid_y = args.W / self.dy #/ self.factor
#         self.grid_y_org = int(grid_y * 1)
#         self.grid_y = int(grid_y * 1)
#
#
#         self.shape_x_org = args.shape_x
#         self.shape_y_org = args.shape_y
#         self.shape_x = args.shape_x
#         self.shape_y = args.shape_y
#
#         self.grid_x_org = args.grid_x
#         self.grid_y_org = args.grid_y
#         self.grid_x = args.grid_x
#         self.grid_y = args.grid_y
#
#         if RI is None:
#             self.refractive_update=np.random.rand(self.shape_x*self.shape_y*self.layer,1)*0.1
#         # else:
#         #     RI_pre=RI[(self.grid_x-self.shape_x)//2:-(self.grid_x-self.shape_x)//2, (self.grid_y-self.shape_y)//2:-(self.grid_y-self.shape_y)//2,1:]
#         #     #RI_pre=
#         #     self.refractive_update = RI_pre.reshape(self.shape_x*self.shape_y*self.layer,1)-self.n_b
#         self.patch_ratio=args.patch_ratio
#
#         ####selfcalibration
#         self.locations=nn.Parameter(torch.tensor(locations))
#         dxyz=torch.tensor([self.dx,self.dy,self.dz])
#         self.dxyz=nn.Parameter(dxyz)
#         self.zz=nn.Parameter(torch.tensor([args.zz*100]))
#         #print(self.locations.mean)
#         #self.test=nn.Parameter(torch.tensor((1.)))
#         self.RI_4 = nn.Parameter(torch.empty((self.grid_x_org // 4, self.grid_y_org // 4, self.layer)))
#         self.RI_2 = nn.Parameter(torch.empty((self.grid_x_org // 2, self.grid_y_org // 2, self.layer)))
#         self.RI_1 = nn.Parameter(torch.empty((self.grid_x_org // 1, self.grid_y_org // 1, self.layer)))
#         ####neural network
#         self.xy_encoding_num=args.xy_encoding_num
#         self.z_encoding_num=args.z_encoding_num
#         self.positional_encoding_type = args.positional_encoding_type
#         self.dia_digree = args.dia_digree
#         self.output_scale = args.output_scale
#
#         # print('args.mlp_skip_layer', args.mlp_skip_layer)
#         # input_dim = int(180//self.dia_digree) * self.xy_encoding_num*2 + self.z_encoding_num * 2 #* self.xy_encoding_num
#         # #input_dim = 640
#         # output_dim = 1 ##RI
#         # # args.mlp_kernel_size=200
#         # # args.mlp_kernel_size=208
#         # self.inputlayer = nn.Linear(input_dim, args.mlp_kernel_size)
#         # self.skiplayer = nn.Linear(args.mlp_kernel_size + input_dim, args.mlp_kernel_size)
#         # # args.mlp_layer_num=6
#         # self.lineares = nn.ModuleList(
#         #     [nn.Linear(args.mlp_kernel_size, args.mlp_kernel_size) for i in range(args.mlp_layer_num)])
#         # self.outputlayer = nn.Linear(args.mlp_kernel_size, output_dim)
#         self.le_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     ###########################
#     ###     Neural Nets     ###
#     ###########################
#     def generate_coordinates(self, grid_x, grid_y, grid_z):
#         # 使用mgrid生成三维网格的坐标
#         x, y, z = np.mgrid[-grid_x / 2:grid_x / 2, -grid_y / 2:grid_y / 2, 0:grid_z]
#
#         # 将这些坐标合并并转换为(N, 3)的数组
#         coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
#         coordinates[:, 0]=coordinates[:,0] / grid_x
#         coordinates[:, 1]=coordinates[:,1] / grid_y
#         coordinates[:, 2]=coordinates[:,2] / grid_z*5 #* (grid_z/3)
#
#         return coordinates
#
#     def forward(self, light_loc_ids, training=True, steps=0,mask=None,ri_path=None,steps_c2f=None):
#         self.dx = self.dxyz[0]
#         self.dy = self.dxyz[0]
#         self.dz = self.dxyz[2]
#         # free_space=self.zz-self.dz*self.layer
#         free_space = self.zz - self.dz * self.layer
#         if ri_path is not None:
#             pass
#
#         else:
#             light_loc = self.locations[light_loc_ids]
#             # self.dx = self.dxyz[0]
#             # self.dy = self.dxyz[0]
#             # self.dz = self.dxyz[2]
#             # # free_space=self.zz-self.dz*self.layer
#             # free_space=self.zz-self.dz*self.layer
#             # print(self.locations.mean()*1e7)
#             #steps_c2f = [500, 1000, 2000]
#             #steps_c2f = [-1, -1, -1]
#             down_sampling_id = [4, 2, 1]
#             patch_ratios = [0.9, 0.975, 0.99375]
#             # down_sampling_id = [1, 1, 1]
#             # patch_ratios=[0.99375,0.99375,0.99375]
#
#
#             if steps < steps_c2f[0]:
#                 RI = self.RI_4[:, :, :]
#                 down_sampling = down_sampling_id[0]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             elif steps == steps_c2f[0]:
#                 RI = self.RI_4[:, :, :]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=2, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#                 self.RI_2.data = RI.clone().detach()
#                 RI = self.RI_2[:, :, :]
#                 down_sampling = down_sampling_id[1]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             elif steps < steps_c2f[1]:
#                 RI = self.RI_2[:, :, :]
#                 down_sampling = down_sampling_id[1]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             elif steps == steps_c2f[1]:
#                 RI = self.RI_2[:, :, :]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=2, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#                 self.RI_1.data = RI.clone().detach()
#                 RI = self.RI_1[:, :, :]
#                 down_sampling = down_sampling_id[2]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             elif steps < steps_c2f[2]:
#                 RI = self.RI_1[:, :, :]
#                 down_sampling = down_sampling_id[2]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             else:
#                 RI = self.RI_1[:, :, :]
#                 down_sampling = down_sampling_id[2]
#                 refractive_index = RI.permute(2, 0, 1).unsqueeze(0)  # 变为 (1, M, N, N)
#                 upsampled_tensor = F.interpolate(refractive_index, scale_factor=down_sampling, mode='bilinear')
#                 RI = upsampled_tensor.squeeze(0).permute(1, 2, 0)
#             #print(torch.max(RI), torch.min(RI))
#             #
#             RI = (self.sigmoid(RI)-0.5)/5#-0.1~0.1
#             RI = self.le_relu(RI)# 0-0.1
#             RI = RI + 1.33
#
#         #print(torch.mean(RI))
#         # patch=torch.ones((RI.shape[0],RI.shape[1],1))*1.33
#         # RI=torch.cat((RI,patch),dim=2)
#         intensity = self.rendering(light_loc, RI,free_space)
#
#         return RI-1.33, intensity, ..., self.locations
#
#
#     def __neural_repres(self, in_node, skip_layers=[], mlp_layer_num=10, kernel_size=256, L_xy=6, L_z=5):
#         # positional encoding
#         in_node=in_node.float()
#         if self.positional_encoding_type == "exp_diag":
#             s = torch.sin(torch.arange(0, 180, self.dia_digree) * np.pi / 180)[:, None]
#             c = torch.cos(torch.arange(0, 180, self.dia_digree) * np.pi / 180)[:, None]
#             fourier_mapping = torch.cat((s, c), dim=1).T
#             if(torch.cuda.is_available()):
#                 fourier_mapping = fourier_mapping.to('cuda').cuda()
#             xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
#
#             for l in range(self.xy_encoding_num):
#                 cur_freq = torch.cat(
#                     [
#                         torch.sin(2 ** l * np.pi * xy_freq),
#                         torch.cos(2 ** l * np.pi * xy_freq),
#                     ],
#                     dim=-1,
#                 )
#                 if l == 0:
#                     tot_freq = cur_freq
#                 else:
#                     tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
#                 # print('z', in_node[:, 2].max(), in_node[:, 2].min(), in_node[:, 2].mean())
#             for l in range(self.z_encoding_num):
#                 cur_freq = torch.cat(
#                     [
#                         torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
#                         torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
#                     ],
#                     dim=-1,
#                 )
#
#                 tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
#
#         # else:
#         #     raise NotImplementedError(FLAGS.positional_encoding_type)
#
#         # input to MLP
#
#         in_node = tot_freq
#         x = self.inputlayer(in_node)
#         x = self.le_relu(x)
#
#         layer_cout = 1
#         for f in self.lineares:
#             layer_cout += 1
#             if layer_cout in skip_layers:
#                 x = torch.cat([x, tot_freq], -1)
#                 x = self.skiplayer(x)
#                 x = self.le_relu(x)
#             x = f(x)
#             # print('x min a:',x.min())
#             x = self.le_relu(x)
#             # print('x min b:',x.min())
#         x = self.outputlayer(x)
#         output = self.le_relu(x)
#         #output = 2*torch.sigmoid(x)
#         #output = output / self.output_scale
#
#
#         return output
#
#     def rendering(self, light_source, refractive_index,free_space=75):
#         # print("refractive index shape:", refractive_index.shape)
#         # print("light source shape:", light_source.shape)
#         # print("input:",self.input[0])
#         self.wavelength = 0.6  # fluorescence wavelength
#         # objective immersion media
#
#         # background refractive index, PDMS
#         self.n_b = 1.33
#         #fx_illu_list = (light_source[ :, 0] - self.grid_x//2)*self.dx/self.grid_x
#         #fy_illu_list = (light_source[ :, 1] - self.grid_y // 2) * self.dy / self.grid_y
#         fx_illu_list = light_source[ :, 0]*self.dx
#         fy_illu_list = light_source[ :, 1]*self.dy
#         fz_illu_list = torch.zeros_like(fx_illu_list)#light_source[:,2]
#         intensityfield=self.multislice(refractive_index,  fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, dx=self.dx,
#                    dy=self.dy, dz=self.dz,free_space=free_space)
#
#         return intensityfield
#
#     def multislice(self, refractive_index, fx_illu_list, fy_illu_list, fz_illu_list, dx=0.2, dy=0.2, dz=0.2,free_space=75):
#         # Setup solver objects
#         solver_params = dict(wavelength=self.wavelength, na=self.NA, \
#                              RI_measure=self.n_measure, sigma=2 * np.pi * dz / self.wavelength, \
#                              fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, \
#                              pad=False, pad_size=(50, 50))
#         ## add value to the phantom
#         phase_obj_3d = PhaseObject3D(shape=refractive_index.shape, RI_obj=refractive_index,voxel_size=(dy, dx, dz), RI=self.n_b,free_space=free_space)
#         #phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] = phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] + refractive_index
#         #phase_obj_3d.RI_obj=refractive_index
#         solver_obj = TomographySolver(phase_obj_3d, **solver_params)
#         solver_obj.setScatteringMethod(model="MultiPhaseContrast")
#         forward_field_mb, fields = solver_obj.forwardPredict(field=True)
#
#         forward_field_mb = torch.squeeze(torch.stack(forward_field_mb))
#         intensityfield = torch.abs(forward_field_mb * torch.conj(forward_field_mb))
#         return intensityfield
#
#     def save(self, directory, epoch=None, train_provider=None):
#         if epoch is not None:
#             directory = os.path.join(directory, "{}_model/".format(epoch))
#         else:
#             directory = os.path.join(directory, "latest/".format(epoch))
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         path = os.path.join(directory, "model")
#         if train_provider is not None:
#             train_provider.save(directory)
#         torch.save(self.state_dict(), path)
#         print("saved to {}".format(path))
#         return path
#
#     def restore(self, model_path):
#
#         param = torch.load(model_path)
#         # param_model=self.state_dict()
#         # new_dict={}
#         # for k,v in param.items():
#         #     if k in param_model:
#         #         print(k)
#         #         print(v)
#         self.load_state_dict(param, strict=False)
#
#
#     def load_ri(self,RI):
#         self.refractive_update=RI
#
#
#
#
#
