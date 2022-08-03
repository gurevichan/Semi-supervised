from resnets import cifar_WRN_28_n, cifar_wide_resnet

model = cifar_wide_resnet(28, 2, 'preactivated', shortcut_type='B', dropout=0, l2_reg=2.5e-4)
import pdb;
pdb.set_trace()
a = 3