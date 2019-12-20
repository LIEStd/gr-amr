#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 LIEStd.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


from gnuradio import gr
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import pmt


class amr_pytorch(gr.sync_block):
    """
    Automatic Modulation Recognition With Pytorch
        Create a pytorch module for automatic modulation recognition.
    Args:
        Norm Power: Whether to normalize power of input signal power.
        State Dict: The trained model state dict.
        Vec Length: Length of input signal vector.
        Classes: Modulation classes list.
        Cuda: whether to use cuda.
    """

    def __init__(self,  norm_power=True, state_dict='cnn11.pth', vlen=512, classes='', cuda=False):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='AMR_Pytorch',  # will show up in GRC
            in_sig=[(np.complex64, vlen)],
            out_sig=[]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.norm_power = norm_power
        self.state_dict = state_dict
        self.classes = classes
        self.n_classes = len(classes)
        self.cuda = cuda
        self.message_port_register_out(pmt.intern('classification'))
        self.model = self.load_model()


    def load_model(self):
        try:
            model = CNN(self.n_classes)
            model.eval()
        except:
            print("Failed to load model!")
            quit()

        try:
            model.load_state_dict(torch.load(self.state_dict))
        except:
            print("Failed to load state dictionary!")
            quit()

        return (model)

    def normalize_power(self, x):
        for i in range(x.shape[0]):
            points = x[i].shape[2]
            energy = np.sum(x[i] ** 2)
            power = energy / points
            x[i] = x[i] / sqrt(power)
        return x



    def work(self, input_items, output_items):
        """example: multiply with constant"""
        """input items: 1 * n_items * item"""

        # print("Input items dim:", np.shape(input_items))
        input_data = []
        n_items = np.array(input_items[0]).shape[0]

        for i in range(n_items):
            item = np.array(input_items[0][i])

            # complex data must be split into real
            # and imaginary floats for the ANN
            input_data.append(np.array([[item.real, item.imag]]))

        # print("Input data dim:", np.shape(input_data))

        input_data = np.array(input_data)
        if self.norm_power:
            input_data = self.normalize_power(input_data)

        input_tensor = torch.tensor(input_data)

        if self.cuda:
            input_tensor = input_tensor.cuda()
            self.model.cuda()

        out_distributions = np.array([])

        try:
            softmax = nn.Softmax(dim=1)
            out_distributions = softmax(self.model(input_tensor))
        except:
            print("Error While Predicting!")
            quit()

        pmtv = pmt.make_dict()
        for distribution in out_distributions:
            pmtv = pmt.make_tuple(pmt.to_pmt(("{Prediction Probablity}:")),
                                  pmt.to_pmt((self.classes[distribution.argmax()], distribution[distribution.argmax()].item())))

            self.message_port_pub(pmt.intern("classification"), pmtv)

        return len(input_items[0])


class CNN(nn.Module):
    def __init__(self, out_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 7), stride=2, padding=(1, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.Dropout(0.2), nn.SELU())
        self.fc2 = nn.Linear(256, out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
