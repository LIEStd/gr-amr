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
from amr.model import NN


class amr_pytorch(gr.sync_block):
    """
    Automatic Modulation Recognition With Pytorch
        Create a pytorch module for automatic modulation recognition.
        Input: Signal vector.
        Output: Modulation index according to parameter classes.
        Print: Prediction and possibility.
    Args:
        Norm Power: Whether to normalize power of input signal power.
        State Dict: The trained model state dict.
        Vec Length: Length of input signal vector.
        Classes: Modulation classes list.
        Cuda: whether to use cuda.
    """

    def __init__(self,  norm_power=True, state_dict='cnn11.pth',
                 vlen=512, classes='', cuda=False):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='AMR_Pytorch',  # will show up in GRC
            in_sig=[(np.complex64, vlen)],
            out_sig=[np.int32]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.norm_power = norm_power
        self.state_dict = state_dict
        self.classes = classes
        self.n_classes = len(classes)
        self.cuda = cuda
        self.model = self.load_model()


    def load_model(self):
        """
        Load the pytorch model.
        :return: trained pytorch model
        """
        try:
            model = NN(self.n_classes)
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
        """
        Normalize the power of input vectors
        :param x: input vectors
        :return: normalized vector
        """
        for i in range(x.shape[0]):
            points = x[i].shape[2]
            energy = np.sum(x[i] ** 2)
            power = energy / points
            x[i] = x[i] / sqrt(power)
        return x


    def work(self, input_items, output_items):
        """input items: 1 * n_items * item"""
        in0 = input_items[0]
        out = output_items[0]
        # print("Input items dim:", np.shape(input_items))

        # Split the complex data into real and imaginary floats for the ANN
        input_data = []
        for i in range(in0.shape[0]):
            item = np.array(in0[i])
            input_data.append(np.array([[item.real, item.imag]]))

        # print("Input data dim:", np.shape(input_data))

        input_data = np.array(input_data)
        if self.norm_power:
            input_data = self.normalize_power(input_data)

        input_tensor = torch.tensor(input_data)

        if self.cuda:
            input_tensor = input_tensor.cuda()
            self.model.cuda()

        # calculate neural network out
        out_distributions = np.array([])
        try:
            softmax = nn.Softmax(dim=1)
            out_distributions = softmax(self.model(input_tensor))
        except:
            print("Error While Predicting!")
            quit()

        # make prediction
        pred = torch.max(out_distributions, dim=1)[1].numpy()
        out[:] = pred

        for distribution in out_distributions:
            print("Predicting: {{{a}, {b}}}".format(a=self.classes[distribution.argmax()],
                          b=round(distribution[distribution.argmax()].item(), 4)))

        return len(input_items[0])

