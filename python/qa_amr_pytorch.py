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

from gnuradio import gr, gr_unittest
from gnuradio import blocks
from amr_pytorch import amr_pytorch
import numpy as np


class qa_amr_pytorch(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_t(self):
        # set up fg
        self.tb.run()
        # check data
        n = 1024
        src_data = np.random.rand(n) + np.random.rand(n) * 1j
        stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex * 1, 512)
        src = blocks.vector_source_c(src_data)
        classes = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'WBFM', 'AM-DSB', 'AM-SSB']
        amr = amr_pytorch(False, '/home/lie/gnuradio/grc3.8/cnn11.pth', 512, classes, False)
        snk = blocks.vector_sink_i(1, 1)
        self.tb.connect(src, stream_to_vector, amr, snk)
        self.tb.run()
        print(snk.data())


if __name__ == '__main__':
    gr_unittest.run(qa_amr_pytorch)

