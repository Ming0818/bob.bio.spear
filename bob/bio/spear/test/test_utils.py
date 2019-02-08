#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Seyyed Saeed Sarfjoo <ssarfjoo@idiap.ch>
# @date: Thr  8 Feb 10:10:43 CET 2019
#
# Copyright (C) 2012-2019 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''Runs a specific user program'''


import sys
import numpy
import pkg_resources

import bob.bio.base
import bob.bio.spear
from bob.bio.spear.utils.spectrogram_utils import spectrogram_computation, read



def test_spectrogram(filename='data/sample.wav'):
  """ Test spectrogram computation
  
  Parameters
  ----------
  filename: str
    Input file name for computing the spectrogram.
    
  """
  path = pkg_resources.resource_filename('bob.bio.spear.test', filename)
  rate,data = read(path)
  spec_feat = spectrogram_computation(rate, data)
  assert numpy.sum(spec_feat) > 1e+7, "Can not compute the spectrogram of " + path
