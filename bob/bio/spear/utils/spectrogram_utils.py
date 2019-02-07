#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Seyyed Saeed Sarfjoo <ssarfjoo@idiap.ch>
# Thr 7 Feb 17:21:35 CET 2019
#
# Copyright (C) 2012-2019 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import os
import math
import numpy
import scipy.io.wavfile
import sys

from numpy.fft import fft
import logging

logger = logging.getLogger("bob.bio.spear")

def spectrogram_computation(rate, data, win_length_ms=25, win_shift_ms=10, n_filters=24,
                            f_min=0., f_max=4000., pre_emphasis_coef=0.97, mel_scale=True):
  """ Compute spectrogram from the input signal

  Parameters
  ----------
  rate: int32
    Sampling rate of input signal
  data: 1D :py:class:`numpy.ndarray` (floats)
    The input audio signal.
  win_length_ms: int32
    Window length in msec (Default: 25).
  win_shift_ms: int32
    Window shift in msec (Default: 10).
  n_filters: int32
    Number of filters for computing the filter bank (Default: 24).
  f_min: float
    Minimum frequency in band pass filter (Default: 0.)
  f_max: float
    Maximum frequency in band pass filter (Default: 4000.)
  pre_emphasis_coef: float
    Pre emphasis coeficient (Default: 0.97)
  mel_scale: bool
    Compute features in mel scale (Default: True)

  Returns
  -------
  features: 2D :py:class:`numpy.ndarray` (floats)
    Spectrogram from the input signal
  """
  win_length = int (rate * win_length_ms / 1000)
  win_shift = int (rate * win_shift_ms / 1000)
  win_size = int (2.0 ** math.ceil(math.log(win_length) / math.log(2)))
  m = int (math.log(win_size) / math.log(2))

  # Hamming initialisation
  hamming_kernel = init_hamming_kernel(win_length)

  # Compute cut-off frequencies
  p_index = init_freqfilter(rate, win_size,  mel_scale, n_filters, f_min, f_max)

  data_size = data.shape[0]
  n_frames = int(1 + (data_size - win_length) / win_shift)

  # create features set
  try:
      features = numpy.zeros([n_frames, int(win_size/2)+1], dtype=numpy.float64)
  except ValueError:
      logger.error("The input data shape is not suitable! " +  str(data.shape))

  last_frame_elem = 0
  # compute cepstral coefficients
  for i in range(n_frames):
    # create a frame
    frame = numpy.zeros(win_size, dtype=numpy.float64)
    vec = numpy.arange(win_length)
    frame[vec] = data[vec + i * win_shift]
    som = numpy.sum(frame)
    som = som / win_size
    frame[vec] -= som  # normalization by mean here

    frame_, last_frame_elem = pre_emphasis(frame[vec], win_shift, pre_emphasis_coef, last_frame_elem)
    frame[vec] = frame_

    # Hamming windowing
    frame = hamming_window(frame, hamming_kernel, win_length)

    filters, spec_row = log_filter_bank(frame, n_filters, p_index, win_size)

    features[i] = spec_row[0:int(win_size/2)+1]

  return numpy.array(features)

def init_hamming_kernel(win_length):
  """ Hamming initialisation

  Parameters
  ----------
  win_length: int32
    The window length for computing the Hamming

  Returns
  -------
  hamming_kernel : 1D :py:class:`numpy.ndarray` (floats)
    The Hamming kernel values
  """
  cst = 2 * math.pi / (win_length - 1.0)
  hamming_kernel = numpy.zeros(win_length)

  for i in range(win_length):
    hamming_kernel[i] = (0.54 - 0.46 * math.cos(i * cst))
  return hamming_kernel

def init_freqfilter(rate, win_size,  mel_scale, n_filters, f_min, f_max):
  """ Compute cut-off frequencies

  Parameters
  ----------
  rate: int32
    Rate for computing cut-off frequencies
  win_size: int32
    Window size for computing the frequencies
  mel_scale: bool
    Use mel scale for computing the frequencies
  n_filters: int32
    Number if filters in FFT
  f_min: int32
    Minimum frequency
  f_max: int32
    Maximum frequency 

  Returns
  -------
  p_index : 1D :py:class:`numpy.ndarray` (floats)
    The frequency cut-off values
  """
  p_index = numpy.array(numpy.zeros(n_filters + 2), dtype=numpy.float64)
  if (mel_scale):
    # Mel scale
    m_max = mel_python(f_max)
    m_min = mel_python(f_min)

    for i in range(n_filters + 2):
      alpha = float(i) / (n_filters+1)
      f = mel_inv_python(m_min * (1 - alpha) + m_max * alpha)
      factor = float(f) / rate
      p_index[i] = win_size * factor
  else:
    # linear scale
    for i in range(n_filters + 2):
      alpha = float(i) / (n_filters+1)
      f = f_min * (1.0 - alpha) + f_max * alpha
      p_index[i] = float(win_size) / rate * f
  return p_index

def init_dct_kernel(n_filters, n_ceps, dct_norm):
  """ Inittialize DCT kernel

  Parameters
  ----------
  n_filters: int32
    Number if filters in FFT
  n_ceps: int32
    Number if Cepstrums in FFT
  dct_norm: bool
    Use normilized DCT 

  Returns
  -------
  dct_kernel : 2D :py:class:`numpy.ndarray` (floats)
    The DCT Kernel matrix
  """
  dct_kernel = numpy.zeros([n_ceps, n_filters], dtype=numpy.float64)

  dct_coeff = 1.0
  if dct_norm:
    dct_coeff = math.sqrt(2.0/n_filters)

  for i in range(0, n_ceps):
    for j in range(0, n_filters ):
      dct_kernel[i][j] = dct_coeff * math.cos(math.pi * i * (j + 0.5) / float(n_filters))

  if dct_norm:
    column_multiplier = numpy.ones(n_ceps, dtype=numpy.float64)
    column_multiplier[0] = math.sqrt(0.5)  # first element sqrt(0.5), the rest are 1.
    for j in range(0, n_filters):
      dct_kernel[:, j] = column_multiplier * dct_kernel[:, j]

  return dct_kernel

def read(filename):
  """ Read audio.FrameContainer containing preprocessed frames

  Parameters
  ----------
  filename: str
    The address of audio file to read

  Returns
  -------
  rate: int32
    Sampling rate of the audio file
  data: 1D :py:class:`numpy.ndarray` (floats)
    Raw audio array
  """

  fileName, fileExtension = os.path.splitext(filename)
  wav_filename = filename
  rate, data = scipy.io.wavfile.read(str(wav_filename)) # the data is read in its native format
  if data.dtype =='int16':
    data = numpy.cast['float'](data)
  return [rate,data]

def compare(v1, v2, width):
  """ Compare the abstract of two values 

  Parameters
  ----------
  v1: float
    The first parameter
  v2: float
    The second parameter
  width: float
    The margin for computing the difference

  Returns
  -------
  ans: bool
    The comparison result
  """
  return abs(v1-v2) <= width

def mel_python(f):
  """ Return the normalized frequency in mel scale 

  Parameters
  ----------
  f: float
    The input frequency

  Returns
  -------
  ans: float
    The normalized frequency in mel scale 
  """
  return 2595.0*math.log10(1.+f/700.0)

def mel_inv_python(value):
  """ Return the normalized frequency in inverse mel scale 

  Parameters
  ----------
  value: float
    The input frequency

  Returns
  -------
  ans: float
    The normalized frequency in inverse mel scale 
  """
  return 700.0 * (10 ** (value / 2595.0) - 1)

def sig_norm(win_length, frame, flag):
  """ Normalize the signal based on energy floor or gain

  Parameters
  ----------
  win_length: int32
    The window length for computing the gain
  frame: 1D :py:class:`numpy.ndarray` (floats)
    The input signal
  flag: bool
    Flag for doing normalization

  Returns
  -------
  gain: float
    The gain of input signal
  """
  gain = 0.0
  for i in range(win_length):
    gain = gain + frame[i] * frame[i]

  ENERGY_FLOOR = 1.0
  if gain < ENERGY_FLOOR:
    gain = math.log(ENERGY_FLOOR)
  else:
    gain = math.log(gain)

  if(flag and gain != 0.0):
    for i in range(win_length):
      frame[i] = frame[i] / gain
  return gain

def pre_emphasis(frame, win_shift, coef, last_frame_elem):
  """ Pre emphasis the input signal

  Parameters
  ----------
  frame: 1D :py:class:`numpy.ndarray` (floats)
    The input signal
  win_shift: int32
    The window shift for framing
  coef: float
    Emphasis coeficient
  last_frame_elem: float
    Last frame element 

  Returns
  -------
  ans: 1D :py:class:`numpy.ndarray` (floats)
    Pre emphasised signal
  """
  if (coef <= 0.0) or (coef > 1.0):
    logger.error("Error: The emphasis coeff. should be between 0 and 1")
    return None

  last_element = frame[win_shift - 1]
  return numpy.append(frame[0]-coef * last_frame_elem, frame[1:]-coef*frame[:-1]), last_element

def hamming_window(vector, hamming_kernel, win_length):
  """ Applying hamming window on signal

  Parameters
  ----------
  vector: 1D :py:class:`numpy.ndarray` (floats)
    The input signal
  hamming_kernel : 1D :py:class:`numpy.ndarray` (floats)
    The Hamming kernel values
  win_length: int32
    The window length

  Returns
  -------
  vector: 1D :py:class:`numpy.ndarray` (floats)
    The input signal with hamming 
  """  
  for i in range(win_length):
    vector[i] = vector[i] * hamming_kernel[i]
  return vector

def log_filter_bank(frame, n_filters, p_index, win_size):
  """ Compute log filter bank

  Parameters
  ----------
  frame: 1D :py:class:`numpy.ndarray` (floats)
    The input frame signal
  n_filters: int32
    Number of filters in filter bank
  p_index : 1D :py:class:`numpy.ndarray` (floats)
    The frequency cut-off values
  win_size: int32
    Window size for computing the frequencies

  Returns
  -------
  filters: 1D :py:class:`numpy.ndarray` (floats)
    Log triangular filter banks based on n_filters
  frame: 1D :py:class:`numpy.ndarray` (floats)
    Magnitude of FFT transform for the input frame signal 
  """
  x1 = numpy.array(frame, dtype=numpy.complex128)
  complex_ = fft(x1)
  abscomplex = numpy.absolute(complex_)
  frame[0:int(win_size / 2) + 1] = abscomplex[0:int(win_size / 2) + 1]

  filters = log_triangular_bank(frame, n_filters, p_index)
  return filters, frame

def log_triangular_bank(data, n_filters, p_index):
  """ Compute log triangular filter banks based on n_filters

  Parameters
  ----------
  data: 1D :py:class:`numpy.ndarray` (floats)
    Magnitude of FFT transform for the input frame signal 
  n_filters: int32
    Number of filters in filter bank
  p_index : 1D :py:class:`numpy.ndarray` (floats)
    The frequency cut-off values

  Returns
  -------
  ans: 1D :py:class:`numpy.ndarray` (floats)
    Log triangular filter banks based on n_filters
  """
  res_ = numpy.zeros(n_filters, dtype=numpy.float64)

  denominator = 1.0 / (p_index[1:n_filters+2] - p_index[0:n_filters+1])

  for i in range(0, n_filters):
    li = int(math.floor(p_index[i] + 1))
    mi = int(math.floor(p_index[i+1]))
    ri = int(math.floor(p_index[i+2]))
    if i == 0 or li == ri:
      li -= 1

    vec_left = numpy.arange(li, mi+1)
    vec_right = numpy.arange(mi+1, ri+1)
    res_[i] = numpy.sum(data[vec_left] * denominator[i] * (vec_left-p_index[i])) + \
              numpy.sum(data[vec_right] * denominator[i+1] * (p_index[i+2]-vec_right))

  FBANK_OUT_FLOOR = sys.float_info.epsilon
  return numpy.log(numpy.where(res_ < FBANK_OUT_FLOOR, FBANK_OUT_FLOOR, res_))

def dct_transform(filters, n_filters, dct_kernel, n_ceps):
  """ Apply DCT transform on the log triangular filter banks

  Parameters
  ----------
  filters: 1D :py:class:`numpy.ndarray` (floats)
    Log triangular filter banks 
  n_filters: int32
    Number of filters in filter bank
  dct_kernel : 1D :py:class:`numpy.ndarray` (floats)
    DCT kernels for computing the cepstrums
  n_ceps: int32
    Number of output cepstrums

  Returns
  -------
  ceps: 1D :py:class:`numpy.ndarray` (floats)
    Output cepstrums after applying the DCT
  """
  ceps = numpy.zeros(n_ceps)
  vec = numpy.array(range(0, n_filters))
  for i in range(0, n_ceps):
    ceps[i] = numpy.sum(filters[vec] * dct_kernel[i])

  return ceps
