
import math
import numpy as np

"""
This module contains functions for mel-scale computations and
mel-scale filterbank extraction
"""


def hz_to_mel(f):
    """ Converts a frequency in Hz to a mel-scale frequency"""
    return 1127.0 * math.log(1.0 + f / 700.0)

def mel_to_hz(f):
    """ Converts a mel-scale frequency to a frequency in Hz"""
    return 700.0 * (math.exp(f / 1127.0 - 1.0))


def compute_mel_banks(num_fft_bins,
                      samp_freq,
                      num_mel_bins,
                      low_freq = 20.0,
                      high_freq = 0.0):
    """Returns mel-scale filterbanks as a numpy ndarray of shape
    (num_mel_bins, num_fft_bins) and dtype float32.  This will be used
    to sum fft-bin energies.

    Args:
       num_fft_bins (int): The number of FFT bins.  Will be equal to half
          the padded window size (i.e. the window size rounded up to
           a power of two
       samp_freq (int or float):  The signal sampling frequency, e.g. 16000
       num_mel_bins (int): The desired number of mel bins, e.g. 40.
       low_freq (float)  The frequency in Hz below which we won't
               extract information
       high_freq (float)  The frequency in Hz below which we won't
                extract information; if not positive, it is interpreted as an
                offset from the Nyquist frequency (== samp_freq / 2).

    Returns:
       Returns a numpy.ndarray of shape (num_mel_bins * num_fft_bins),
    dtype=np.float32 and elements in the interval [0,1]
    """
    assert (num_fft_bins > 0 and samp_freq > 0 and num_mel_bins >= 3 and
            low_freq >= 0 and isinstance(num_mel_bins, int) and
            isinstance(num_fft_bins, int))
    nyquist = 0.5 * samp_freq
    if high_freq  <= 0:
        high_freq += nyquist
    assert high_freq > 0 and high_freq <= samp_freq
    assert low_freq >= 0 and low_freq < nyquist and low_freq < high_freq
    fft_bin_width = nyquist / num_fft_bins
    mel_low_freq = hz_to_mel(low_freq)
    mel_high_freq = hz_to_mel(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_mel_bins + 1)
    ans = np.zeros((num_mel_bins, num_fft_bins), dtype=np.float32)
    for b in range(num_mel_bins):
        left_mel = mel_low_freq + b * mel_freq_delta
        center_mel = left_mel + mel_freq_delta
        right_mel = center_mel + mel_freq_delta
        for i in range(num_fft_bins):
            freq = i * fft_bin_width
            mel = hz_to_mel(freq)
            if mel > left_mel and mel < right_mel:
                weight = ((mel - left_mel) / (center_mel - left_mel) if
                          mel <= center_mel else
                          (right_mel-mel) / (right_mel-center_mel))
                ans[b, i] = weight
    return ans