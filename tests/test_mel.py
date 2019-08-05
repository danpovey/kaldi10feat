import unittest
import os
import numpy as np
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from kaldi10feat.mel import *

class TestMel(unittest.TestCase):

    # This was done so I could check that they are the same as printed out
    # by Kaldi binary compute-mfcc-feats.
    def test_mel_simple(self):
        mel = compute_mel_banks(256, 16000, 23)
        self.assertTrue(isinstance(mel, np.ndarray) and
                        mel.dtype == np.float32)
        np.set_printoptions(threshold=sys.maxsize)
        print("{}".format(mel))

    def test_feature_computer(self):
        signal_length = 800
        mel_computer = MelFeatureComputer(16000)

        for dtype in ['float32', 'float64', 'int16']:
            signal = (100.0 * np.random.randn(signal_length).astype(dtype))
            foo = mel_computer.compute(signal)
            print("dtype={} log-mel={}".format(signal, foo))

        signal = (100.0 * np.random.randn(signal_length)).astype('int16')
        sum1 = mel_computer.compute(signal).sum()
        sum2 = mel_computer.compute(signal.astype('float32')).sum()
        print("{} and {} should differ (checking for int16 scaling)".format(
                sum1, sum2))


if __name__ == "__main__":
    unittest.main()

