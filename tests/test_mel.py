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


if __name__ == "__main__":
    unittest.main()
