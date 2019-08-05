import unittest
import os
import numpy as np
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from kaldi10feat.read_wave import *
# We'll also test computation of mel features on this .
from kaldi10feat.mel import *

class TestReadWave(unittest.TestCase):

    def test_read_wave(self):
        (samprate, data) = read_wave_file("temp.wav")
        mel_computer = MelFeatureComputer(samprate)
        # use channel 0 only.
        feats = mel_computer.compute(data[0,:])
        print("Feats are: {} {}".format(feats.shape, feats))



if __name__ == "__main__":
    unittest.main()

