import unittest
import os
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from kaldi10feat.window import *

class TestWindow(unittest.TestCase):

    def test_window_size(self):
        self.assertTrue(window_size_in_samples(8000, 25) == 200)
        self.assertTrue(window_size_in_samples(8000, 25.01) == 200)

    def test_frame_shift(self):
        self.assertTrue(frame_shift_in_samples(8000, 25) == 200)

    def test_round_up_to_power_of_two(self):
        self.assertTrue(round_up_to_power_of_two(1) == 1)
        self.assertTrue(round_up_to_power_of_two(2) == 2)
        self.assertTrue(round_up_to_power_of_two(3) == 4)
        self.assertTrue(round_up_to_power_of_two(200) == 256)

    def test_first_sample_of_frame(self):
        self.assertTrue(first_sample_of_frame(0, 100, 100) == 0)
        self.assertTrue(first_sample_of_frame(2, 100, 100) == 200)
        self.assertTrue(first_sample_of_frame(1, 100, 300) == 0)


    def test_get_num_frames(self):
        self.assertTrue(get_num_frames(500, 100, 100) == 5)
        self.assertTrue(get_num_frames(599, 100, 100) == 6)
        self.assertTrue(get_num_frames(549, 100, 1000) == 5)
        self.assertTrue(get_num_frames(549, 100, 200, False) < 5)

    def test_povey_window(self):
        a = povey_window(3)
        self.assertTrue(a.dtype == 'float32')
        self.assertTrue(a[0] == 0 and a[1] == 1 and a[2] == 0 and len(a) == 3)

    def test_extract_windows1(self):
        window_func = np.ones((10,), dtype=np.float32)
        samples = np.random.normal(size=(110,)).astype(np.float32)
        w = extract_windows(samples, window_func,
                            frame_shift_in_samples=10,
                            round_to_power_of_two=False)
        self.assertTrue(w.shape == (11, 10))


        print("W is {}".format(w))
        print("samples is {}".format(samples))
        self.assertTrue(np.array_equal(w.reshape((110,)), samples))

    def test_extract_windows2(self):
        window_func = np.ones((10,), dtype=np.float32)
        window_func *= 0.5
        samples = np.random.normal(size=(110,)).astype(np.float32)
        w = extract_windows(samples, window_func,
                            frame_shift_in_samples=10,
                            round_to_power_of_two=False)
        self.assertTrue(w.shape == (11, 10))


        print("W is {}".format(w))
        w *= 2
        print("samples is {}".format(samples))
        self.assertTrue(np.array_equal(w.reshape((110,)), samples))

    def test_extract_windows3(self):
        window_func = povey_window(20)
        #window_func = np.ones((30,), dtype=np.float32)
        samples = np.random.normal(size=(220,)).astype(np.float32)
        w = extract_windows(samples, window_func,
                            frame_shift_in_samples=20)

        print("W is {}".format(w))
        print("samples is {}".format(samples))


if __name__ == "__main__":
    unittest.main()
