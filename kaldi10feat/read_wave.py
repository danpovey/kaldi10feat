import sys
import numpy as np
import wave

"""
This module provides a convenient, simplified interface to the "wave" package,
which allows you to read a 16-bit uncompressed wave file in the form of a numpy
array.  This allows wave files to be read without huge or hard-to-install extra
dependencies like scipy or soundfile
"""


def read_wave_file(file, data_min_proportion=1.0):
    """
    Reads a wave file and returns it as a NumPy array.

    Args:
       file:  Either a filename or a file object

    Returns a 2-tuple of:
         (samprate, data)
    where samprate is the sampling rate as in integer (e.g. 16000), and `data`
    is a numpy array with dtype int16 and shape (num_channels, num_samples).


    Raises:
      RuntimeError: if an error occurred while reading the data.
                 (Note: if more than `data_min_proportion` of the
                 expected data was read, it will succeed even if
                 the file was truncated.)
      wave.Error: whatever errors the wave module encountered
      OsError (via wave module), if a file could not be opened.
    """

    wave_reader = wave.Wave_read(file)
    (nchannels, sampwidth, framerate,
     nframes, comptype, compname) = wave_reader.getparams()

    if comptype != 'NONE':
        raise RuntimeError("Wave file has compression, which is unsupported: comptype={},"
                           "compname={}".format(comptype, compname))
    if sampwidth != 2:
        raise RuntimeError("Wave file has sample width of {}, expected 2.".format(
                sampwidth))

    data_as_bytes = wave_reader.readframes(nframes)
    nframes_read = len(data_as_bytes) // (sampwidth * nchannels)

    assert nframes_read  <= nframes
    if nframes_read < data_min_proportion * nframes:
        raise RuntimeError("Reading data from {0}, read too little data: {1} != {2} "
                           "(min allowed proportion: {3})".format(
                file, nframes_read, nframes, dat_min_proportion))

    dt = np.dtype('int16')
    if sys.byteorder == 'big':
        # Make sure to interpret the data as little-endian even if the machine
        # is big endian.
        dt = dt.newbyteorder('<')

    array = np.frombuffer(data_as_bytes, dt)
    # order='F' because the frame has a higher stride than the channel.
    return (framerate, array.reshape((nchannels, nframes_read), order='F'))

