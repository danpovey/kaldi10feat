
import math
import numpy as np

"""
This module contains functions for windowing.  The main user-level functions
are window_size_in_samples(), povey_window() and extract_windows()
"""

def window_size_in_samples(sampling_rate, window_size_ms=25):
    """ Work out the window size in samples, given the signal sampling
    rate and the frame length in milliseconds.   Analogous to Kaldi10's
    FrameExtractionOptions::WindowSize() in feat/feature-window.h

    Args:
        sampling_rate:  The sampling rate/frequency in Hz.  E.g. 16000.  May
                        be int or float.
        window_size_ms:  The frame length in milliseconds.  May be int or
                        float.
    Returns:
        The number of samples in the window.
    """
    return math.floor(sampling_rate * 0.001 * window_size_ms)


def frame_shift_in_samples(sampling_rate, frame_shift_ms=10):
    """Work out the frame shift in samples, given the signal sampling rate and the
    frame shift in milliseconds.  Analogous to Kaldi10's
    FrameExtractionOptions::WindowShift().  Code-wise the same as
    window_size_in_samples(), see docs there.
    """
    return math.floor(sampling_rate * 0.001 * frame_shift_ms)


def round_up_to_power_of_two(n):
    """Rounds up the arg to a power of two (or zero if n was zero).
    This is done very inefficiently."""
    assert isinstance(n, int)
    while n & (n-1) != 0:   ## '&' is bitwise and
        n = n + 1
    return n


def first_sample_of_frame(frame,
                          frame_shift_in_samples,
                          window_size_in_samples):
    """
    Returns the sample-index of the first sample of the frame with index
    'frame'.  Caution: this may be negative; we treat out-of-range samples
    as zero.
    Analogous to with kaldi10's FirstSampleOfFrame in feat/feature-window.h

    Args:
        frame (int): The frame index >= 0.
        frame_shift_in_samples (int) The frame shift in samples
        window_size_in_samples (int)  The window size in samples
    Returns:
        int: The first sample of this frame (caution: may be negative).
    """
    midpoint_of_frame = frame_shift_in_samples * frame + (frame_shift_in_samples // 2)
    beginning_of_frame = midpoint_of_frame - window_size_in_samples // 2
    assert isinstance(beginning_of_frame, int)  # indirectly check inputs were
                                                # int.
    return beginning_of_frame

def get_num_frames(num_samples,
                   frame_shift_in_samples,
                   window_size_in_samples,
                   flush = True):
    """
    Returns the number of frames we'll extract from a signal of specified length
    and specified frame shift and window size.  (The window size only makes a difference
    if flush == false, which is unusual.)
    Analogous to Kaldi10's NumFrames() in feat/feature-window.h

    Args:
        num_samples (int)  Number of samples in the signal
        frame_shift_in_samples (int)  Frame shift between samples
        window_size_in_samples (int)  Length of window
        flush (bool)          True in the normal case when we are processing the
                     whole signal at once.  May be set to false to suppress final
                     samples that overlap the signal boundary.
    Returns:
        int: The number of frames
    """
    assert(isinstance(num_samples, int) and isinstance(frame_shift_in_samples, int) and
           isinstance(window_size_in_samples, int) and isinstance(flush, bool))

    num_frames = (num_samples + (frame_shift_in_samples // 2)) // frame_shift_in_samples
    if not isinstance(num_frames, int):
        raise ValueError("expected integer num_frames")
    if flush:
        return num_frames
    else:
        end_sample_of_last_frame = (first_sample_of_frame(num_frames - 1,
                                                         frame_shift_in_samples,
                                                         window_size_in_samples) +
                                    window_size_in_samples)
        while num_frames > 0 and end_sample_of_last_frame > num_samples:
            num_frames -= 1
            end_sample_of_last_frame -= frame_shift_in_samples
        return num_frames



#sampling_rate, window_size_ms=25):
def povey_window(window_size_in_samples):
    """
    Returns the 'Povey' window function.  This looks from a distance like the
    Hamming window, except it smoothly goes to zero at the edges.  This means
    that the impulse response has much less of a high-frequency component,
    so we don't have to worry so much about low frequencies bleeding into
    high frequences (and can, say, ditch pre-emphasis without as much
    concern as if we had used the Hamming window).  It's the default in kaldi
    and kaldi10 feature processing.

    Args:
        window_size_in_samples:  The number of samples in the window function
    Returns:
        The window function, as a numpy array of shape (window_size,) and
       dtype float32.  The elements will be in the range [0,1]
    """
    assert isinstance(window_size_in_samples, int) and window_size_in_samples > 1
    ans = np.empty(shape=(window_size_in_samples,), dtype=np.float32)
    a = math.pi * 2.0 / (window_size_in_samples - 1)
    for i in range(window_size_in_samples):
        ans[i] = math.pow(0.5 - 0.5*math.cos(a * i), 0.85)
    return ans


def extract_windows(signal,
                    window,
                    frame_shift_in_samples,
                    flush = True,
                    round_to_power_of_two = True):
    """Extract a sequence of windowed frames from a provided signal and return it
    as a single numpy array.

    Args:
        signal:   The input signal.  Must be a numpy array with signal.ndim = 1.
                 The dtype must be int16 or a floating type.  If it is int16, the signal
                 will be scaled by 1.0 / 32768 in the result.
        window:   The windowing function, e.g. as returned by povey_window().
                 A numpy array with shape (window_size_in_samples,) and dtype
                 float32.
        frame_shift_in_samples (int):  The desired frame-shift, measured in
                 samples.
        flush:   Affects end-of-signal behavior.  True in the normal case when
                 we are processing the whole signal at once.  May be set to
                 false to suppress final samples that overlap the signal boundary.
        round_to_power_of_two:  If true, the returned matrix will have
                 the window length rounded up to a power of two, as if the
                 window were padded with zeros.

    Returns:
        If the number of frames was nonempty, returns a numpy.ndarray with dtype
        float32 and shape=(num_frames, padded_window_size), where
        padded_window_size is len(window) if round_to_power_of_two is false,
        otherwise round_up_to_power_of_two(len(window)).

        If the number of frames was zero, None will be returned.
    """
    assert (isinstance(signal, np.ndarray) and signal.ndim == 1 and
            (signal.dtype in [ np.int16, np.float32, np.float64 ]))
    assert (isinstance(window, np.ndarray) and window.ndim == 1 and
            (signal.dtype in [ np.int16, np.float32, np.float64 ]))
    assert isinstance(frame_shift_in_samples, int)
    assert isinstance(flush, bool)
    assert isinstance(round_to_power_of_two, bool)

    window_size = window.shape[0]
    num_samples = signal.shape[0]
    num_frames = get_num_frames(num_samples, frame_shift_in_samples,
                                window_size, flush)
    if num_frames == 0:
        return None

    padded_window_size = (round_up_to_power_of_two(window_size)
                          if round_to_power_of_two else
                          window_size)

    ans = np.empty((num_frames, padded_window_size), dtype=np.float32)

    if padded_window_size > window_size:
        ans[:,window_size:padded_window_size].fill(0)


    # num_edge_frames is a big overestimate of how many frames at each
    # side might have to be treated as edge cases
    num_edge_frames = window_size // frame_shift_in_samples

    # The following loop only handles frames near the edge, inefficiently.
    for frame in list(range(0, num_edge_frames)) + list(range(num_frames - num_edge_frames, num_frames)):
        if frame < 0 or frame >= num_frames:
            continue
        first_sample = first_sample_of_frame(frame,
                                             frame_shift_in_samples,
                                             window_size)
        end_sample = first_sample + window_size

        first_sample_truncated = max(0, first_sample)
        end_sample_truncated = min(end_sample, num_samples)

        offset1 = first_sample_truncated - first_sample
        offset2 = end_sample_truncated - first_sample

        # the following statement would only make a difference in
        # super-pathological cases where the num-samples is less than a frame's
        # worth.
        ans[frame,:].fill(0)

        # we must have first_sample_truncated < end_sample_truncated since
        # we checked that the signal is nonempty.
        ans[frame, offset1:offset2] = signal[first_sample_truncated:end_sample_truncated]

        # For edge effects, we use reflection.
        if offset1 > 0:
            ans[frame, offset1-1::-1] = ans[frame, offset1:2*offset1]
        if offset2 < window_size:
            k = window_size - offset2
            ans[frame, window_size-1:offset2-1:-1] = ans[frame, offset2-k:offset2]


    # The loop below will handle frames in range(num_edge_frames, num_frames -
    # num_edge_frames), using as large matrices as possible via views.
    first_sample = first_sample_of_frame(num_edge_frames,
                                         frame_shift_in_samples,
                                         window_size)
    num_frames_reduced = num_frames - (2 * num_edge_frames)
    sample_piece_length = (frame_shift_in_samples * num_frames_reduced)

    end_sample = first_sample + (frame_shift_in_samples * num_frames_reduced)

    for start_offset in range(0, window_size, frame_shift_in_samples):
        end_offset = min(start_offset + frame_shift_in_samples, window_size)
        num_cols = end_offset - start_offset

        signal_start = first_sample + start_offset
        signal_end = signal_start + (frame_shift_in_samples * num_frames_reduced)


        signal_reshaped = signal[signal_start : signal_end].reshape(num_frames_reduced,
                                                                    frame_shift_in_samples)


        ans[num_edge_frames : num_frames-num_edge_frames,
                   start_offset : end_offset] = signal_reshaped[:, 0:num_cols]


    # Remove the DC offset from each window (prior to applying the window
    # function).  This is the --remove-dc-offset option in Kaldi, which defaults
    # to true.
    ans[:,0:window_size] -= ans[:,0:window_size].sum(axis=1, keepdims=True) * (1.0 / window_size)

    ans[:,0:window_size] *= window * ((1.0/32768) if signal.dtype == np.int16 else 1.0)
    return ans
