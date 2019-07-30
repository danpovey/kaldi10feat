
#### Kaldi10 -- basic speech recognition feature processing compatible with kaldi10.

Note: as of July 2019, kaldi10 is a not-yet-released version of Kaldi with
limited back compatibility with previous versions of Kaldi, and various
simplifications.

Some differences with previous versions of Kaldi that affect the features
include: removal of preemphasis, and a different convention for determining the
number of frames (it no longer depends on the frame width, only on the frame
shift).
