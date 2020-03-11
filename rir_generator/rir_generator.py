from typing import Optional, Tuple, Union

import numpy

from .c_ext import generate_rir_ext


def generate_rir(
    r: numpy.ndarray, s: numpy.ndarray, l: numpy.ndarray,
    beta: Union[numpy.ndarray, float], c: float = 340, fs: float = 16000,
    n_sample: int = None, mtype: str = 'omnidirectional', order: int = -1,
    dim: int = 3, orientation: numpy.ndarray = None,
    hp_filter: bool = True) -> Tuple[numpy.ndarray, Optional[float]]:
    """Computes the response of an acoustic source to one or more microphones
    in a reverberant room using the image method [1,2].

    [1] J.B. Allen and D.A. Berkley,
        Image method for efficiently simulating small-room acoustics,
        Journal Acoustic Society of America,
        65(4), April 1979, p 943.

    [2] P.M. Peterson,
        Simulating the response of multiple microphones to a single
        acoustic source in a reverberant room, Journal Acoustic
        Society of America, 80(5), November 1986

    Args:
        r: array of shape (M, 3) specifying the (x, y, z) coordinates of
            the receiver(s) in m.
        s: array of shape (3,) specifying the (x,y,z) coordinates of
            the source in m.
        l: array of shape (3,) specifying the room dimensions (x,y,z) in m.
        beta: array of shape (6,) specifying the reflection coefficients
            [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2] or
            beta = reverberation time (T_60) in seconds.
        c: sound velocity in m/s.
        fs: sampling frequency in Hz.
        nsample: number of samples to calculate, default is T_60*fs.
        mtype: [omnidirectional, subcardioid, cardioid, hypercardioid,
            bidirectional], default is omnidirectional.
        order: reflection order, default is -1, i.e. maximum order.
        dim: room dimension (2 or 3), default is 3.
        orientation: direction in which the microphones are pointed, specified
            using azimuth and elevation angles (in radians), default is [0 0].
        hp_filter: use 'false' to disable high-pass filter, the high-pass
            filter is enabled by default.
    Returns:
        numpy.ndarray: array of shape (M, nsample) containing the calculated
            room impulse response(s).
        Option[float]: in case a reverberation time is specified as an input
            parameter the corresponding reflection coefficient is returned.
    """
    l = numpy.asfortranarray(l)
    if isinstance(beta, float):
        beta = numpy.array(beta)
    if n_sample is None:
        n_sample = -1
    if order is None:
        order = -1
    if mtype not in ('omnidirectional', 'subcardioid', 'cardioid',
                     'hypercardioid', 'bidirectional'):
        raise ValueError('Invalid microphone type `{}`'.format(mtype))
    if dim not in (2, 3):
        raise ValueError('Invalid room dimension `{}`'.format(dim))
    if orientation is None:
        orientation = numpy.array([0., 0.])
    if isinstance(orientation, float) or isinstance(orientation, int):
        orientation = numpy.array([orientation, 0.])
    return generate_rir_ext(c, fs, r, s, l, beta, n_sample, mtype,
                            order, dim, orientation, hp_filter)
