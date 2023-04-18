import numpy as np


def rampup_timesteps(time, dt, n=8):
    """
    Create timesteps that ramp up geometrically

    Parameters:
    -----------
    time : float
        The total simulation time so that sum(dt) = time
    dt : float
        Target timestep after initial ramp-up
    n : int, optional
        Number of rampup steps. Defaults to 8.

    Returns:
    --------
    dt : numpy.ndarray
        Array of timesteps.

    Note:
    -----
    The final timestep may be shorter than dt in order to exactly reach T.

    Copyright:
    -----------
    A python version of the corresponding routine from MRST, https://www.sintef.no/projectweb/mrst/
    """

    # Initial geometric series
    dt_init = (dt / 2.0 ** np.arange(n, -1, -1))
    cs_time = np.cumsum(dt_init)
    if np.any(cs_time > time):
        dt_init = dt_init[cs_time < time]

    # Remaining time that must be discretized
    dt_left = time - np.sum(dt_init)
    # Even steps
    dt_rem = np.tile(dt, int(np.floor(dt_left / dt)))
    # Final ministep if present
    dt_final = time - np.sum(dt_init) - np.sum(dt_rem)
    # Less than to account for rounding errors leading to a very small
    # negative time-step.
    if dt_final <= 0:
        dt_final = np.array([])
    # Combined timesteps
    dT = np.concatenate((dt_init, dt_rem, np.array([dt_final])))

    return dT