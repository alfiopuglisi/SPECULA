#+
# NAME:
#   modal_pushpull_signal
# PURPOSE:
#   generate a modal push-pull time history to be use for calibration purpose
# CATEGORY:
#   AO simulation.
# CALLING SEQUENCE:
# function modal_pushpull_signal, n_modes, amplitude=amplitude, vect_amplitude=vect_amplitude
# INPUTS:
#   n_modes         number of modes
# KEYWORD:
#   amplitude	    amplitude of mode 0
#   vect_amplitude  modal amplitude vector
#   linear          vect_amplitude change as 1/rad_order instead of 1/sqrt(radorder)
#   constant        vect_amplitude is constant across modes
#   min_amplitude   min value for vect_amplitude
#   only_push       makes an only push signal
#   pattern         push_pull pattern, default [-1, 1], can be any sequence of numbers
#   ncycles         number of cycle of push-pull
#   repeat_ncycles   set it to have ncycles of push and then ncycles of pull
#   nsamples        how many samples to hold in each position, default=1
# OUTPUTS:
#   time_hist       modal time history
# COMMON BLOCKS:
#   None.
# SIDE EFFECTS:
#   None.
# RESTRICTIONS:
#   None
# MODIFICATION HISTORY:
#   Created 12-SEP-2014 by Guido Agapito guido.agapito@inaf.it
#-
import numpy as np
from specula.lib.zernike_generator import ZernikeGenerator

def modal_pushpull_signal(n_modes, amplitude=None, vect_amplitude=None, constant=False, first_mode=0,
                                linear=None, min_amplitude=None, only_push=False,
                                pattern=[1, -1],
                                ncycles=1, repeat_ncycles=False, nsamples=1, xp=np):

    if only_push:
        pattern = [1]

    if vect_amplitude is None:
        radorder = xp.array([ZernikeGenerator.degree(x)[0] for x in xp.arange(first_mode, n_modes) + 2])
        if linear:
            vect_amplitude = amplitude/radorder
        elif constant:
            vect_amplitude = np.repeat(amplitude, len(radorder))
        else:
            vect_amplitude = amplitude/np.sqrt(radorder)
        if min_amplitude is not None:
            vect_amplitude = xp.minimum(vect_amplitude, min_amplitude)

    # Prepend zero values equal to the number of skipped modes
    vect_amplitude = np.hstack((
        np.repeat(0, first_mode), vect_amplitude
    ))

    n_pokes = len(pattern)

    real_n_modes = n_modes - first_mode
    time_hist = xp.zeros((n_pokes * real_n_modes * ncycles, n_modes))
    for mode in range(first_mode, n_modes):
        hist_idx = mode - first_mode
        poke_pattern = vect_amplitude[mode] * xp.array(pattern)
        if repeat_ncycles:
            time_hist[n_pokes*hist_idx*ncycles:n_pokes*(hist_idx+1)*ncycles, mode] = \
                xp.repeat(poke_pattern, ncycles)
        else:
            for j in range(ncycles):
                time_hist[n_pokes*(ncycles*hist_idx+j):n_pokes*(ncycles*hist_idx+j+1), mode] = poke_pattern

    return np.repeat(time_hist, nsamples, axis=0)
