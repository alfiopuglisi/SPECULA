
import numpy as np
from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj
from specula.data_objects.recmat import Recmat

class Intmat(BaseDataObj):
    '''
    An Interaction Matrix is a matrix with shape [n_slopes, n_modes]
    '''
    def __init__(self,
                 intmat,
                 slope_mm: list = None,
                 slope_rms: list = None,
                 pupdata_tag: str = '',
                 subapdata_tag: str = '',
                 norm_factor: float= 0.0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize an :class:`~specula.data_objects.intmat.Intmat` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.intmat = self.to_xp(intmat)
        self.slope_mm = slope_mm
        self.slope_rms = slope_rms
        self.pupdata_tag = pupdata_tag
        self.subapdata_tag = subapdata_tag
        self.norm_factor = norm_factor

    def get_value(self):
        '''
        Get the intmat as a numpy/cupy array
        '''
        return self.intmat

    def set_value(self, v):
        '''
        Set new values for the intmat
        Arrays are not reallocated
        '''
        assert v.shape == self.intmat.shape, \
            f"Error: input array shape {v.shape} does not match intmat shape {self.intmat.shape}"
        self.intmat[:]= self.to_xp(v)

    def reduce_size(self, n_modes_to_be_discarded):
        if n_modes_to_be_discarded >= self.nmodes:
            raise ValueError(f'nModesToBeDiscarded should be less than nmodes (<{self.nmodes})')
        self.intmat = self.intmat[:, :self.nmodes - n_modes_to_be_discarded]

    def reduce_slopes(self, n_slopes_to_be_discarded):
        if n_slopes_to_be_discarded >= self.nslopes:
            raise ValueError(f'nSlopesToBeDiscarded should be less than nslopes (<{self.nslopes})')
        self.intmat = self.intmat[:self.nslopes - n_slopes_to_be_discarded, :]

    def set_start_mode(self, start_mode):
        nmodes = self.intmat.shape[1]
        if start_mode >= nmodes:
            raise ValueError(f'start_mode should be less than nmodes (<{nmodes})')
        self.intmat = self.intmat[:, start_mode:]

    @property
    def nmodes(self):
        return self.intmat.shape[1]

    @property
    def nslopes(self):
        return self.intmat.shape[0]

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PUP_TAG'] = self.pupdata_tag
        hdr['SA_TAG'] = self.subapdata_tag
        hdr['NORMFACT'] = self.norm_factor
        return hdr

    def save(self, filename, overwrite=False):
        if not filename.endswith('.fits'):
            filename += '.fits'
        hdr = self.get_fits_header()
        # Save fits file
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)
        fits.append(filename, cpuArray(self.intmat))
        if self.slope_mm is not None:
            fits.append(filename, self.slope_mm)
        if self.slope_rms is not None:
            fits.append(filename, self.slope_rms)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError
    
    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename, ext=0)
        intmat = fits.getdata(filename, ext=1)
        norm_factor = float(hdr.get('NORMFACT', 0.0))
        pupdata_tag = hdr.get('PUP_TAG', '')
        subapdata_tag = hdr.get('SA_TAG', '')
        # Reading additional fits extensions
        with fits.open(filename) as hdul:
            num_ext = len(hdul)
        if num_ext >= 4:
            slope_mm = fits.getdata(filename, ext=2)
            slope_rms = fits.getdata(filename, ext=3)
        else:
            slope_mm = slope_rms = None
        return Intmat(intmat, slope_mm, slope_rms, pupdata_tag, subapdata_tag, norm_factor, target_device_idx=target_device_idx)

    def generate_rec(self, nmodes=None, cut_modes=0, w_vec=None, interactive=False):
        if nmodes is not None:
            intmat = self.intmat[:, :nmodes]
        else:
            intmat = self.intmat
        recmat = self.pseudo_invert(intmat, n_modes_to_drop=cut_modes, w_vec=w_vec, interactive=interactive)
        rec = Recmat(recmat, target_device_idx=self.target_device_idx)
        rec.im_tag = self.norm_factor  # TODO wrong
        return rec

    def pseudo_invert(self, matrix, n_modes_to_drop=0, w_vec=None, interactive=False):
        # TODO handle n_modes_to_drop, and w_vec
        return self.xp.linalg.pinv(matrix)

    def build_from_slopes(self, slopes, disturbance):
        times = list(slopes.keys())
        nslopes = len(slopes[times[0]])
        nmodes = len(disturbance[times[0]])
        intmat = self.xp.zeros((nslopes, nmodes), dtype=self.dtype)
        iter_per_mode = self.xp.zeros(nmodes, dtype=self.dtype)
        slope_mm = self.xp.zeros((nmodes, 2), dtype=self.dtype)
        slope_rms = self.xp.zeros(nmodes, dtype=self.dtype)

        for t in times:
            amp = disturbance[t]
            mode = self.xp.where(amp)[0][0]
            intmat[:, mode] += slopes[t] / amp[mode]
            iter_per_mode[mode] += 1

        for m in range(nmodes):
            if iter_per_mode[m] > 0:
                intmat[:, m] /= iter_per_mode[m]

        im = Intmat(intmat)
        im._slope_mm = slope_mm
        im._slope_rms = slope_rms
        return im
