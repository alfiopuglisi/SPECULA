from specula.base_data_obj import BaseDataObj
from specula.data_objects.ifunc_inv import IFuncInv, cut_modes
from astropy.io import fits

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.compute_zern_ifunc import compute_zern_ifunc
from specula.lib.fast_pinv import fast_pinv
from specula.lib.graph_matmul import SparseColumns
from iteras.lib import storage


def compute_kl_ifunc(*args, **kwargs):
    raise NotImplementedError


def compute_mixed_ifunc(*args, **kwargs):
    raise NotImplementedError


class IFunc(BaseDataObj):
    """
    Influence functions data object.
    This class holds the influence function matrix and the corresponding mask.
    Influence functions data are stored as [modes, pixels].
    """
    def __init__(self,
                 ifunc=None,
                 type_str: str=None,
                 mask=None,
                 npixels: int=None,
                 nzern: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 start_mode: int=None,
                 nmodes: int=None,
                 n_act: int=None,
                 circ_geom: bool=True,
                 angle_offset: float=0,
                 do_mech_coupling: bool=False,
                 coupling_coeffs: list=[0.31, 0.05],
                 do_slaving: bool=False,
                 slaving_thr: float=0.1,
                 idx_modes=None,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(precision=precision, target_device_idx=target_device_idx)
        self.type_str = type_str

        if ifunc is None:
            if type_str is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (self.to_xp(mask) > 0).astype(self.dtype)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")

            type_lower = type_str.lower()
            if type_lower == 'kl':
                if nmodes is None:
                    raise ValueError('nmodes parameter is mandatory with type "kl"')
                ifunc, mask = compute_kl_ifunc(npixels, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                               xp=self.xp, dtype=self.dtype)
            elif type_lower in ['zern', 'zernike']:
                if nzern is not None:
                    raise ValueError('nzern is ignored with type "zern" or "zernike", please use nmodes instead')
                if nmodes is None:
                    raise ValueError('nmodes parameter is mandatory with type "zern" or "zernike"')
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                 xp=self.xp, dtype=self.dtype)
            elif type_lower == 'mixed':
                if nmodes is None or nzern is None:
                    raise ValueError('Both nzern and nmodes parameters are mandatory with type "mixed"')
                ifunc, mask = compute_mixed_ifunc(npixels, nzern=nzern, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                  xp=self.xp, dtype=self.dtype)
            elif type_lower == 'zonal':
                if n_act is None:
                    raise ValueError('nact parameter is mandatory with type "zonal"')
                ifunc, mask, _, _ = compute_zonal_ifunc(npixels, n_act, circ_geom=circ_geom,
                                                  angle_offset=angle_offset, do_mech_coupling=do_mech_coupling,
                                                  coupling_coeffs=coupling_coeffs, do_slaving=do_slaving,
                                                  slaving_thr=slaving_thr, obsratio=obsratio, diaratio=diaratio,
                                                  mask=mask, xp=self.xp, dtype=self.dtype)
            else:
                raise ValueError(f'Invalid ifunc type {type_str}')

        ifunc = self.to_xp(ifunc)
        mask = self.to_xp(mask)

        self._influence_function = ifunc
        self._sparse = None
        self._mask_inf_func = mask
        self._idx_inf_func = self.xp.where(self._mask_inf_func)
        self.cut(start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes)

    @property
    def influence_function(self):
        '''
        The influence functions (modes, pixels). After make_sparse(), they are
        rebuilt from the sparse form at each access: a new full-size array,
        meant for occasional use (e.g. calibrations), not at every step.
        '''
        if self._influence_function is None:
            return self._sparse.to_dense()
        return self._influence_function

    @influence_function.setter
    def influence_function(self, ifunc):
        self._influence_function = self.to_xp(ifunc, dtype=self.dtype)
        self._sparse = None

    def make_sparse(self):
        '''
        If the influence functions are mostly zeros (zonal or segmented ones,
        e.g. M4 or the ELT M1 segments), keep only their sparse form and drop
        the dense array, and return the sparse form (a SparseColumns, built once).
        Return None, and keep the dense array, if they are dense.
        '''
        if self._sparse is None:
            self._sparse = SparseColumns.from_dense(self._influence_function, self.xp)
            if self._sparse is not None:
                self._influence_function = None
        return self._sparse

    @property
    def is_sparse(self):
        '''True if only the sparse form is kept (see make_sparse())'''
        return self._influence_function is None

    @property
    def mask_inf_func(self):
        return self._mask_inf_func

    @mask_inf_func.setter
    def mask_inf_func(self, mask_inf_func):
        self._mask_inf_func = self.to_xp(mask_inf_func, dtype=self.dtype)
        self._idx_inf_func = self.xp.where(self._mask_inf_func)

    @property
    def idx_inf_func(self):
        return self._idx_inf_func

    @property
    def size(self):
        return self._sparse.shape if self.is_sparse else self._influence_function.shape

    def nmodes(self):
        return self.size[0]

    def npoints(self):
        return self.size[1]

    @property
    def type(self):
        return self._sparse.w.dtype if self.is_sparse else self._influence_function.dtype

    def get_value(self):
        return self.influence_function

    def set_value(self, v):
        '''Set a new influence function.
        Arrays are not reallocated, unless only the sparse form was kept.'''
        assert v.shape == self.size, \
            f"Error: input array shape {v.shape} does not match influence function shape {self.size}"
        if self.is_sparse:
            self.influence_function = v
        else:
            self._influence_function[:] = self.to_xp(v)

    def cut(self, start_mode=None, nmodes=None, idx_modes=None):
        self.influence_function = cut_modes(self.influence_function, start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes)

    def ifunc_2d_to_3d(self, normalize=True):
        '''Convert a 2D influence function to a 3D array using a mask.'''
        npixels = self._mask_inf_func.shape[0]
        ifunc = self.influence_function
        nmodes = ifunc.shape[0]
        ifunc_3d = self.xp.zeros((npixels, npixels, nmodes), dtype=self.dtype)
        idx = self.xp.where(self._mask_inf_func > 0)

        ifunc_3d[idx[0], idx[1], :] = ifunc.T

        if normalize:
            ifunc_rms = self.xp.sqrt(self.xp.mean(ifunc**2, axis=1))
            # Broadcasting: divide each mode by its rms value
            ifunc_3d[idx[0], idx[1], :] /= ifunc_rms[self.xp.newaxis, :]

        return ifunc_3d

    def inverse(self, nmodes=None, remove_piston=True):
        """Return the pseudoinverse of the influence function.

        When ``remove_piston`` is True, each mode is centered by subtracting
        its mean value before computing the pseudoinverse.
        """
        ifunc = self.influence_function
        if nmodes is not None and nmodes != self.nmodes():
            ifunc = cut_modes(ifunc, nmodes=nmodes)

        if remove_piston:
            ifunc = ifunc - self.xp.mean(ifunc, axis=1, keepdims=True)

        inv = fast_pinv(ifunc, self.xp)
        return IFuncInv(inv, mask=self._mask_inf_func, precision=self.precision,
                        target_device_idx=self.target_device_idx)

    @staticmethod
    def from_header(hdr):
        raise NotImplementedError

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'influence_function': self.influence_function.T, 'mask_inf_func': self._mask_inf_func},
                             overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None, exten=1):
        # In FITS files, *exten* is the index of the HDU of the influence function,
        # followed by the mask: the arrays are looked up by name only with the default value
        with storage.read_object(filename) as f:
            mask = f.read('mask_inf_func' if exten == 1 else None, exten + 1)
            ifunc = f.load('influence_function' if exten == 1 else None, exten,
                           target_device_idx=target_device_idx).T
        return IFunc(ifunc, mask=mask, target_device_idx=target_device_idx)
