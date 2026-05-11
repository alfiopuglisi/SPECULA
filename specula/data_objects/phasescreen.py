
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from specula.lib.calc_phasescreen import calc_phasescreen

class Phasescreen(BaseDataObj):
    """
    Phasescreen field data object.
    """
    def __init__(self, 
                 dimx: int, 
                 dimy: int,
                 L0: float,
                 seed: int,
                 pixel_pitch: int,
                 target_device_idx: int=None, 
                 precision: int=None):
        """
        Initialize an :class:`~specula.data_objects.phasescreen.Phasescreen` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.L0 = L0
        self.seed = seed
        self.pixel_pitch = pixel_pitch
        self.phasescreen = self.xp.zeros((dimy, dimx), dtype=self.dtype)
        self.rebuild_with_cache(rebuild_func=self.rebuild)

    def get_value(self):
        '''
        Get the phasescreen as a numpy/cupy array
        '''
        return self.phasescreen

    def set_value(self, v):
        '''
        Set a new phasescreen.
        Arrays are not reallocated
        '''
        assert v.shape == self.phasescreen.shape, \
            f"Error: input array shape {v.shape} does not match phasescreen field shape {self.i.shape}"
        self.phasescreen[:]= self.to_xp(v)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Phasescreen'
        hdr['DIMX'] = self.phasescreen.shape[1]
        hdr['DIMY'] = self.phasescreen.shape[0]
        hdr['L0'] = self.L0
        hdr['SEED'] = self.seed
        hdr['PIXPITCH'] = self.pixel_pitch
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr, data=cpuArray(self.phasescreen))
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        L0 = hdr['L0']
        seed = hdr['SEED']
        pixel_pitch = hdr['PIXPITCH']
        phasescreen = Phasescreen(dimx, dimy, L0=L0, pixel_pitch=pixel_pitch, seed=seed, target_device_idx=target_device_idx)
        return phasescreen
    
    @staticmethod
    def restore(filename, update=None, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Phasescreen':
            raise ValueError(f"Error: file {filename} does not contain a Phasescreen object")
        if update is None:
            phasescreen = Phasescreen.from_header(hdr, target_device_idx=target_device_idx)
        else:
            phasescreen = update
        with fits.open(filename) as hdul:
            phasescreen.phasescreen[:] = phasescreen.to_xp(hdul[0].data)  # pylint: disable=no-member
        return phasescreen

    def array_for_display(self):
        return self.phasescreen

    def cache_filename(self):
        # Multiple filenames for PASSATA compatibility, first one used by default
        dimension = self.phasescreen.shape[0]
        pixel_pitch = self.pixel_pitch
        precision_str = 'single' if self.precision==1 else 'double'
        L0 = self.L0
        xp = self.xp
        name = f'ps_seed{xp.around(self.seed)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0):.4f}_{precision_str}.fits'
        name1 = f'ps_seed{xp.around(self.seed)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{xp.around(L0):.4f}_{precision_str}.fits'
        name2 = f'ps_seed{float(self.seed)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0):.4f}_{precision_str}.fits'
        name3 = f'ps_seed{float(self.seed)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{xp.around(L0):.4f}_{precision_str}.fits'
        return [name, name1, name2, name3]

    def rebuild(self):
        self.phasescreen[:] = calc_phasescreen(L0=self.L0,
                                               dimension=self.phasescreen.shape[0],
                                               pixel_pitch=self.pixel_pitch,
                                               seed=self.seed,
                                               xp=self.xp,
                                               precision=self.precision,
                                               target_device_idx=self.target_device_idx)
