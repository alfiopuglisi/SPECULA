
from astropy.io import fits
from specula.base_data_obj import BaseDataObj

class BaseValue(BaseDataObj):
    def __init__(self, description='', value=None, target_device_idx=None):
        """
        Initialize the base value object.

        Parameters:
        description (str, optional)
        value (any, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(target_device_idx=target_device_idx)
        self.description = description
        self.value = value
        
    def get_value(self):
        return self.value

    def set_value(self, val, force_copy=False):
        if not self.value is None and not force_copy:
            self.value[:] = self.to_xp(val)
        else:
            self.value = self.to_xp(val)

    def save(self, filename):
        hdr = fits.Header()
        if self.value is not None:
            hdr['VALUE'] = str(self.value)  # Store as string for simplicity
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            if self.value is not None:
                hdr['VALUE'] = str(self.value)
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            value_str = hdr.get('VALUE', None)
            if value_str is not None:
                self.value = eval(value_str)  # Convert back from string to original type

    def array_for_display(self):
        return self.value
    
    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseValue'
        return hdr