
import os
import pickle
from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.lib.utils import import_class


class DataSource(BaseProcessingObj):
    '''Data source object'''

    def __init__(self,
                outputs: list,         # TODO =[],
                store_dir: str,        # TODO ="",
                data_format: str='fits'):
        super().__init__()
        self.items = {}
        self.storage = {}
        self.data_filename = ''
        self.tn_dir = store_dir
        self.data_format = data_format
        self.headers = {}
        self.obj_type = {}

        for aout in outputs:            
            self.loadFromFile(aout)
        for k in self.storage.keys():
            if self.obj_type[k] not in ['BaseValue', 'BaseDataObj']:
                self.outputs[k] = import_class(self.obj_type[k]).from_header(self.headers[k])
            else:
                self.outputs[k] = BaseValue(target_device_idx=self.target_device_idx)

    def loadFromFile(self, name):
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        if self.data_format=='fits':
            self.load_fits(name)
        elif self.data_format=='pickle':
            self.load_pickle(name)

    def load_pickle(self, name):
        filename = os.path.join(self.tn_dir,name + '.pickle')
        with open( filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        times = unserialized_data['times']
        data = unserialized_data['data']

        if 'hdr' in unserialized_data:
            self.headers[name] = unserialized_data['hdr']
            self.obj_type[name] = self.headers[name]['OBJ_TYPE']
        
        self.storage[name] = { t:data[i] for i, t in enumerate(times.tolist())}

    def load_fits(self, name):
        filename = os.path.join(self.tn_dir, name+'.fits')
        with fits.open(filename) as hdul:
            self.headers[name] = dict(hdul[0].header)
            self.obj_type[name] = self.headers[name]['OBJ_TYPE']
            times = hdul[1].data.copy()
            data = hdul[0].data.copy()
        self.storage[name] = { t:data[i] for i, t in enumerate(times.tolist())}

    def size(self, name, dimensions=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self.storage[name]
        return h.shape if not dimensions else h.shape[dimensions]

    def trigger_code(self):
        for k in self.storage.keys():            
            value = self.storage[k][self.current_time]
            if value.ndim == 0:
                value = self.xp.array([value])
            self.outputs[k].set_value(self.outputs[k].xp.array(value))
            self.outputs[k].generation_time = self.current_time

        