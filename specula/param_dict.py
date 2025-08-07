
import copy
import yaml
from collections import namedtuple


Output = namedtuple('Output', 'obj_name output_key delay input_name')


def split_output(output_name):
    '''
    Split an output name into object name, output key (if any, might not be defined for data objects),
    delay (default to zero) and an input name, if any (used by DataStore and similar)
    '''
    if ':' in output_name:
        output_name, delay = output_name.split(':')
        delay = int(delay)
    else:
        delay = 0
    if '-' in output_name:
        input_name, output_name = output_name.split('-')
    else:
        input_name = None

    if '.' in output_name:
        obj_name, output_key = output_name.split('.')
    else:
        obj_name = output_name
        output_key = None

    return Output(obj_name, output_key, delay, input_name)

def output_owner(output_name):
    return split_output(output_name).obj_name

def output_delay(output_name):
    return split_output(output_name).delay
    

class ParamDict():
    
    def __init__(self, simul_idx: int=0):
        self.params = {}
        self.simul_idx = simul_idx
        self.verbose = False  # TODO
    
    def copy(self):
        newdict = ParamDict(simul_idx=self.simul_idx)
        newdict.params = copy.deepcopy(self.params)
        return newdict

    def items(self):
        for k, v in self.params.items():
            yield k, v

    def keys(self):
        for key in self.params.keys():
            yield key
            
    def values(self):
        for value in self.params.values():
            yield value

    def __len__(self):
        return len(self.params)

    def __getitem__(self, k):
        return self.params[k]

    def __setitem__(self, k, v):
        self.params[k] = v

    def __delitem__(self, k):
        del self.params[k]

    def pop(self, k):
        self.params.pop(k)

    def load(self, *param_files):
        '''
        Load params from one or more YAML files on disk
        '''
        print('Reading parameters from', param_files[0])
        with open(param_files[0], 'r') as stream:
            self.params = yaml.safe_load(stream)

        for filename in param_files[1:]:
            print('Reading additional parameters from', filename)
            with open(filename, 'r') as stream:
                additional_params = yaml.safe_load(stream)                
                self.combine_params(additional_params)

    def combine_params(self, additional_params):
        '''
        Add/update/remove params with additional_params
        '''
        for name, values in additional_params.items():
            doRemoveIdx = False            
            if '_' in name:
                ri = name.split('_')
                # check for a remove (with simulation index) list, something of the form:  remove_3: ['atmo', 'rec', 'dm2']                
                if len(ri) == 2:
                    if ri[0] == 'remove':
                        if int(ri[1]) == self.simul_idx:
                            doRemoveIdx = True
                        else:
                            continue

                # check for a override (with simulation index) parameters structure, something of the form:  dm_override_2: { ... }                
                if ri[-1].isnumeric() and ri[-2] == 'override':
                    if int(ri[-1]) == self.simul_idx:
                        separator = "_"
                        objname = separator.join(ri[:-2])                        
                        if objname not in self.params:
                            raise ValueError(f'Parameter file has no object named {objname}')
                        self.params[objname].update(values)
                    continue

            if name == 'remove' or doRemoveIdx:
                for objname in values:
                    if objname not in self.params:
                        raise ValueError(f'Parameter file has no object named {objname}')
                    del self.params[objname]
                    print(f'Removed {objname}')
                    # Remove corresponding inputs
                    self.params = self.remove_inputs(self.params, objname)
            elif name.endswith('_override'):
                objname = name[:-9]
                if objname not in self.params:
                    raise ValueError(f'Parameter file has no object named {objname}')
                self.params[objname].update(values)
            else:
                if name in self.params:
                    raise ValueError(f'Parameter file already has an object named {name}')
                self.params[name] = values

    def apply_overrides(self, overrides):
        '''
        Apply YAML overrides to params
        '''
        if len(overrides) > 0:
            for k, v in yaml.full_load(overrides).items():
                obj_name, param_name = k.split('.')
                self.params[obj_name][param_name] = v
                print(obj_name, param_name, v)

    def remove_inputs(self, obj_to_remove):
        '''
        Modify params removing all references to the specificed object name
        '''
        key = 'inputs'
        for name, pars in self.params.items():
            if key not in pars:
                continue
            inputs_copy = copy.deepcopy(pars[key])
            for input_name, output_name in pars[key].items():
                if isinstance(output_name, str):
                    owner = output_owner(output_name)
                    if owner == obj_to_remove:
                        del inputs_copy[input_name]
                        if self.verbose:
                            print(f'Deleted {input_name} from {pars[key]}')
                elif isinstance(output_name, list):
                    newlist = [x for x in output_name if output_owner(x) != obj_to_remove]
                    diff = set(output_name).difference(set(newlist))
                    inputs_copy[input_name] = newlist
                    if len(diff) > 0:
                        if self.verbose:
                            print(f'Deleted {diff} from {pars[key]}')
            pars[key] = inputs_copy

    def has_delayed_output(self, obj_name):
        '''
        Find out if an object has an output
        that is used as a delayed input for another
        object in the pars dictionary
        '''
        for name, pars in self.params.items():
            if 'inputs' not in pars:
                continue
            for input_name, output_name in pars['inputs'].items():
                if isinstance(output_name, str):
                    outputs_list = [output_name]
                elif isinstance(output_name, list):
                    outputs_list = output_name
                else:
                    raise ValueError('Malformed output: must be either str or list')

                for x in outputs_list:
                    owner = output_owner(x)
                    delay = output_delay(x)
                    if owner == obj_name and delay < 0:
                        # Delayed input detected
                        return True
        return False
    
    def is_leaf(self, pars):
        '''
        Returns True if the passed object parameter dictionary
        does not specify any inputs for the current iterations (delay=0)
        Inputs coming from previous iterations (:-1 syntax) are ignored.
        '''
        if 'inputs' not in pars:
            return True

        for input_name, output_name in pars['inputs'].items():
            if isinstance(output_name, str):
                maxdelay = output_delay(output_name)
            elif isinstance(output_name, list):
                maxdelay = -1
                if len(output_name) > 0:
                    maxdelay = max([output_delay(x) for x in output_name])
            if maxdelay == 0:
                return False
        return True
            
    def filter_by_class(self, *classnames, exclude=None):
        '''
        Yield (key, pars) tuples for each YAML object with the specified classname(s),
        or all YAML objects excluding the specified classname.
        '''
        for key, pars in self.items():
            try:
                my_classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')
            if len(classnames) > 0 and my_classname in classnames:
                yield key, pars
            elif exclude is not None and my_classname != exclude:
                yield key, pars
                
    def get_by_class(self, classname):
        '''
        Find exactly one YAML object with the specified classname
        and return a (key, pars) tuple
        '''
        parlist = list(self.filter_by_class(classname))
        if len(parlist) == 0:
            raise ValueError(f'No object with {classname=} found')
        if len(parlist) >= 2:
            raise ValueError(f'Multiple objects with {classname=} found instead of one (found N={len(parlist)})')
        return parlist[0]
    
    def build_order(self):
        '''
        Return the correct object build order, taking into account
        dependencies specified by _ref and _dict_ref parameters
        '''
        build_order = []

        def add_to_build_order(key):
            if key in build_order:
                return

            pars = self.params[key]
            for name, value in pars.items():
                if name.endswith('_ref'):
                    objlist = value if type(value) is list else [value]
                    for output in objlist:
                        owner = output_owner(output)
                        if owner not in build_order:
                            add_to_build_order(owner)

            build_order.append(key)

        for key in self.params.keys():
            add_to_build_order(key)

        return build_order