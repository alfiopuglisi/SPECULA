import typing
import inspect
import itertools
from pathlib import Path
from collections import Counter
from specula import process_rank, MPI_DBG
from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj

from specula.loop_control import LoopControl
from specula.lib.utils import import_class, get_type_hints
from specula.calib_manager import CalibManager
from specula.param_dict import ParamDict, split_output
from specula.processing_objects.data_store import DataStore
from specula.connections import InputList, InputValue

import hashlib


def computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name):
    s = output_obj_name + '%' + dest_object + '%' + str(output_attr_name) + '%' + str(input_attr_name)
    rr = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**6
    return rr


class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self,
                 *param_files,
                 simul_idx=0,
                 overrides=None,
                 diagram=False,
                 diagram_title=None,
                 diagram_filename=None
                 ):
        if len(param_files) < 1:
            raise ValueError('At least one Yaml parameter file must be present')
        self.all_objs_ranks = {}
        self.remote_objs_ranks = {}
        self.param_files = param_files
        self.objs = {}
        self.simul_idx = simul_idx
        self.verbose = False  #TODO
        self.mainParams = None
        if overrides is None:
            self.overrides = []
        else:
            self.overrides = overrides
        self.diagram = diagram
        self.diagram_title = diagram_title
        self.diagram_filename = diagram_filename

    def input_ref(self, input_name):
        '''
        return reference to the input, or None if the object is remote.
        '''
        return self.get_ref(input_name, use_inputs=True)
        
    def get_ref(self, output_name, use_inputs=False):
        '''
        return reference to the output, or None if the object is remote.
        '''
        output = split_output(output_name)
        obj_name = output.obj_name
        output_key = output.output_key

        if not obj_name in self.objs:
            if obj_name in self.remote_objs_ranks:
                ref = None
            else:
                raise ValueError(f'Object {obj_name} does not exist anywhere')
        elif output_key is None:
            ref = self.objs[obj_name]
        else:
            if use_inputs:
                array_to_check, display_str = self.objs[obj_name].local_inputs, 'input'
            else:
                array_to_check, display_str = self.objs[obj_name].outputs, 'output'
            if not output_key in array_to_check:
                raise ValueError(f'Object {obj_name} does not define an {display_str} with name {output_key}')
            else:
                ref = array_to_check[output_key]
        return ref
  
    def trigger_order(self, params_orig):
        '''
        Work on a copy of the parameter file.
        1. Find leaves, add them to trigger
        2. Remove leaves, remove their inputs from other objects
          2a. Objects will become a leaf when all their inputs have been removed
        3. Repeat from step 1. until there is no change
        4. Check if any objects have been skipped
        '''
        assert isinstance(params_orig, ParamDict), \
            "input params must be an instance of ParamDict class"
        order = []
        order_index = []
        params = params_orig.copy()
        for index in itertools.count():
            leaves = [name for name, pars in params.items() if params.is_leaf(pars)]
            if len(leaves) == 0:
                break
            start = len(params)
            for leaf in leaves:
                if params.has_delayed_output(leaf):
                    continue
                order.append(leaf)
                order_index.append(index)
                del params[leaf]
                params.remove_inputs(leaf)
            end = len(params)
            if start == end:
                raise ValueError('Cannot determine trigger order: circular loop detected in {leaves}')
        if len(params) > 0:
            print('Warning: the following objects will not be triggered:', params.keys())
        return order, order_index

    def setSimulParams(self, params):
        _, main_pars = self.get_by_class('SimulParams')
        self.mainParams = main_pars

    def create_input_list_inputs(self, params):
        '''
        Create inputs for objects that use input_list parameter.
        Currently supported: DataStore, DataBuffer
        
        TODO the modifications of param dict done here should be
        somehow moved to ParamDict
        '''
        supported_classes = ['DataBuffer', 'DataStore']

        for key, pars in params.filter_by_class(*supported_classes):
            if key not in self.objs:
                continue

            try:
                input_list = pars['inputs']['input_list']
            except KeyError:
                continue
               
            for single_output in input_list:
                output = split_output(single_output)
                ref = self.get_ref(single_output)
                if type(ref) is list:
                    self.objs[key].inputs[output.input_name] = InputList(type=type(ref[0]))
                else:
                    self.objs[key].inputs[output.input_name] = InputValue(type=type(ref))
                params[key]['inputs'][output.input_name] = single_output
            del params[key]['inputs']['input_list']

            if pars['class'] == 'DataBuffer':
                self.objs[key].setOutputs()
   
    def build_objects(self, params):

        assert isinstance(params, ParamDict), \
            "input params must be an instance of ParamDict class"

        self.setSimulParams(params)

        cm = CalibManager(self.mainParams['root_dir'])
        skip_pars = 'class inputs outputs'.split()

        if MPI_DBG: print(process_rank, 'building objects')

        for key in params.build_order():

            pars = params[key]
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            klass = import_class(classname)
            args = inspect.getfullargspec(getattr(klass, '__init__')).args
            hints = get_type_hints(klass)

            target_device_idx = pars.get('target_device_idx', None)
                        
            par_target_rank = pars.get('target_rank', None)
            if par_target_rank is None:
                target_rank = 0
                self.all_objs_ranks[key] = 0
            else:
                target_rank = par_target_rank     
                self.all_objs_ranks[key] = par_target_rank
                del pars['target_rank']        

            # create the simulations objects for this process. Data Objects are created
            # on all ranks (processes) by default, unless a specific rank has been specified.

            build_this_object = (process_rank == target_rank) or \
                                (issubclass(klass, BaseDataObj) and (par_target_rank == None)) or \
                                (issubclass(klass, BaseDataObj) and (par_target_rank == process_rank)) or \
                                (process_rank == None)

            # If not build, remember the remote rank of this object (needed for connections setup)
            if not build_this_object:
                self.remote_objs_ranks[key] = target_rank
                continue

            if 'tag' in pars:
                if 'target_device_idx' in pars:
                    del pars['target_device_idx']
                if len(pars) > 2:
                    raise ValueError('Extra parameters with "tag" are not allowed')
                filename = cm.filename(classname, pars['tag'])
                # tags are restored into each process (multiple copies), target_rank is not checked
                print('Restoring:', filename)
                self.objs[key] = klass.restore(filename, target_device_idx=target_device_idx)
                self.objs[key].printMemUsage()
                self.objs[key].name = key
                continue

            pars2 = {}
            for name, value in pars.items():

                if key != 'data_source' and name in skip_pars:
                    continue

                if key == 'data_source' and name in ['class']:
                    continue

                # dict_ref field contains a dictionary of names and associated data objects (defined in the same yml file)
                elif name.endswith('_dict_ref'):
                    data = {x : self.objs[x] for x in value}
                    pars2[name[:-4]] = data

                elif name.endswith('_ref'):
                    data = self.objs[value]
                    pars2[name[:-4]] = data

                # data fields are read from a fits file
                elif name.endswith('_data'):
                    data = cm.read_data(value)
                    pars2[name[:-5]] = data

                # object fields are data objects which are loaded from a fits file
                # the name of the object is the string preceeding the "_object" suffix,
                # while its type is inferred from the constructor of the current class
                elif name.endswith('_object'):
                    parname = name[:-7]
                    if value is None:
                        pars2[parname] = None
                    elif parname in hints:
                        partype = hints[parname]

                        # Handle Optional and Union types (for python <3.11)
                        if hasattr(partype, "__origin__") and partype.__origin__ is typing.Union:
                            # Extract actual class type from Optional/Union
                            # (first non-None type argument)
                            for arg in partype.__args__:
                                if arg is not type(None):  # Skip NoneType
                                    partype = arg
                                    break
                        # data objects are restored into each process (multiple copies), target_rank is not checked
                        filename = cm.filename(parname, value)  # TODO use partype instead of parname?
                        print('Restoring:', filename)
                        parobj = partype.restore(filename, target_device_idx=target_device_idx)
                        parobj.printMemUsage()

                        pars2[parname] = parobj
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                else:
                    pars2[name] = value

            # Add global and class-specific params if needed
            my_params = {}

            if 'data_dir' in args and 'data_dir' not in my_params:  # TODO special case
                my_params['data_dir'] = cm.root_subdir(classname)

            if 'params_dict' in args:
                my_params['params_dict'] = params

            if 'input_ref_getter' in args:
                my_params['input_ref_getter'] = self.input_ref

            if 'output_ref_getter' in args:
                my_params['output_ref_getter'] = self.get_ref

            if 'info_getter' in args:
                my_params['info_getter'] = self.get_info

            my_params.update(pars2)
            try:
                self.objs[key] = klass(**my_params)
            except Exception:
                print(f'Exception building', key)
                raise
            if classname != 'SimulParams':
                self.objs[key].stopMemUsageCount()

            self.objs[key].name = key

            # TODO this could be more general like the getters above
            if type(self.objs[key]) is DataStore:
                self.objs[key].setParams(params.params)

    def connect(self, output_name, input_name, dest_object):
        '''
        Connect the output *output_name* to the input *input_name*
        of the object *dest_object*, which might be local or remote.

        This routine handles the three cases:
        1. local output to local input - use Python references
        2. local output to remote input - use addRemoteOutput() to send the output to the remote object
        3. remote output to local input - use set_remote_rank() to set the remote rank of the input
        '''
        output = split_output(output_name)
        ref = self.get_ref(output_name)
        local_dest_object = dest_object in self.objs.keys()

        send = ref is not None and local_dest_object is False
        recv = ref is None and local_dest_object is True
        local = ref is not None and local_dest_object is True
        if send or recv:
            tag = computeTag(output.obj_name, dest_object, output.output_key, input_name)

        if MPI_DBG: print(process_rank, f'{output.obj_name}.{output.output_key} -> {dest_object} : {send=} {recv=} {local=}', flush=True)

        if recv:
            if MPI_DBG: print(process_rank, f'CONNECT Connecting remote output {output.obj_name}.{output.output_key} to local input {dest_object}.{input_name} with tag {tag}')
            self.objs[dest_object].inputs[input_name].append(None,
                                                            remote_rank = self.remote_objs_ranks[output.obj_name],
                                                            tag=tag)
        if local:
            if MPI_DBG: print(process_rank, f'CONNECT Connecting local output {output.obj_name}.{output.output_key} to local input {dest_object}.{input_name}')
            self.objs[dest_object].inputs[input_name].append(ref)

        if send:
            self.objs[output.obj_name].addRemoteOutput(output.output_key, (self.remote_objs_ranks[dest_object], 
                                                                            tag,
                                                                            output.delay))
                
    def connect_objects(self, params):
        self.connections = []
        
        for dest_object, pars in params.items():

            if MPI_DBG: print(process_rank, 'connect_objects for', dest_object, flush=True)

            local_dest_object = dest_object in self.objs.keys()

            # Check that outputs exist (or for remote objects, that they are defined in the params)
            if 'outputs' in pars:
                for output_name in pars['outputs']:
                    if local_dest_object:
                        # check that this output was actually created by this dest_object
                        if not output_name in self.objs[dest_object].outputs:
                            raise ValueError(f'Object {dest_object} does not have an output called {output_name}')
                    else:
                        # remote object case
                        # TODO these checks are almost all reduntant
                        if not ( self.all_objs_ranks[dest_object] != process_rank \
                             and 'outputs' in params[dest_object] \
                             and output_name in params[dest_object]['outputs'] ):
                            raise ValueError(f'Remote Object {dest_object} does not have an output called {output_name}')

            if 'inputs' not in pars:
                continue

            for input_name, output_name in pars['inputs'].items():

                if MPI_DBG: print(process_rank, 'ASSIGNMENT of input_name:', input_name, flush=True)
                if MPI_DBG: print(process_rank, 'output_name', output_name, flush=True)

                if local_dest_object and input_name != 'input_list':
                    if not input_name in self.objs[dest_object].inputs:
                        raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')

                if not isinstance(output_name, (str, list)):
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')

                for single_output_name in output_name if isinstance(output_name, list) else [output_name]:
                    if MPI_DBG: print(process_rank, 'List input', flush=True)

                    output = split_output(single_output_name)
                    ref = self.get_ref(single_output_name)

                    # Remote-to-remote: nothing to do
                    if not local_dest_object and ref is None:
                        continue
                    
                    try:
                        self.connect(single_output_name, input_name, dest_object)
                    except ValueError:
                        print(f'Exception while connecting {single_output_name} {dest_object}.{input_name}')
                        raise

                    a_connection = {}
                    a_connection['start'] = output.obj_name
                    a_connection['end'] = dest_object
                    a_connection['start_label'] = output.output_key
#                    a_connection['middle_label'] = self.objs[dest_object].inputs[use_input_name]
#                    a_connection['end_label'] = self.objs[dest_object].inputs[use_input_name]
                    self.connections.append(a_connection)

    def isReplay(self, params):
        return 'data_source' in params

    def build_replay(self, params):
        replay_params = params.copy()
        obj_to_remove = []
        data_source_outputs = {}
        try:
            replay_params.data_store_to_data_source()
        except ValueError:
            print('Warning: no DataStore found, replay_params is identical to params')
            return replay_params

        _, pars = params.get_by_class('DataStore')  # It also checks that only one is present
        
        for name in pars['inputs']['input_list']:
            output = split_output(name)
            obj_to_remove.append(output.obj_name)
            output_name = f'{output.obj_name}.{output.output_key}'
            data_source_outputs[output_name] = 'data_source.' + output.input_name

        # Remove objects whose outputs have been saved and will
        # be replayed by DataSource
        for obj_name in set(obj_to_remove):
            del replay_params[obj_name]

        # Replace inputs whose data has been saved so that
        # they are now references to DataSource
        for _, pars in replay_params.filter_by_class(exclude='DataSource'):
            if 'inputs' in pars.keys():
                for input_name, output_name in pars['inputs'].items():
                    if type(output_name) is list:
                        print('TODO: list of inputs is not handled in output replay')
                        continue
                    if output_name in data_source_outputs.keys():
                        pars['inputs'][input_name] = data_source_outputs[output_name]

        for obj in self.objs.values():
            if type(obj) is DataStore:
                obj.setReplayParams(replay_params)
        return replay_params

    def arrangeInGrid(self, trigger_order, trigger_order_idx):
        rows = []
        n_cols = max(trigger_order_idx) + 1                
        n_rows = max( list(dict(Counter(trigger_order_idx)).values()))        
        # names_to_orders = dict(zip(trigger_order, trigger_order_idx))
        orders_to_namelists = {}
        for order in range(n_cols):
            orders_to_namelists[order] = []
        for name, order in zip(trigger_order, trigger_order_idx):
            orders_to_namelists[order].append(name)

        for ri in range(n_rows):
            r = []
            for ci in range(n_cols):
                col_elements = len(orders_to_namelists[ci])
                if ri<col_elements:
                    block_name = orders_to_namelists[ci][ri]
                else:
                    block_name = ""                
                r.append(block_name)
            rows.append(r)
        return rows

    def buildDiagram(self):
        from orthogram import Color, DiagramDef, write_png, Side, FontWeight, TextOrientation

        print('Building diagram...')

        d = DiagramDef(label=self.diagram_title, text_fill=Color(0, 0, 1), scale=2.0, collapse_connections=True)
        rows = self.arrangeInGrid(self.trigger_order, self.trigger_order_idx)
        # a row is a list of strings, which are labels for the cells
        for r in rows:
            d.add_row(r)        
        for c in self.connections:
            aconn = d.add_connection(c['start'], c['end'], buffer_fill=Color(1.0,1.0,1.0), buffer_width=1, 
                             exits=[Side.RIGHT], entrances=[Side.LEFT, Side.BOTTOM, Side.TOP])
            #aconn.set_start_label(c['middle_label'],font_weight=FontWeight.BOLD, text_fill=Color(0, 0.5, 0), text_orientation=TextOrientation.HORIZONTAL)
        write_png(d, self.diagram_filename)
        print('Diagram saved.')

    def run(self):
        params = ParamDict()
        params.load(*self.param_files)
        params.apply_overrides(self.overrides)
        self.setSimulParams(params)

        self.trigger_order, self.trigger_order_idx = self.trigger_order(params)
        print(f'{self.trigger_order=}')
        print(f'{self.trigger_order_idx=}')

        if not self.isReplay(params):
            replay_params = self.build_replay(params)
        else:
            replay_params = None

        self.build_objects(params)
        self.create_input_list_inputs(params)
        self.connect_objects(params)

        if replay_params is not None:
            for obj in self.objs.values():
                if type(obj) is DataStore:
                    obj.setReplayParams(replay_params)

        # Initialize housekeeping objects
        self.loop = LoopControl()

        if self.diagram or self.diagram_filename or self.diagram_title:
            if self.diagram_filename is None:
                self.diagram_filename = str(Path(self.param_files[0]).with_suffix('.png'))
            if self.diagram_title is None:
                self.diagram_title = str(Path(self.param_files[0]).with_suffix(''))
            self.buildDiagram()

        # Build loop
        for name, idx in zip(self.trigger_order, self.trigger_order_idx):
            if name not in self.remote_objs_ranks:
                obj = self.objs[name]
                if isinstance(obj, BaseProcessingObj):
                    self.loop.add(obj, idx)
        
        self.loop.max_global_order = max(self.trigger_order_idx)
        print('self.loop.max_global_order', self.loop.max_global_order, flush=True)

        # Default display web server
        if 'display_server' in self.mainParams and self.mainParams['display_server'] and process_rank in [0, None]:
            from specula.processing_objects.display_server import DisplayServer
            disp = DisplayServer(params, self.input_ref, self.get_ref, self.get_info)
            self.objs['display_server'] = disp
            self.loop.add(disp, idx+1)
            disp.name = 'display_server'

        # Run simulation loop
        self.loop.run(run_time=self.mainParams['total_time'], dt=self.mainParams['time_step'], speed_report=True)

        print(process_rank, 'Simulation finished', flush=True)
#        if data_store.has_key('sr'):
#            print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * self.mainParams['total_time'] / self.mainParams['time_step']])) * 100.}")

    def get_info(self):
        '''Quick info string intended for web interfaces'''
        name= f'{self.param_files[0]}'
        curtime= f'{self.loop.t / self.loop._time_resolution:.3f}'
        stoptime= f'{self.loop.run_time / self.loop._time_resolution:.3f}'

        info = f'{curtime}/{stoptime}s'
        return name, info
