from specula.base_processing_obj import BaseProcessingObj


class CompositeWFS(BaseProcessingObj):
    """
    Abstract base class for WFSs made of several internal WFS objects.

    Derived classes fill the ``self._wfs_instances`` list in their __init__(),
    connect the inputs of the internal WFSs in connect_wfs_inputs() and
    combine their outputs into the composite output in combine_wfs_outputs().

    Each internal WFS goes through its regular processing steps: in particular,
    each one runs its own trigger(), with its own CUDA stream and graph, so that
    internal WFSs on different devices run in parallel. The composite object
    does not compute anything in trigger(), its processing steps are those
    of BaseProcessingObj, even when it derives from a WFS class.
    """

    def connect_wfs_inputs(self):
        '''
        Override to set the inputs of the internal WFSs
        '''
        raise NotImplementedError

    def combine_wfs_outputs(self):
        '''
        Override to combine the outputs of the internal WFSs
        '''
        raise NotImplementedError

    def setup(self):
        BaseProcessingObj.setup(self)
        self.connect_wfs_inputs()
        for wfs in self._wfs_instances:
            wfs.setup()

    def check_ready(self, t):
        '''
        The internal WFSs are checked after our own check_ready(),
        because our prepare_trigger() may update their inputs.
        Their check_ready() also calls their prepare_trigger().
        '''
        ready = BaseProcessingObj.check_ready(self, t)
        if ready:
            for wfs in self._wfs_instances:
                wfs.check_ready(t)
        return ready

    def prepare_trigger(self, t):
        '''
        The internal WFSs are not called here: their prepare_trigger()
        is called by their own check_ready() (see check_ready() above),
        which also sets their inputs_changed flag and refreshes their inputs.
        Calling it here as well would run it twice per step.

        BaseProcessingObj is called explicitly to skip the prepare_trigger()
        of a WFS base class (e.g. SH for DistributedSH), since the composite
        object does not compute anything itself.
        '''
        BaseProcessingObj.prepare_trigger(self, t)

    def trigger(self):
        BaseProcessingObj.trigger(self)
        for wfs in self._wfs_instances:
            wfs.trigger()

    def trigger_code(self):
        '''
        Nothing to do here, the internal WFSs run in their own trigger()
        '''
        return

    def post_trigger(self):
        BaseProcessingObj.post_trigger(self)
        for wfs in self._wfs_instances:
            wfs.post_trigger()

        if self.target_device_idx >= 0:
            self._target_device.use()
        self.combine_wfs_outputs()
