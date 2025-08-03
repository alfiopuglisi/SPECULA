from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


class ModalrecImplicitPolc(Modalrec):
    """
    Modal reconstructor with implicit Pseudo Open Loop Control (POLC).
    
    This class is used to reconstruct the slopes using the implicit POLC method.
    It uses the command matrix (C = P * R, P projection matrix and R reconstruction
    matrix) and the H matrix (H = I - C * D, I identity and D interaction matrix)
    to compute the delta commands.
    
    It is typically used in the context of MCAO systems where the reconstruction
    matrix is defined on virtual DMs on a large number of layers (no. layers > no. real DMs).
    
    The implicit POLC method is used to reduce the computational cost of the
    reconstruction process by using smaller matrices, it also reduces the
    memory footprint of the reconstruction process (particularly useful
    for large systems and when using GPUs).
    """

    def __init__(self,
                 nmodes: int=None,      # TODO =0,
                 recmat: Recmat=None,
                 projmat: Recmat=None,
                 intmat: Intmat=None,
                 ncutmodes: int=None,
                 nSlopesToBeDiscarded: int=None,
                 dmNumber: int=0,
                 input_modes_index: list=None,
                 input_modes_slice: list=None,
                 output_slice: list=None,
                 in_commands_size: int=None,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(
                 nmodes,
                 recmat,
                 projmat,
                 intmat,
                 polc=True,
                 filtmat=None,
                 identity=False,
                 ncutmodes=ncutmodes,
                 nSlopesToBeDiscarded=nSlopesToBeDiscarded,
                 dmNumber=dmNumber,
                 noProj=False,
                 input_modes_index=input_modes_index,
                 input_modes_slice=input_modes_slice,
                 output_slice=output_slice,
                 in_commands_size=in_commands_size,
                 target_device_idx=target_device_idx,
                 precision=precision)

        if self.recmat is None or self.recmat.recmat is None:
            raise ValueError("Recmat object not valid")
        if self.projmat is None or self.projmat.recmat is None:
            raise ValueError("Projmat object not valid")
        if self.intmat is None or self.intmat.intmat is None:
            raise ValueError("Intmat object not valid")

        # set up the command matrix as the product of the projection matrix and the reconstruction matrix
        comm_mat = self.projmat.recmat @ self.recmat.recmat
        self.comm_mat = Recmat(comm_mat, target_device_idx=target_device_idx, precision=precision)
        # Now self.recmat and self.projmat can be removed to save memory
        self.recmat = None
        self.projmat = None

        # set up the H matrix
        h_mat = self.comm_mat.recmat @ self.intmat.intmat
        h_mat = self.xp.identity(h_mat.shape[0], dtype=self.dtype) - h_mat
        self.h_mat = Recmat(h_mat, target_device_idx=target_device_idx, precision=precision)
        # Now self.intmat can be removed to save memory
        self.intmat = None

    def trigger_code(self):

        commandsobj = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']
        if commandsobj is None:
            commandsobj = commands_list
            commands = self.xp.hstack([x.value for x in commands_list]) # TODO this line does not work on the first step
        else:
            commands = self.to_xp(commandsobj.value, dtype=self.dtype)

        # this is true on the first step only
        if commandsobj is None or commands.shape == ():
            commands = self.xp.zeros(self.comm_mat.recmat.shape[0], dtype=self.dtype)

        if self.input_modes_index is not None:
            commands = commands[self.input_modes_index]

        if self.input_modes_slice is not None:
            commands = commands[self.input_modes_slice]

        output_modes = self.comm_mat.recmat @ self.slopes - self.h_mat.recmat @ commands

        self.modes.value = output_modes[self.output_slice]

    def setup(self):
        super().setup()

        commands = self.local_inputs['in_commands']
        commands_list = self.local_inputs['in_commands_list']
        if not commands and (not commands_list or not all(commands_list)):
            raise ValueError("When POLC is used, either 'commands' or 'commands_list' must be given as an input")