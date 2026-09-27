from typing import List, Union
from specula import cpuArray, ASEC2RAD, np
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams


# Phasescreens are always defined at 500 nm
ATMO_WAVELENGTH = 500.0

# rot90() of a square array expressed as index transformation matrices, for k=0..3
_ROT90 = np.array([np.linalg.matrix_power([[0, 1], [-1, 0]], k) for k in range(4)])


class AtmoEvolution(BaseProcessingObj):
    """
    Atmospheric turbulence evolution processing object.
    Generates and evolves atmospheric phase screens based on input parameters such as
    seeing, wind speed, and wind direction.

    Each time step is split so that trigger_code() can be captured in a CUDA graph
    (see BaseProcessingObj.build_stream(), called in setup()):

    - prepare_trigger() does all the host-side work: it reads the seeing, wind speed
      and wind direction inputs, updates the layer positions (including screen
      cycling), and computes the rotation matrix, window shift and scale coefficient
      of each layer (_update_params()). These are packed in the self.params_cpu array,
      which is transferred to the device array self.params with a single copy.
    - trigger_code() computes each layer with two ndimage affine_transform() calls
      (window interpolation, then rotation) and a multiplication by the scale
      coefficient, all reading their parameters from the device array self.params.
      No host values or transfers are involved, so the captured graph stays valid
      when positions, wind direction or seeing change.
    - post_trigger() updates the host-side state: last_t and the generation_time
      of the output layers.

    The same code runs on CPU, using scipy instead of cupyx.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 L0: Union[float, List[float]],
                 heights: list,
                 Cn2: list,
                 data_dir: str = "",
                 fov: float=0.0,
                 pixel_phasescreens: int=8192,
                 seed: int=1,
                 extra_delta_time: float=0,
                 fov_in_m: float=None,
                 pupil_position:list =[0,0],
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Note
        ----
        Phase screens are always generated at a reference wavelength of 500 nm.

        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing global simulation settings.
        L0 : float or list [m]
            Outer scale(s) of turbulence for each layer in meters.
        heights : list [m]
            Heights of the atmospheric layers in meters (at zenith).
        Cn2 : list [1]
            Fractional Cn2 values for each layer (must sum to 1.0).
        data_dir : str
            Directory path for storing/loading phase screen data (automatically set by simul.py).
        fov : float [arcsec], optional
            Field of view in arcseconds. Default is 0.0.
        pixel_phasescreens : int [1], optional
            Size of the square phase screens in pixels. Default is 8192.
        seed : int [1], optional
            Seed for random number generation. Must be >0. Default is 1.
        extra_delta_time : float or list [s], optional
            Extra time offset for phase screen evolution in seconds. Default is 0.
        fov_in_m : float [m], optional
            Field of view in meters. If provided, overrides fov parameter. Default is None.
        pupil_position : list [m], optional
            [x, y] position of the pupil in meters. Default is [0, 0].
        target_device_idx : int [1], optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int [1], optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.pixel_pupil = simul_params.pixel_pupil
        self.pixel_pitch = simul_params.pixel_pitch
        zenithAngleInDeg = simul_params.zenithAngleInDeg

        self.n_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_phasescreens, dtype=self.dtype)
        self.last_effective_position = cpuArray(np.zeros(self.n_phasescreens, dtype=self.dtype))
        self.last_t = 0
        self.cycle_screens = True
        self.delta_time = None

        if not hasattr(extra_delta_time,"__len__"):
            self.extra_delta_time = cpuArray(self.n_phasescreens*[extra_delta_time])
        else:
            self.extra_delta_time = cpuArray(extra_delta_time)

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
            self.logger.info(f'zenith angle is defined as: {zenithAngleInDeg} deg')
            self.logger.info(f'airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        heights = np.array(heights, dtype=self.dtype)
        # distances from the pupil accounting for zenith angle
        self.pupil_distances = heights * self.airmass

        fov_rad = fov * ASEC2RAD
        self.pixel_layer = np.ceil(
            (self.pixel_pupil \
                + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) \
                / self.pixel_pitch \
                + abs(self.pupil_distances) / self.pixel_pitch * fov_rad) / 2.0
        ) * 2.0

        if fov_in_m is not None:
            self.pixel_layer = np.full_like(
                heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2
            )

        self.L0 = L0
        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.data_dir = data_dir

        self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < max(self.pixel_layer):
            raise ValueError('Error: phase-screens dimension must be'
                             'greater than layer dimension!')

        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_phasescreens):
            layer = Layer(self.pixel_layer[i],
                          self.pixel_layer[i],
                          self.pixel_pitch, heights[i],
                          precision=self.precision,
                          target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list

        # Per-layer parameters with shape (n_layer_lists, n_phasescreens, 13), computed
        # in prepare_trigger() by _update_params() and uploaded to self.params
        # (which is the same array on CPU): see _update_params() for the columns
        self.params_cpu = np.zeros((1, self.n_phasescreens, 13))
        self.params = self.to_xp(self.params_cpu)
        # Interpolated (not rotated) layer windows
        self.windows = [self.xp.zeros((int(n), int(n)), dtype=self.dtype) for n in self.pixel_layer]
        # Coordinates in double precision as in scipy
        self.affine_kwargs = {'float64_coords': True} if self.target_device_idx >= 0 else {}

        self.seed = seed
        self.scale_coeff = 1.0

        if self.seed <= 0:
            raise ValueError('seed must be >0')

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

        self.compute()

    @classmethod
    def input_names(cls):
        return {'seeing': InputDesc(BaseValue, 'Atmospheric seeing value'),
                'wind_speed': InputDesc(BaseValue, 'Wind speed for each atmospheric layer'),
                'wind_direction': InputDesc(BaseValue, 'Wind direction for each atmospheric layer')}

    @classmethod
    def output_names(cls):
        return {'layer_list': OutputDesc(list, 'List of atmospheric phase screen layers')}

    def compute(self):
        # Phase screens list
        self.phasescreens = []
        self.phasescreens_sizes = []

        self.pixel_phasescreens = int(self.xp.max(self.pixel_layer))
        temp_screens = []

        if len(self.xp.unique(self.to_xp(self.L0))) == 1:
            # Number of rectangular phase screens from a single square phasescreen
            n_ps_from_square_ps = self.xp.floor(
                self.pixel_square_phasescreens / self.pixel_phasescreens
            )
            # Number of square phasescreens
            n_ps = self.xp.ceil(float(self.n_phasescreens) / n_ps_from_square_ps)

            # Seed vector
            seed = self.xp.arange(self.seed, self.seed + int(n_ps))

            # Square phasescreens
            if hasattr(self.L0, '__len__'):
                L0 = self.L0[0]
            else:
                L0 = self.L0
            L0 = np.array([L0])
            square_phasescreens = phasescreens_manager(L0, self.pixel_square_phasescreens,
                                                        self.pixel_pitch, self.data_dir,
                                                        seed=seed, precision=self.precision,
                                                        xp=self.xp)

            square_ps_index = -1
            ps_index = 0

            for i in range(self.n_phasescreens):
                # Increase square phase-screen index
                if i % n_ps_from_square_ps == 0:
                    square_ps_index += 1
                    ps_index = 0

                temp_screen = square_phasescreens[square_ps_index][
                    int(self.pixel_phasescreens) * ps_index:
                    int(self.pixel_phasescreens) * (ps_index + 1), :
                ]
                temp_screens.append(temp_screen)
                ps_index += 1

        else:
            seed = self.seed + self.xp.arange(self.n_phasescreens)

            if len(seed) != len(self.L0):
                raise ValueError('Number of elements in seed and L0 must be the same!')

            # Square phasescreens
            square_phasescreens = phasescreens_manager(self.L0,
                                                       self.pixel_square_phasescreens,
                                                       self.pixel_pitch,
                                                       self.data_dir,
                                                       seed=seed,
                                                       precision=self.precision,
                                                       xp=self.xp)

            for i in range(self.n_phasescreens):
                temp_screen = square_phasescreens[i][ :int(self.pixel_phasescreens), :]
                temp_screens.append(temp_screen)


        # Normalize all phasescreens

        for i, temp_screen in enumerate(temp_screens):

            temp_screen = self.to_xp(temp_screen, dtype=self.dtype)
            temp_screen *= self.xp.sqrt(self.Cn2[i])
            temp_screen -= self.xp.mean(temp_screen)

            # Convert to nm
            temp_screen *= ATMO_WAVELENGTH / (2 * np.pi)

            # Flip x-axis for each odd phase-screen
            if i % 2 != 0:
                temp_screen = self.xp.flip(temp_screen, axis=1)

            # Contiguous, otherwise affine_transform() makes a copy at each call
            self.phasescreens.append(self.xp.ascontiguousarray(temp_screen))
            self.phasescreens_sizes.append(temp_screen.shape[1])

        self.phasescreens_sizes_array = np.asarray(self.phasescreens_sizes)

    def setup(self):
        super().setup()

        # check that seeing is a 1-element array
        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')

        # Check that wind speed and direction have the correct length
        if len(self.local_inputs['wind_speed'].value) != self.n_phasescreens:
            raise ValueError('Wind speed input must be a {self.n_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_phasescreens:
            raise ValueError('Wind direction input must be a {self.n_phasescreens}-elements array')

        self.build_stream()

    def prepare_trigger(self, t):
        """Host-side part of the time step.

        Reads the inputs, computes the seeing scale coefficient, updates the
        positions and layer parameters (_update_positions()) and uploads the
        parameters to the device with a single host-to-device copy.
        """
        super().prepare_trigger(t)
        self.delta_time = cpuArray(
            self.n_phasescreens*[self.t_to_seconds(self.current_time - self.last_t)]
        )
        seeing = float(cpuArray(self.local_inputs['seeing'].value[0]))
        if seeing > 0:
            r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
            self.scale_coeff = (self.pixel_pitch / r0)**(5./6.)
        else:
            self.scale_coeff = 0.0

        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        # Compute the delta position in pixels (time evolution)
        delta_position = wind_speed * self.delta_time / self.pixel_pitch  # [pixel]

        self._update_positions(wind_speed, wind_direction, delta_position)
        if self.params is not self.params_cpu:
            self.params.set(self.params_cpu)

    def _update_positions(self, wind_speed, wind_direction, delta_position):
        """Update positions and parameters (self.params_cpu) of all layer lists.

        Derived classes with more layer lists override this method.
        """
        self.last_effective_position[:] = self._update_params(
            wind_speed, wind_direction, delta_position, self.extra_delta_time,
            self.last_position, self.params_cpu[0]
        )

    def trigger_code(self):
        """Compute the layers from the parameters uploaded by prepare_trigger().

        On GPU, only kernel launches are performed, so that this method
        can be captured in a CUDA graph.
        """
        self._update_layer_list(self.layer_list, self.params[0])

    def post_trigger(self):
        """Host-side state update: last_t and generation_time of all output layers."""
        super().post_trigger()
        self.last_t = self.current_time
        for layer_list in self.outputs.values():
            for layer in layer_list:
                layer.generation_time = self.current_time

    def _update_params(self, wind_speed, wind_direction, delta_position, extra_delta_time,
                       last_position, params):
        """Update positions and compute the layer parameters for a layer list.

        Runs on the host (numpy) in prepare_trigger(). Positions are accumulated
        and cycled as before; the effective position (including the extra offset)
        gives the (2, 3) affine matrix that extracts the layer window from the phase
        screen with linear interpolation (params[:, 0:6]). The wind direction is
        split in a rot90() count and a residual angle, which are combined in the
        (2, 3) affine matrix [A | offset] of a rotation around the layer center,
        with A = R90^k @ R(residual) (params[:, 6:12]). This is the same index
        transformation as rot90() followed by ndimage rotate(), and A is an exact
        permutation when the residual angle is zero. params[:, 12] is the scale
        coefficient.

        Parameters
        ----------
        wind_speed : array [m/s]
            Wind speed for each layer [m/s]
        wind_direction : array [deg]
            Wind direction for each layer [deg]
        delta_position : array [pixels]
            Position change since last frame [pixels]
        extra_delta_time : array [s]
            Extra time offset for each layer [s]
        last_position : array [pixels]
            Last accumulated position (will be updated in place)
        params : array [1]
            Layer parameters array (will be updated in place)
        """

        # Compute extra offset that doesn't get accumulated
        extra_offset = wind_speed * extra_delta_time / self.pixel_pitch  # [pixel]

        # Update position with delta_position
        new_position = last_position + delta_position  # [pixel]

        # Cycle screens considering the effective position
        if self.cycle_screens:
            new_position = np.where(
                new_position + extra_offset + self.pixel_layer >= self.phasescreens_sizes_array,
                0,
                new_position
            )

        # Effective position = accumulated position + constant offset
        effective_position = new_position + extra_offset  # [pixel]

        # Get quotient and remainder for wind direction
        wdf, wdi = np.modf(wind_direction / 90.0)
        wdf_rad = np.radians(wdf * 90)
        cos, sin = np.cos(wdf_rad), np.sin(wdf_rad)

        # Index transformation of rot90() followed by ndimage.rotate() around the layer center
        mat = _ROT90[wdi.astype(int) % 4] @ np.moveaxis([[cos, sin], [-sin, cos]], -1, 0)
        center = (self.pixel_layer - 1) / 2
        params[:, 0:6] = [1, 0, 0, 0, 1, 0]
        params[:, 5] = effective_position
        offset = center[:, None] * (1 - mat.sum(axis=2))
        params[:, 6:12] = np.concatenate([mat, offset[:, :, None]], axis=2).reshape(-1, 6)
        params[:, 12] = self.scale_coeff

        # Update position in place
        last_position[:] = new_position

        return effective_position

    def _update_layer_list(self, layer_list, params):
        """Update a layer list using the parameters computed by _update_params().

        For each layer: window extraction with linear interpolation along x,
        rotation (both with affine_transform(order=1), parameters read from
        params) and multiplication by the scale coefficient.

        Parameters
        ----------
        layer_list : list [1]
            List of Layer objects to update
        params : array [1]
            Parameters of this layer list (a slice of self.params)
        """
        for ii, (p, window) in enumerate(zip(self.phasescreens, self.windows)):
            phase = layer_list[ii].phaseInNm
            self.ndimage_affine_transform(p, params[ii, 0:6].reshape(2, 3), output=window,
                                          output_shape=window.shape, order=1, **self.affine_kwargs)
            self.ndimage_affine_transform(window, params[ii, 6:12].reshape(2, 3), output=phase,
                                          order=1, **self.affine_kwargs)
            phase *= params[ii, 12]
