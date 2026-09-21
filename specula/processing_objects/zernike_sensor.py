from specula.processing_objects.modulated_pyramid import ModulatedPyramid

class ZernikeSensor(ModulatedPyramid):
    """
    Zernike Sensor processing object.
    Based on phase-shifting focal-plane spot technique, the class 
    inherits from ModulatedPyramid but replaces the pyramid structure with
    a π/2 (default value) phase-shifting spot in the focal plane.
    """

    def __init__(self,
                 simul_params,
                 wavelengthInNm,
                 fov,
                 pup_diam,
                 output_resolution,
                 spot_radius_lambda: float= 1.0,  # Spot radius in λ/D units
                 phase_shift_pi: float = 0.5,  # π/2 phase shift
                 fft_res: float = 4.0,
                 target_device_idx=None,
                 precision=None):

        self.spot_radius_lambda = spot_radius_lambda
        self.phase_shift_pi = phase_shift_pi

        # Force modulation to zero (no modulation for Zernike sensor)
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_step=1,
            fft_res=fft_res,
            pup_dist=1,
            pup_margin=0,
            min_pup_dist=0,
            fov_errinf=0.1,
            fov_errsup=10.0,
            focal_plane_mask_type='zernike_spot',
            spot_radius_lambda=spot_radius_lambda,
            phase_shift_pi=phase_shift_pi,
            target_device_idx=target_device_idx,
            precision=precision
        )