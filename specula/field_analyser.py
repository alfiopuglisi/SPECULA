
import os
import yaml
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from astropy.io import fits

from specula import main_simul
from specula.param_dict import ParamDict
from specula.lib.calc_psf import calc_psf_geometry


# TODO candidate to be moved into module functions
def _run_simulation_with_params(params_dict: dict, output_dir: Path, verbose: bool=False):
    """
    Common simulation execution logic using minimal temporary file
    """
    if isinstance(params_dict, ParamDict):
        params_dict = params_dict.params

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Computing simulation with parameters to be saved by DataStore in: {output_dir}")

    # Create minimal temporary YAML file. It will still exist after the with statement exits.
    # The delete_on_close parameter could help make it simpler, but it required python 3.12+
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        yaml.dump(params_dict, temp_file, default_flow_style=False, sort_keys=False)
        temp_params_file = temp_file.name

    try:
        main_simul(yml_files=[temp_params_file])
    except Exception as e:
        print(f"Simulation failed: {e}")
        print(f"Check DataStore output in: {output_dir}")
        print(f"Temp params file for debugging: {temp_params_file}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_params_file)
        except:
            pass  # File cleanup failure is not critical
            
            
class FieldAnalyser:
    """
    Class to analyze field PSF, modal analysis, and phase cubes
    for a given tracking number in the Specula framework.
    This class replicates the functionality of the previous compute_off_axis_psf,
    compute_off_axis_modal_analysis, and compute_off_axis_cube methods,
    providing a structured way to handle field sources and their analysis.
    Attributes:
        data_dir (str): Directory containing tracking number data.
        tracking_number (str): The tracking number for the analysis.
        polar_coordinates (np.ndarray): Polar coordinates of field sources.
        wavelength_nm (float): Wavelength in nanometers.
        start_time (float): Start time for the analysis.
        end_time (Optional[float]): End time for the analysis, if applicable.
        verbose (bool): Whether to print verbose output during processing.
    """

    def __init__(self,
                 data_dir: str,
                 tracking_number: str,
                 polar_coordinates: np.ndarray,
                 wavelength_nm: float = 750.0,
                 start_time: float = 0.1,
                 end_time: Optional[float] = None,
                 verbose: bool = False):

        self.data_dir = Path(data_dir)
        self.tracking_number = tracking_number
        self.polar_coordinates = np.atleast_2d(polar_coordinates)
        self.wavelength_nm = wavelength_nm
        self.start_time = start_time
        self.end_time = end_time
        self.verbose = verbose

        # Loaded parameters
        self.params = None
        self.sources = []
        self.distances = []

        # Paths - modify to create separate directories
        self.tn_dir = self.data_dir / tracking_number
        self.base_output_dir = self.data_dir  # Base directory for analysis results

        # Create separate directories for each analysis type
        self.psf_output_dir = self.base_output_dir / f"{tracking_number}_PSF"
        self.modal_output_dir = self.base_output_dir / f"{tracking_number}_MA"
        self.cube_output_dir = self.base_output_dir / f"{tracking_number}_CUBE"

        # Verify that the tracking number directory exists
        if not self.tn_dir.exists():
            raise FileNotFoundError(f"Tracking number directory not found: {self.tn_dir}")

        self._load_simulation_params()
        self._setup_sources()

    def _load_simulation_params(self):
        """Load simulation parameters from tracking number"""
        params_file = self.tn_dir / "params.yml"
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        self.params = ParamDict()
        self.params.load(params_file)

    def _setup_sources(self):
        """Setup field sources"""
        if self.polar_coordinates.shape[0] == 2:
            # Format: [[r1, r2, ...], [theta1, theta2, ...]]
            coords = self.polar_coordinates.T
        else:
            # Format: [[r1, theta1], [r2, theta2], ...]
            coords = self.polar_coordinates

        for r, theta in coords:
            source_dict = {
                'polar_coordinates': [float(r), float(theta)],
                'height': float('inf'),  # star
                'magnitude': 8,
                'wavelengthInNm': self.wavelength_nm
            }
            self.sources.append(source_dict)
            self.distances.append(r)

    def _get_psf_filenames(self, source_dict: dict) -> Tuple[str, str]:
        """
        Generate PSF and SR filenames for a given source

        Args:
            source: source parameters dictionary
            pixel_size_mas: PSF pixel size in milliarcseconds
            
        Returns:
            Tuple of (psf_filename, sr_filename) without .fits extension
        """
        r, theta = source_dict['polar_coordinates']
        psf_filename = f"psf_r{r:.1f}t{theta:.1f}_pix{self.psf_pixel_size_mas:.2f}mas_wl{self.wavelength_nm:.0f}nm"
        sr_filename = f"sr_r{r:.1f}t{theta:.1f}_pix{self.psf_pixel_size_mas:.2f}mas_wl{self.wavelength_nm:.0f}nm"
        return psf_filename, sr_filename

    def _get_modal_filename(self, source_dict: dict, modal_params: dict) -> str:
        """
        Generate modal analysis filename for a given source
        
        Args:
            source: source parameters dictionary
            modal_params: Modal analysis parameters
            
        Returns:
            Filename without .fits extension
        """
        r, theta = source_dict['polar_coordinates']
        modal_filename = f"modal_r{r:.1f}t{theta:.1f}"

        # Add modal parameters to filename
        if 'nmodes' in modal_params:
            modal_filename += f"_nmodes{modal_params['nmodes']}"
        elif 'nzern' in modal_params:
            modal_filename += f"_nzern{modal_params['nzern']}"

        if 'type_str' in modal_params:
            modal_filename += f"_{modal_params['type_str']}"

        if 'obsratio' in modal_params:
            modal_filename += f"_obs{modal_params['obsratio']:.2f}"

        return modal_filename

    def _get_cube_filename(self, source_dict: dict) -> str:
        """
        Generate phase cube filename for a given source

        Args:
            source_idx: Index of the source

        Returns:
            Filename without .fits extension
        """
        r, theta = source_dict['polar_coordinates']
        cube_filename = f"cube_r{r:.1f}t{theta:.1f}_wl{self.wavelength_nm:.0f}nm"
        return cube_filename

    def _build_replay_params_from_datastore(self) -> dict:
        """
        Build replay params using the existing build_replay mechanism in ParamsDict,
        making sure that a propagation object is retained together with all its inputs.
        """
        return self.params.build_targeted_replay('prop', set_store_dir=str(self.tn_dir))

    def _build_replay_params_psf(self) -> dict:
        """
        Build replay_params for field PSF calculation using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        if self.verbose:
            print(f"Base replay_params keys: {list(replay_params.keys())}")

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Add PSF objects for each field source
        psf_input_list = []
        for i, source_dict in enumerate(self.sources):
            psf_name = f'psf_field_{i}'

            # Build PSF config with pixel_size_mas
            psf_config = {
                'class': 'PSF',
                'simul_params_ref': 'main',
                'wavelengthInNm': self.wavelength_nm,
                'pixel_size_mas': self.psf_pixel_size_mas,
                'start_time': self.start_time,
                'inputs': {
                    'in_ef': f'prop.out_field_source_{i}_ef'
                },
                'outputs': ['out_int_psf', 'out_int_sr']
            }

            replay_params[psf_name] = psf_config

            # Create input_list entries with desired filenames
            psf_filename, sr_filename = self._get_psf_filenames(source_dict)
            psf_input_list.extend([
                f'{psf_filename}-{psf_name}.out_int_psf',
                f'{sr_filename}-{psf_name}.out_int_sr'
            ])

        # Add DataStore to save PSF results
        replay_params['data_store_psf'] = {
            'class': 'DataStore',
            'store_dir': str(self.psf_output_dir),
            'data_format': 'fits',
            'create_tn': False,  # Use existing directory structure
            'inputs': {
                'input_list': psf_input_list
            }
        }

        if self.verbose:
            print(f"Final replay_params keys: {list(replay_params.keys())}")
            print(f"PSF files to be saved: {psf_input_list}")

        return replay_params

    def _build_replay_params_modal(self, modal_params: dict) -> dict:
        """
        Build replay_params for field modal analysis using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Create simple IFunc with modal_params (let ModalAnalysis handle the complexity)
        ifunc_config = {
            'class': 'IFunc',
            'type_str': modal_params.get('type_str', 'zernike'),
            'nmodes': modal_params.get('nmodes', modal_params.get('nzern', 100)),
            'npixels': modal_params.get('npixels', replay_params['main']['pixel_pupil'])
        }

        # Add optional parameters if present
        for param in ['obsratio', 'diaratio', 'start_mode', 'idx_modes']:
            if param in modal_params:
                ifunc_config[param] = modal_params[param]

        replay_params['modal_analysis_ifunc'] = ifunc_config

        # Add ModalAnalysis for each source
        modal_input_list = []
        for i, source_dict in enumerate(self.sources):
            modal_name = f'modal_analysis_{i}'
            modal_config = {
                'class': 'ModalAnalysis',
                'ifunc_ref': 'modal_analysis_ifunc',
                'inputs': {'in_ef': f'prop.out_field_source_{i}_ef'},
                'outputs': ['out_modes']
            }

            # Add ModalAnalysis-specific parameters
            for param in ['dorms', 'wavelengthInNm']:
                if param in modal_params:
                    modal_config[param] = modal_params[param]

            replay_params[modal_name] = modal_config

            # Create filename for this source
            modal_filename = self._get_modal_filename(source_dict, modal_params)
            modal_input_list.append(f'{modal_filename}-{modal_name}.out_modes')

        # Add DataStore to save results
        replay_params['data_store_modal'] = {
            'class': 'DataStore',
            'store_dir': str(self.modal_output_dir),
            'data_format': 'fits',
            'create_tn': False,
            'inputs': {
                'input_list': modal_input_list
            }
        }

        if self.verbose:
            print(f"Modal files to be saved: {modal_input_list}")

        return replay_params

    def _build_replay_params_cube(self) -> dict:
        """
        Build replay_params for field phase cubes using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Build input_list for phase cubes
        cube_input_list = []
        for i, source_dict in enumerate(self.sources):
            cube_filename = self._get_cube_filename(source_dict)
            cube_input_list.append(f'{cube_filename}-prop.out_field_source_{i}_ef')

        # Add DataStore to save phase cubes
        replay_params['data_store_cube'] = {
            'class': 'DataStore',
            'store_dir': str(self.cube_output_dir),
            'data_format': 'fits',
            'create_tn': False,  # Use existing directory structure
            'inputs': {
                'input_list': cube_input_list
            }
        }

        if self.verbose:
            print(f"Cube files to be saved: {cube_input_list}")

        return replay_params

    def _add_field_sources_to_params(self, replay_params: dict):
        """
        Add field sources and update propagation object
        """
        # Find the propagation object
        prop_key, prop_config = replay_params.get_by_class('AtmoPropagation')

        if self.verbose:
            print(f"Found propagation object: '{prop_key}'")

        # Add field sources
        for i, source_dict in enumerate(self.sources):
            source_name = f'field_source_{i}'
            replay_params[source_name] = {
                'class': 'Source',
                'polar_coordinates': source_dict['polar_coordinates'],
                'magnitude': source_dict['magnitude'],
                'wavelengthInNm': source_dict['wavelengthInNm'],
                'height': source_dict['height']
            }

        # Update propagation to use our sources only
        source_refs = [f'field_source_{i}' for i in range(len(self.sources))]
        prop_config['source_dict_ref'] = source_refs

        # and the corresponding ef outputs.
        output_list = [f'out_field_source_{i}_ef' for i in range(len(self.sources))]
        prop_config['outputs'] = output_list

        if self.verbose:
            print(f"Updated propagation object '{prop_key}':")
            print(f"  Sources: {source_refs}")
            print(f"  Outputs: {output_list}")

    def compute_field_psf(self,
                        psf_sampling: Optional[float] = None, 
                        psf_pixel_size_mas: Optional[float] = None,
                        force_recompute: bool = False) -> Dict:
        """
        Calculate field PSF using SPECULA's replay system
        
        Args:
            psf_sampling: PSF sampling factor (alternative to psf_pixel_size_mas)
            psf_pixel_size_mas: Desired PSF pixel size in milliarcseconds (alternative to psf_sampling)
            force_recompute: Force recomputation even if files exist
            
        Note:
            Either psf_sampling or psf_pixel_size_mas must be specified, but not both.
        """

        # Validate input parameters
        if psf_sampling is not None and psf_pixel_size_mas is not None:
            raise ValueError("Cannot specify both psf_sampling and psf_pixel_size_mas. Choose one.")

        if psf_sampling is None and psf_pixel_size_mas is None:
            psf_sampling = 7.0

        # Get simul_params from main configuration
        _, main_config = self.params.get_by_class('SimulParams')

        pixel_pitch = main_config['pixel_pitch']
        pixel_pupil = main_config['pixel_pupil']

        psf_geometry = calc_psf_geometry(
                                    pixel_pupil,
                                    pixel_pitch,
                                    self.wavelength_nm,
                                    nd=psf_sampling,
                                    pixel_size_mas=psf_pixel_size_mas)
        
        self.psf_sampling = psf_geometry.nd
        self.psf_pixel_size_mas = psf_geometry.pixel_size_mas

        # Check if all individual PSF files exist
        all_exist = True
        if not force_recompute:
            for source_dict in self.sources:
                psf_filename, sr_filename = self._get_psf_filenames(source_dict)
                psf_path = self.psf_output_dir / f"{psf_filename}.fits"
                sr_path = self.psf_output_dir / f"{sr_filename}.fits"

                if not psf_path.exists() or not sr_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing PSF results from: {self.psf_output_dir}")
                return self._load_psf_results()

        if self.verbose:
            print(f"Computing field PSF for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_psf()
        _run_simulation_with_params(replay_params, self.psf_output_dir, verbose=self.verbose)

        if self.verbose:
            print(f"Actual PSF pixel size: {self.psf_pixel_size_mas:.2f} mas")

        # Extract results from DataStore (files are automatically saved)
        results = self._load_psf_results()

        return results

    def compute_modal_analysis(self, modal_params: Optional[Dict] = None, force_recompute: bool = False) -> Dict:
        """
        Calculate field modal analysis using replay system

        Args:
            modal_params: Simple dictionary with basic parameters:
                        - type_str: 'zernike', 'kl', etc. (default: 'zernike')
                        - nmodes/nzern: number of modes (default: 100)
                        - obsratio, diaratio: pupil parameters (optional)
                        - dorms: compute RMS flag (optional)
                        If None, attempts to extract from DM configuration
            force_recompute: Force recomputation even if files exist
        """
        if modal_params is None:
            modal_params = self._extract_modal_params_from_dm()

        # Validate and set defaults
        if 'nmodes' not in modal_params and 'nzern' not in modal_params:
            modal_params['nmodes'] = 100
        if 'type_str' not in modal_params:
            modal_params['type_str'] = 'zernike'

        # Check if files exist
        all_exist = True
        if not force_recompute:
            for source_dict in self.sources:
                modal_filename = self._get_modal_filename(source_dict, modal_params)
                modal_path = self.modal_output_dir / f"{modal_filename}.fits"
                if not modal_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing modal analysis from: {self.modal_output_dir}")
                return self._load_modal_results(modal_params)

        if self.verbose:
            print(f"Computing field modal analysis for {len(self.sources)} sources...")
            print(f"Modal parameters: {modal_params}")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_modal(modal_params)
        _run_simulation_with_params(replay_params, self.modal_output_dir, verbose=self.verbose)

        # Extract results from DataStore (files are automatically saved)
        results = self._load_modal_results(modal_params)

        return results

    def compute_phase_cube(self, force_recompute: bool = False) -> Dict:
        """Calculate field phase cubes using replay system"""

        # Check if all individual cube files exist
        all_exist = True
        if not force_recompute:
            for source_dict in self.sources:
                cube_filename = self._get_cube_filename(source_dict)
                cube_path = self.cube_output_dir / f"{cube_filename}.fits"

                if not cube_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing phase cubes from: {self.cube_output_dir}")
                return self._load_cube_results()

        if self.verbose:
            print(f"Computing field phase cubes for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_cube()
        _run_simulation_with_params(replay_params, self.cube_output_dir, verbose=self.verbose)

        # Extract results from DataStore (files are automatically saved)
        results = self._load_cube_results()

        return results

    def _load_psf_results(self) -> Dict:
        """Extract PSF results from DataStore files"""
        results = {
            'psf_list': [],
            'sr_list': [],
            'distances': self.distances,
            'coordinates': self.polar_coordinates,
            'wavelength_nm': self.wavelength_nm,
            'pixel_size_mas': self.psf_pixel_size_mas,
            'psf_sampling': self.psf_sampling
        }

        # Load PSF and SR data from saved files
        for source_dict in self.sources:
            psf_filename, sr_filename = self._get_psf_filenames(source_dict)

            # Load PSF
            psf_path = self.psf_output_dir / f"{psf_filename}.fits"
            with fits.open(psf_path) as hdul:
                results['psf_list'].append(hdul[0].data)

            # Load SR
            sr_path = self.psf_output_dir / f"{sr_filename}.fits"
            with fits.open(sr_path) as hdul:
                results['sr_list'].append(hdul[0].data)

        return results

    def _load_modal_results(self, modal_params: dict) -> Dict:
        """Load existing modal results from DataStore files"""
        results = {
            'modal_coeffs': [],
            'residual_variance': [],
            'residual_average': [],
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm,
            'modal_params': modal_params
        }

        for source_dict in self.sources:
            modal_filename = self._get_modal_filename(source_dict, modal_params)
            modal_path = self.modal_output_dir / f"{modal_filename}.fits"

            with fits.open(modal_path) as hdul:
                modal_coeffs = hdul[0].data
                results['modal_coeffs'].append(modal_coeffs)

                # Calculate statistics from time series
                if len(modal_coeffs) > 0:
                    # Filter by time if needed (assuming first dimension is time)
                    results['residual_average'].append(np.mean(modal_coeffs, axis=0))
                    results['residual_variance'].append(np.var(modal_coeffs, axis=0))
                else:
                    results['residual_average'].append(np.zeros(modal_coeffs.shape[1]))
                    results['residual_variance'].append(np.zeros(modal_coeffs.shape[1]))

        return results

    def _load_cube_results(self) -> Dict:
        """Load existing cube results from DataStore files"""
        results = {
            'phase_cubes': [],
            'times': None,
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm
        }

        for source_dict in self.sources:
            cube_filename = self._get_cube_filename(source_dict)
            cube_path = self.cube_output_dir / f"{cube_filename}.fits"

            with fits.open(cube_path) as hdul:
                results['phase_cubes'].append(hdul[0].data)

                if results['times'] is None and len(hdul) > 1:
                    results['times'] = hdul[1].data

        return results

    def _extract_modal_params_from_dm(self) -> Dict:
        """
        Extract modal parameters from DM configuration with simple fallback
        """
        # Try to find a DM with height=0 and extract basic parameters
        if self.params is None:
            return {'type_str': 'zernike', 'nmodes': 100}

        # Look for DM with height=0
        for obj_name, obj_config in self.params.filter_by_class('DM'):
            if obj_config.get('height', None) == 0:
                # Extract simple parameters
                modal_params = {}

                # Direct copy of relevant parameters
                for param in ['type_str', 'nmodes', 'nzern', 'obsratio', 'diaratio']:
                    if param in obj_config:
                        modal_params[param] = obj_config[param]

                # If we have an ifunc_ref, try to get nmodes from it
                if 'ifunc_ref' in obj_config and obj_config['ifunc_ref'] in self.params:
                    ifunc_config = self.params[obj_config['ifunc_ref']]
                    if isinstance(ifunc_config, dict):
                        for param in ['nmodes', 'nzern', 'type_str', 'obsratio']:
                            if param in ifunc_config and param not in modal_params:
                                modal_params[param] = ifunc_config[param]

                # Ensure we have basic parameters
                if 'nmodes' not in modal_params and 'nzern' not in modal_params:
                    modal_params['nmodes'] = 100
                if 'type_str' not in modal_params:
                    modal_params['type_str'] = 'zernike'

                if self.verbose:
                    print(f"Extracted modal parameters from DM '{obj_name}': {modal_params}")

                return modal_params

        # Fallback to defaults
        if self.verbose:
            print("No suitable DM found, using default modal parameters")

        return {'type_str': 'zernike', 'nmodes': 100}