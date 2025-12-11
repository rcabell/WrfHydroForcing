from dataclasses import InitVar, dataclass, field
from datetime import datetime
import os
from pprint import pprint
import tomllib

from forcing_engine import inputs
from forcing_engine.exceptions import FEConfigException

############# DATA CLASSES FOR CONFIGURATION OPTIONS #############
# TODO: move these to their own files later

@dataclass
class ConfigBase:
    def __post_init__(self):
        self.validate()

    def validate(self):
        pass

@dataclass
class InputDependent(ConfigBase):
    class InputConfig: pass
    input_ref: InitVar[InputConfig]

    def __post_init__(self, input_ref: InputConfig):
        self.num_forcings = len(input_ref)
        self.validate()

@dataclass
class InputConfig(ConfigBase):
    forcing_types: list[str]
    forcing_filetypes: list[str]
    directories: list[str]
    input_mandatory: list[bool]

    forcings: list = field(default_factory=list)

    def validate(self):
        # validate input configs lengths
        if len(self.forcing_types) == 0:
            raise FEConfigException("At least one forcing type must be specified in ForcingTypes.")
        if len(self.forcing_types) != len(self.forcing_filetypes):
            raise FEConfigException("ForcingTypes and ForcingFileTypes lists must be of the same length.")
        if len(self.directories) != len(self.forcing_types):
            raise FEConfigException("InputDirectories list must match the length of forcing types.")

        if len(self.input_mandatory) != len(self.forcing_types):
            if len(self.input_mandatory) == 0:
                # default all to True
                self.input_mandatory = [True] * len(self.forcing_types)
            else:
                raise FEConfigException("InputMandatory list must match the length of forcing types.")

        for ftype in self.forcing_filetypes:
            allowed_types = ['GRIB1', 'GRIB2', 'NETCDF']
            if ftype not in allowed_types:
                raise FEConfigException(f"Unsupported forcing file type specified: {ftype}. "
                                        f"Supported types are: {allowed_types}")

        # look for forcing input classes
        self.forcings = [inputs.find(ftype) for ftype in self.forcing_types]
        if None in self.forcings:
            missing = [
                ftype for ftype, fclass in zip(self.forcing_types, self.forcings) if fclass is None
            ]
            raise FEConfigException(f"Unsupported forcing types specified: {missing}")

        # check for input directories
        for idx, dir_path in enumerate(self.directories):
            if not os.path.exists(dir_path) and self.input_mandatory[idx]:
                raise FEConfigException(f"Input forcing directory '{dir_path}' does not exist "
                                        f"and is marked as mandatory.")

    def __len__(self):
        return len(self.forcings)

@dataclass
class OutputConfig(ConfigBase):
    frequency: int
    directory: str
    scratch_dir: str
    compress: bool = False
    floating_point: bool = True
    include_lqfrac: bool = False
    suboutput_hour: int = 0
    suboutput_frequency: int = 0

    def validate(self):
        if self.frequency <= 0:
            raise FEConfigException("Output frequency must be a positive integer representing minutes.")
        if self.suboutput_frequency < 0:
            raise FEConfigException("Sub-output frequency must be a non-negative integer representing minutes.")
        if self.suboutput_hour < 0:
            raise FEConfigException("Sub-output hour must be a non-negative integer representing hours.")
        if not self.directory:
            raise FEConfigException("Output directory must be specified.")
        if not self.scratch_dir:
            self.scratch_dir = self.directory       # default scratch to output dir
        if os.path.exists(self.directory) is False:
            raise FEConfigException(f"Output directory '{self.directory}' does not exist.")
        if os.path.exists(self.scratch_dir) is False:
            raise FEConfigException(f"Scratch directory '{self.scratch_dir}' does not exist.")

@dataclass
class ForecastConfig(InputDependent):
    start_date: datetime
    end_date: datetime
    frequency_hours: int
    forecast_horizons_minutes: list[int]
    forecast_offsets_minutes: list[int] = field(default_factory=list)
    shift_hours: int = 0
    ana_flag: bool = False
    looback_hours: int = -9999

    def validate(self):
        if len(self.forecast_horizons_minutes) != self.num_forcings:
            raise FEConfigException("ForecastHorizonsMinutes length must match number of input forcing types.")
        if len(self.forecast_offsets_minutes) != self.num_forcings:
            if len(self.forecast_offsets_minutes) == 0:
                self.forecast_offsets_minutes = [0] * self.num_forcings
            else:
                raise FEConfigException("ForecastOffsetsMinutes length must match number of input forcing types.")

        if self.start_date >= self.end_date:
            raise FEConfigException("Forecast start date must be earlier than end date.")
        if self.frequency_hours <= 0:
            raise FEConfigException("Forecast frequency must be a positive integer representing hours.")
        if any(h < 0 for h in self.forecast_horizons_minutes):
            raise FEConfigException("All forecast horizons must be non-negative integers representing minutes.")
        if any(o < 0 for o in self.forecast_offsets_minutes):
            raise FEConfigException("All forecast offsets must be non-negative integers representing minutes.")
        if self.shift_hours < 0:
            raise FEConfigException("Forecast shift hours must be a non-negative integer.")
        if self.looback_hours > 0:
            raise FEConfigException("Lookback hours must be non-positive (zero or negative).")

@dataclass
class GeospatialConfig(InputDependent):
    geogrid_file: str
    spatial_metadata_file: str
    ignored_border_widths: list[int]

    def validate(self):
        if len(self.ignored_border_widths) != self.num_forcings:
            raise FEConfigException("IgnoredBorderWidths length must match number of input forcing types.")
        if any(w < 0 for w in self.ignored_border_widths):
            raise FEConfigException("All ignored border widths must be non-negative integers.")

@dataclass
class RegriddingConfig(InputDependent):
    methods: list[str]

    def validate(self):
        if len(self.methods) != self.num_forcings:
            raise FEConfigException("Regridding methods length must match number of input forcing types.")
        for method in self.methods:
            allowed_methods = ['Bilinear', 'NearestNeighbor', 'Conservative']
            if method not in allowed_methods:
                raise FEConfigException(f"Unsupported regridding method: {method}. "
                                        f"Supported methods are: {allowed_methods}")

@dataclass
class InterpolationConfig(InputDependent):
    methods: list[str]

    def validate(self):
        if len(self.methods) != self.num_forcings:
            raise FEConfigException("Interpolation methods length must match number of input forcing types.")
        for method in self.methods:
            allowed_methods = ['None', 'NearestNeighbor', 'Linear']
            if method not in allowed_methods:
                raise FEConfigException(f"Unsupported interpolation method: {method}. "
                                        f"Supported methods are: {allowed_methods}")

@dataclass
class BiasCorrectionConfig(InputDependent):
    temperature_methods: list[str]
    surface_pressure_methods: list[str]
    humidity_methods: list[str]
    wind_methods: list[str]
    shortwave_methods: list[str]
    longwave_methods: list[str]
    precipitation_methods: list[str]

    def validate(self):
        n_forcings = self.num_forcings
        for attr_name in [
            'temperature_methods', 'surface_pressure_methods', 'humidity_methods',
            'wind_methods', 'shortwave_methods', 'longwave_methods', 'precipitation_methods'
        ]:
            methods = getattr(self, attr_name)
            if len(methods) != n_forcings:
                friendly_name = attr_name.replace('_methods', '').replace('_', ' ').title()
                raise FEConfigException(f"{friendly_name} bias correction methods length "
                                        f"must match number of input forcing types.")

        # Supported methods for each variable
        supported_methods = {
            'temperature_methods': ['None', 'CFS', 'HRRR-analysis', 'GFS', 'HRRR-parametric'],
            'surface_pressure_methods': ['None', 'CFS'],
            'humidity_methods': ['None', 'CFS', 'HRRR-analysis'],
            'wind_methods': ['None', 'CFS', 'HRRR-analysis', 'GFS', 'HRRR-parametric'],
            'shortwave_methods': ['None', 'CFS', 'HRRR-analysis'],
            'longwave_methods': ['None', 'CFS', 'HRRR-analysis', 'GFS'],
            'precipitation_methods': ['None', 'CFS'],
        }

        for attr_name, allowed in supported_methods.items():
            methods = getattr(self, attr_name)
            for method in methods:
                if method not in allowed:
                    friendly_name = attr_name.replace('_methods', '').replace('_', ' ').title()
                    raise FEConfigException(
                        f"Unsupported bias correction method '{method}' for {friendly_name}. "
                        f"Supported methods: {allowed}"
                    )

@dataclass
class DownscalingConfig(InputDependent):
    temperature_methods: list[str]
    surface_pressure_methods: list[str]
    shortwave_methods: list[str]
    precipitation_methods: list[str]
    humidity_methods: list[str]
    params_directory: str

    def validate(self):
        n_forcings = self.num_forcings
        for attr_name in [
            'temperature_methods', 'surface_pressure_methods', 'humidity_methods',
            'shortwave_methods', 'precipitation_methods'
        ]:
            methods = getattr(self, attr_name)
            if len(methods) != n_forcings:
                friendly_name = attr_name.replace('_methods', '').replace('_', ' ').title()
                raise FEConfigException(f"{friendly_name} downscaling methods length must match "
                                        f"number of input forcing types.")

        # Supported methods for each variable
        supported_methods = {
            'temperature_methods': ['None', 'Simple', 'Calculated', 'Dynamic'],
            'surface_pressure_methods': ['None', 'Elevation'],
            'shortwave_methods': ['None', 'Topographic'],
            'precipitation_methods': ['None', 'PRISM'],
            'humidity_methods': ['None', 'Classic'],
        }
        param_dir_required = False
        for attr_name, allowed in supported_methods.items():
            methods = getattr(self, attr_name)
            for method in methods:
                if method in ['Calculated', 'PRISM']:
                    param_dir_required = True
                if method not in allowed:
                    friendly_name = attr_name.replace('_methods', '').replace('_', ' ').title()
                    raise FEConfigException(
                        f"Unsupported downscaling method '{method}' for {friendly_name}. "
                        f"Supported methods: {allowed}"
                    )

         # if params_directory is required, check if it exists
        if param_dir_required:
            if not os.path.exists(self.params_directory):
                raise FEConfigException(f"Downscaling parameters directory '{self.params_directory}' does not exist "
                                        f"but is required by the selected downscaling method(s).")

@dataclass
class SupplementalForcingConfig(ConfigBase):
    forcing_types: list[str] = field(default_factory=list)
    forcing_filetypes: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    input_mandatory: list[bool] = field(default_factory=list)
    regrid_methods: list[str] = field(default_factory=list)
    interpolation_methods: list[str] = field(default_factory=list)
    input_offsets_hours: list[int] = field(default_factory=list)
    rqi_method: str = "None"
    rqi_threshold: float = 0.0
    param_dir: str = ""

    forcings: list = field(default_factory=list)

    def validate(self):
        n_forcings = len(self.forcing_types)
        if len(self.forcing_filetypes) != n_forcings:
            raise FEConfigException("SupplementalForcingFileTypes length must match number of forcing types.")
        if len(self.directories) != n_forcings:
            raise FEConfigException("SupplementalForcingDirectories length must match number of forcing types.")
        if len(self.input_mandatory) != n_forcings:
            raise FEConfigException("SupplementalInputMandatory length must match number of forcing types.")
        if len(self.regrid_methods) != n_forcings:
            raise FEConfigException("SupplementalRegridMethods length must match number of forcing types.")
        if len(self.interpolation_methods) != n_forcings:
            raise FEConfigException("SupplementalInterpolationMethods length must match number of forcing types.")
        if len(self.input_offsets_hours) != n_forcings:
            if len(self.input_offsets_hours) == 0:
                self.input_offsets_hours = [0] * n_forcings
            else:
                raise FEConfigException("SupplementalInputOffsetsHours length must match number of forcing types.")

        # look for forcing input classes
        self.forcings = [inputs.find(ftype) for ftype in self.forcing_types]

        if len(self.input_mandatory) != len(self.forcing_types):
            if len(self.input_mandatory) == 0:
                self.input_mandatory = [True] * len(self.forcing_types)   # default all to True
            else:
                raise FEConfigException("SupplementalInputMandatory list must match the length of forcing types.")

        # ensure vailid file types
        for ftype in self.forcing_filetypes:
            allowed_types = ['GRIB1', 'GRIB2', 'NETCDF']
            if ftype not in allowed_types:
                raise FEConfigException(f"Unsupported suuplemental forcing file type specified: {ftype}. "
                                        f"Supported types are: {allowed_types}")
        # check for input directories
        for idx, dir_path in enumerate(self.directories):
            if not os.path.exists(dir_path) and self.input_mandatory[idx]:
                raise FEConfigException(f"Supplemental forcing input directory '{dir_path}' does not exist "
                                        f"and is marked as mandatory.")

        # ensure valid regridding methods
        for method in self.regrid_methods:
            allowed_methods = ['Bilinear', 'NearestNeighbor', 'Conservative']
            if method not in allowed_methods:
                raise FEConfigException(f"Unsupported supplemental regridding method: {method}. "
                                        f"Supported methods are: {allowed_methods}")

        # ensure valid interpolation methods
        for method in self.interpolation_methods:
            allowed_methods = ['None', 'NearestNeighbor', 'Linear']
            if method not in allowed_methods:
                raise FEConfigException(f"Unsupported supplemental interpolation method: {method}. "
                                        f"Supported methods are: {allowed_methods}")

        # input offsets validation should be positive integers
        for offset in self.input_offsets_hours:
            if offset < 0:
                raise FEConfigException("All supplemental input offsets must be non-negative integers representing hours.")

        # validate RQI method
        allowed_rqi_methods = ['None', 'MRMS', 'NWM']
        if self.rqi_method not in allowed_rqi_methods:
            raise FEConfigException(f"Unsupported RQI method: {self.rqi_method}. "
                                    f"Supported methods are: {allowed_rqi_methods}")

        # RQI threshold should be between 0 and 1
        if not (0.0 <= self.rqi_threshold <= 1.0):
            raise FEConfigException("RQI threshold must be between 0.0 and 1.0.")

        # if RQI is MRMS, param_dir must exist
        if self.rqi_method == 'MRMS':
            if not os.path.exists(self.param_dir):
                raise FEConfigException(f"Supplemental forcing parameter directory '{self.param_dir}' does not exist "
                                        f"but is required by the selected RQI method.")

    def __len__(self):
        return len(self.forcings)

#######################################################################

class ConfigOptions:
    def __init__(self, config_file, config_name=None, version=None):
        if config_file is not None:
            # TODO: check file type, assume TOML for now but could also support JSON/YAML later
            self.config_file = config_file
            self.config_type = "TOML"
            self.create_from_toml(config_file)
        else:
            # Placeholder for handling a runtime-constructed configuration
            raise FEConfigException("Runtime-defined configuration not yet implemented.")

        if config_name is not None:
            self.meta_config = config_name
        if version is not None:
            self.version = version

    def create_from_toml(self, config_file):
        config_data = tomllib.load(open(config_file, 'rb'))

        input_table = config_data.get('Input', {})
        self.input = InputConfig(
            forcing_types=input_table.get('InputForcings', []),
            forcing_filetypes=input_table.get('InputForcingTypes', []),
            directories=input_table.get('InputForcingDirectories', []),
            input_mandatory=input_table.get('InputMandatory', []),
        )

        output_table = config_data.get('Output', {})
        self.output = OutputConfig(
            frequency=output_table.get('OutputFrequency', 60),
            suboutput_hour=output_table.get('SubOutputHour', 0),
            suboutput_frequency=output_table.get('SubOutFreq', 0),
            directory=output_table.get('OutDir'),
            scratch_dir=output_table.get('ScratchDir'),
            compress=output_table.get('compressOutput', False),
            floating_point=output_table.get('floatOutput', True),
            include_lqfrac=output_table.get('includeLQFrac', False),
        )

        forecast_table = config_data.get('Forecast', {})
        self.forecast = ForecastConfig(
            ana_flag=forecast_table.get('AnaFlag', False),
            start_date=forecast_table.get('RefcstBDateProc'),
            end_date=forecast_table.get('RefcstEDateProc'),
            frequency_hours=forecast_table.get('ForecastFrequency', 1),
            forecast_horizons_minutes=forecast_table.get('ForecastInputHorizons', []),
            forecast_offsets_minutes=forecast_table.get('ForecastInputOffsets', []),
            looback_hours=forecast_table.get('LookBack', -9999),
            shift_hours=forecast_table.get('ForecastShift', 0),
            input_ref=self.input
        )

        geospatial_table = config_data.get('Geospatial', {})
        self.geospatial = GeospatialConfig(
            geogrid_file=geospatial_table.get('GeogridIn', ''),
            spatial_metadata_file=geospatial_table.get('SpatialMetaIn', ''),
            ignored_border_widths=geospatial_table.get('IgnoredBorderWidths', []),
            input_ref=self.input
        )

        regridding_table = config_data.get('Regridding', {})
        self.regridding = RegriddingConfig(
            methods=regridding_table.get('RegridOpt', []),
            input_ref=self.input
        )

        interpolation_table = config_data.get('Interpolation', {})
        self.interpolation = InterpolationConfig(
            methods=interpolation_table.get('ForcingTemporalInterpolation', []),
            input_ref=self.input
        )

        bias_correction_table = config_data.get('BiasCorrection', {})
        self.bias_correction = BiasCorrectionConfig(
            temperature_methods=bias_correction_table.get('TemperatureBiasCorrection', []),
            surface_pressure_methods=bias_correction_table.get('PressureBiasCorrection', []),
            humidity_methods=bias_correction_table.get('HumidityBiasCorrection', []),
            wind_methods=bias_correction_table.get('WindBiasCorrection', []),
            shortwave_methods=bias_correction_table.get('SwBiasCorrection', []),
            longwave_methods=bias_correction_table.get('LwBiasCorrection', []),
            precipitation_methods=bias_correction_table.get('PrecipBiasCorrection', []),
            input_ref=self.input
        )

        downscaling_table = config_data.get('Downscaling', {})
        self.downscaling = DownscalingConfig(
            temperature_methods=downscaling_table.get('TemperatureDownscaling', []),
            surface_pressure_methods=downscaling_table.get('PressureDownscaling', []),
            shortwave_methods=downscaling_table.get('ShortwaveDownscaling', []),
            precipitation_methods=downscaling_table.get('PrecipDownscaling', []),
            humidity_methods=downscaling_table.get('HumidityDownscaling', []),
            params_directory=downscaling_table.get('DownscalingParamDirs', ''),
            input_ref=self.input
        )

        supplemental_table = config_data.get('SuppForcing', {})
        self.supplemental = SupplementalForcingConfig(
            forcing_types=supplemental_table.get('SupplementalForcingTypes', []),
            forcing_filetypes=supplemental_table.get('SupplementalForcingFileTypes', []),
            directories=supplemental_table.get('SupplementalForcingDirectories', []),
            input_mandatory=supplemental_table.get('SupplementalInputMandatory', []),
            regrid_methods=supplemental_table.get('SupplementalRegridMethods', []),
            interpolation_methods=supplemental_table.get('SupplementalInterpolationMethods', []),
            input_offsets_hours=supplemental_table.get('SupplementalInputOffsetsHours', []),
        )

    def print_config(self):
        [pprint(section) for section in (self.input, self.output, self.forecast, self.geospatial, self.regridding,
                                         self.interpolation, self.bias_correction, self.downscaling, self.supplemental)]
