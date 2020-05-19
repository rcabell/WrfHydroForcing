import os

import numpy as np

from .regridder import Regridder
from core import file_io


class RegridHRRR(Regridder):

    def regrid_input(self, input_forcings):
        # If the expected file is missing, this means we are allowing missing files, simply
        # exit out of this routine as the regridded fields have already been set to NDV.
        if not os.path.isfile(input_forcings.file_in2):
            self.log_status("No HRRR input file found for this timestep", root_only=True)
            return

        # Check to see if the regrid complete flag for this
        # output time step is true. This entails the necessary
        # inputs have already been regridded and we can move on.
        if input_forcings.regridComplete:
            self.log_status("No HRRR regridding required for this timestep", root_only=True)
            return

        # Create a path for a temporary NetCDF files that will
        # be created through the wgrib2 process.
        input_forcings.tmpFile = os.path.join(self.config.scratch_dir, "HRRR_CONUS_TMP.nc")
        input_forcings.tmpFileHeight = os.path.join(self.config.scratch_dir, "HRRR_CONUS_TMP_HEIGHT.nc")

        # This file shouldn't exist, but if it does (previously failed execution of the program), remove it...
        if self.IS_MPI_ROOT:
            if os.path.isfile(input_forcings.tmpFile):
                self.log_status("Found old temporary file: {} - Removing...".format(input_forcings.tmpFile))
                try:
                    os.remove(input_forcings.tmpFile)
                except OSError:
                    self.log_critical("Unable to remove temporary file: " + input_forcings.tmpFile)

        fields = []
        for forcing_idx, grib_var in enumerate(input_forcings.grib_vars):
            self.log_status("Converting CONUS HRRR Variable: {}".format(grib_var), root_only=True)
            fields.append(':' + grib_var + ':' +
                          input_forcings.grib_levels[forcing_idx] + ':'
                          + str(input_forcings.fcst_hour2) + " hour fcst:")
        fields.append(":(HGT):(surface):")      # add HGT variable to main set (no separate file)

        # Create a temporary NetCDF file from the GRIB2 file.
        cmd = '$WGRIB2 -match "(' + '|'.join(fields) + ')" ' + input_forcings.file_in2 + \
              " -netcdf " + input_forcings.tmpFile
        with self.parallel_error_checking():
            datasource = file_io.open_grib2(input_forcings.file_in2, input_forcings.tmpFile, cmd,
                                            self.config, self.mpi, inputVar=None)

        for forcing_idx, grib_var in enumerate(input_forcings.grib_vars):
            self.log_status("Processing Conus HRRR Variable: {}".format(grib_var), root_only=True)

            if self.need_regrid_object(datasource, forcing_idx, input_forcings):
                with self.parallel_error_checking():
                    self.log_status("Calculating HRRR regridding weights", root_only=True)
                    self.calculate_weights(datasource, forcing_idx, input_forcings)

                # Regrid the height variable.
                with self.parallel_error_checking():
                    var_tmp = None
                    if self.IS_MPI_ROOT:
                        try:
                            var_tmp = datasource.variables['HGT_surface'][0, :, :]
                        except (ValueError, KeyError, AttributeError) as err:
                            self.log_critical("Unable to extract HRRR elevation from {}: {}".format(
                                    input_forcings.tmpFile, err))

                        var_local_tmp = self.mpi.scatter_array(input_forcings, var_tmp, self.config)

                with self.parallel_error_checking():
                    try:
                        input_forcings.esmf_field_in.data[:, :] = var_local_tmp
                    except (ValueError, KeyError, AttributeError) as err:
                        self.log_critical("Unable to place input HRRR data into ESMF field: {}".format(err))

                with self.parallel_error_checking():
                    self.log_status("Regridding HRRR surface elevation data to the WRF-Hydro domain.", root_only=True)
                    try:
                        input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                                 input_forcings.esmf_field_out)
                    except ValueError as ve:
                        self.log_critical("Unable to regrid HRRR surface elevation using ESMF: {}".format(ve))

                # Set any pixel cells outside the input domain to the global missing value.
                with self.parallel_error_checking():
                    try:
                        input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                            self.config.globalNdv
                    except (ValueError, ArithmeticError) as npe:
                        self.log_critical("Unable to perform HRRR mask search on elevation data: {}".format(npe))

                    try:
                        input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                    except (ValueError, KeyError, AttributeError) as err:
                        self.log_critical("Unable to extract regridded HRRR elevation data from ESMF: {}".format(err))

            # EXTRACT AND REGRID THE INPUT FIELDS:

            var_tmp = None
            with self.parallel_error_checking():
                if self.IS_MPI_ROOT:
                    self.log_status(
                        "Processing input HRRR variable: {}".format(input_forcings.netcdf_var_names[forcing_idx]))
                    try:
                        var_tmp = datasource.variables[input_forcings.netcdf_var_names[forcing_idx]][0, :, :]
                    except (ValueError, KeyError, AttributeError) as err:
                        self.log_critical(
                            "Unable to extract {} from {} ({})".format(input_forcings.netcdf_var_names[forcing_idx],
                                                                       input_forcings.tmpFile, err))

                var_local_tmp = self.mpi.scatter_array(input_forcings, var_tmp, self.config)

            with self.parallel_error_checking():
                try:
                    input_forcings.esmf_field_in.data[:, :] = var_local_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical("Unable to place input HRRR data into ESMF field: {}".format(err))

            with self.parallel_error_checking():
                self.log_status("Regridding Input HRRR Field: {}".format(input_forcings.netcdf_var_names[forcing_idx]),
                                root_only=True)
                try:
                    input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                             input_forcings.esmf_field_out)
                except ValueError as ve:
                    self.log_critical("Unable to regrid input HRRR forcing data: {}".format(ve))

            # Set any pixel cells outside the input domain to the global missing value.
            with self.parallel_error_checking():
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        self.config.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    self.log_critical("Unable to perform mask test on regridded HRRR forcings: {}".format(npe))

            with self.parallel_error_checking():
                try:
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[forcing_idx], :, :] = \
                        input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    self.log_critical("Unable to extract regridded HRRR forcing data from ESMF field: {}".format(err))

            # If we are on the first timestep, set the previous regridded field to be
            # the latest as there are no states for time 0.
            if self.config.current_output_step == 1:
                input_forcings.regridded_forcings1[input_forcings.input_map_output[forcing_idx], :, :] = \
                    input_forcings.regridded_forcings2[input_forcings.input_map_output[forcing_idx], :, :]

        # Close the temporary NetCDF file and remove it.
        if self.IS_MPI_ROOT:
            try:
                datasource.close()
            except OSError:
                self.log_critical("Unable to close NetCDF file: {}".format(input_forcings.tmpFile))

            try:
                os.remove(input_forcings.tmpFile)
            except OSError:
                # TODO: could this be just a warning?
                self.log_critical("Unable to remove NetCDF file: {}".format(input_forcings.tmpFile))
