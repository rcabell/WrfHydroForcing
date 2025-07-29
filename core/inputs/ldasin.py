from core import err_handler, ioMod, regrid

import numpy as np

import datetime
import os

# INPUT PARAMETERS

PRODUCT_NAME = "ldasin"
CYCLE_FREQ = 60
GRIB_VARS_IN = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']
GRIB_LEVELS_IN = None
GRIB_MSG_INDEX = None
NETCDF_VARS = ['U2D', 'V2D', 'LWDOWN', 'RAINRATE', 'T2D', 'Q2D', 'PSFC', 'SWDOWN', 'LQFRAC']
INPUT_MAP_TO_OUTPUTS = [0,1,2,3,4,5,6,7,8]
FORECAST_HORIZONS = None


def find_ldasin_neighbors(input_forcings, config_options, d_current, mpi_config):
    """
    Function to calculate the previous and next LDASIN cycles based on the current timestep.
    :param input_forcings:
    :param config_options:
    :param d_current:
    :param mpi_config:
    :return:
    """
    if mpi_config.rank == 0:
        config_options.statusMsg = "Processing LDASIN Data. Calculating neighboring " \
                                   "files for this output timestep"
        err_handler.log_msg(config_options, mpi_config)

    # First find the current LDASIN forecast cycle that we are using.
    offset = 1 if config_options.ana_flag else 0
    current_cycle = config_options.current_fcst_cycle - datetime.timedelta(
        seconds=(offset + input_forcings.userCycleOffset) * 60.0)

    # Calculate the current forecast hour within this LDASIN cycle.
    dt_tmp = d_current - current_cycle
    current_hour = int(dt_tmp.days*24) + int(dt_tmp.seconds/3600.0)

    # Calculate the previous file to process.
    min_since_last_output = (current_hour * 60) % 60
    if min_since_last_output == 0:
        min_since_last_output = 60
    prev_date = d_current - datetime.timedelta(seconds=min_since_last_output * 60)
    input_forcings.fcst_date1 = prev_date
    if min_since_last_output == 60:
        min_until_next_output = 0
    else:
        min_until_next_output = 60 - min_since_last_output
    next_date = d_current + datetime.timedelta(seconds=min_until_next_output * 60)
    input_forcings.fcst_date2 = next_date

    # Calculate the output forecast hours needed based on the prev/next dates.
    dt_tmp = next_date - current_cycle
    next_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour2 = next_hour
    dt_tmp = prev_date - current_cycle
    prev_hour = int(dt_tmp.days * 24.0) + int(dt_tmp.seconds / 3600.0)
    input_forcings.fcst_hour1 = prev_hour
    err_handler.check_program_status(config_options, mpi_config)

    # Calculate expected file paths.
    tmp_file1 = os.path.join(input_forcings.inDir, f"{prev_date.strftime('%Y%m%d%H')}00.LDASIN_DOMAIN1")
    if mpi_config.rank == 0:
        config_options.statusMsg = "Previous LDASIN file being used: " + tmp_file1
        err_handler.log_msg(config_options, mpi_config)

    tmp_file2 = tmp_file1
    if mpi_config.rank == 0:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next LDASIN file being used: " + tmp_file2
            err_handler.log_msg(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2:
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
            input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
            input_forcings.file_in1 = tmp_file1
            input_forcings.file_in2 = tmp_file2
        else:
            # Check to see if we are restarting from a previously failed instance. In this case,
            # We are not on the first timestep, but no previous forcings have been processed.
            # We need to process the previous input timestep for temporal interpolation purposes.
            if input_forcings.regridded_forcings1 is None:
                # if not np.any(input_forcings.regridded_forcings1):
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Restarting forecast cycle. Will regrid previous: " + \
                                               input_forcings.productName
                    err_handler.log_msg(config_options, mpi_config)
                input_forcings.rstFlag = 1
                input_forcings.regridded_forcings1 = input_forcings.regridded_forcings1
                input_forcings.regridded_forcings2 = input_forcings.regridded_forcings2
                input_forcings.file_in2 = tmp_file1
                input_forcings.file_in1 = tmp_file1
                input_forcings.fcst_date2 = input_forcings.fcst_date1
                input_forcings.fcst_hour2 = input_forcings.fcst_hour1
            else:
                # The LDASIN window has shifted. Reset fields 2 to
                # be fields 1.
                input_forcings.regridded_forcings1[:, :, :] = input_forcings.regridded_forcings2[:, :, :]
                input_forcings.file_in1 = tmp_file1
                input_forcings.file_in2 = tmp_file2
        input_forcings.regridComplete = False
    err_handler.check_program_status(config_options, mpi_config)

    # Ensure we have the necessary new file
    if mpi_config.rank == 0:
        if not os.path.exists(input_forcings.file_in2):
            if input_forcings.enforce == 1:
                config_options.errMsg = "Expected input LDASIN file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input LDASIN file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv

def regrid_inputs(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    """
    Function for handling regridding of custom input NetCDF hourly forcing files.
    :param input_forcings:
    :param config_options:
    :param wrf_hydro_geo_meta:
    :param mpi_config:
    :return:
    """
    # If the expected file is missing, this means we are allowing missing files, simply
    # exit out of this routine as the regridded fields have already been set to NDV.
    if not os.path.isfile(input_forcings.file_in2):
        return

    # Check to see if the regrid complete flag for this
    # output time step is true. This entails the necessary
    # inputs have already been regridded and we can move on.
    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No Custom Hourly NetCDF regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    fill_values = {'T2D': 288.0, 'Q2D': 0.005, 'PSFC': 101300.0, 'RAINRATE': 0,
                   'U2D': 1.0, 'V2D': 1.0, 'SWDOWN': 80.0, 'LWDOWN': 310.0, 'LQFRAC': 1}

    # Open the input NetCDF file containing necessary data.
    id_tmp, lat_var, lon_var = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config,
                                                         open_on_all_procs=True)

    for force_count, nc_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing Custom NetCDF Forcing Variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config)
        calc_regrid_flag = regrid.check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)

        if calc_regrid_flag:
            regrid.calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, lat_var, lon_var)

            # Read in the height field, which is used for downscaling purposes, if available
            hgt_field = ioMod.get_height_field(id_tmp, config_options, mpi_config)
            if hgt_field is not None:
                var_sub_tmp = mpi_config.scatter_array(input_forcings, hgt_field, config_options)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to place NetCDF elevation data into the ESMF field object: " \
                                            + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Regridding elevation data to the WRF-Hydro domain."
                    err_handler.log_msg(config_options, mpi_config)
                try:
                    input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                             input_forcings.esmf_field_out)
                except ValueError as ve:
                    config_options.errMsg = "Unable to regrid elevation data to the WRF-Hydro domain " \
                                            "using ESMF: " + str(ve)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                # Set any pixel cells outside the input domain to the global missing value.
                try:
                    input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = \
                        config_options.globalNdv
                except (ValueError, ArithmeticError) as npe:
                    config_options.errMsg = "Unable to compute mask on elevation data: " + str(npe)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                try:
                    input_forcings.height[:, :] = input_forcings.esmf_field_out.data
                except (ValueError, KeyError, AttributeError) as err:
                    config_options.errMsg = "Unable to extract ESMF regridded elevation data to a local " \
                                            "array: " + str(err)
                    err_handler.log_critical(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)
            else:
                input_forcings.height = None
                if mpi_config.rank == 0:
                    config_options.statusMsg = f"Unable to locate HGT_surface in: {input_forcings.file_in2}. " \
                                               f"Downscaling will not be available."
                    err_handler.log_msg(config_options, mpi_config)

            # close netCDF file on non-root ranks
            if mpi_config.rank != 0:
                id_tmp.close()

        # Regrid the input variables.
        var_tmp = None
        fill = fill_values.get(input_forcings.grib_vars[force_count], config_options.globalNdv)
        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding Custom netCDF input variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config)
            try:
                # config_options.statusMsg = f"Using {fill} to replace missing values in input"
                # err_handler.log_msg(config_options, mpi_config)
                var_tmp = id_tmp.variables[nc_var][:].filled(fill)[0, :, :]
            except Exception as err:
                config_options.errMsg = "Unable to extract " + nc_var + \
                                        " from: " + input_forcings.file_in2 + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:, :] = var_sub_tmp
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local array into local ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid input Custom netCDF forcing variables using ESMF: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Set any pixel cells outside the input domain to the global missing value.
        try:
            input_forcings.esmf_field_out.data[np.where(input_forcings.regridded_mask == 0)] = fill
        except (ValueError, ArithmeticError) as npe:
            config_options.errMsg = "Unable to calculate mask from input Custom netCDF regridded forcings: " + str(
                npe)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # Convert the hourly precipitation total to a rate of mm/s
        if nc_var == 'APCP_surface' or nc_var == 'PREC_ACC_NC':
            try:
                ind_valid = np.where(input_forcings.esmf_field_out.data != fill)
                input_forcings.esmf_field_out.data[ind_valid] = input_forcings.esmf_field_out.data[
                                                                    ind_valid] / 3600.0
                del ind_valid
            except (ValueError, ArithmeticError, AttributeError, KeyError) as npe:
                config_options.errMsg = "Unable to run NDV search on Custom netCDF precipitation: " + str(npe)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.esmf_field_out.data
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place local ESMF regridded data into local array: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]
        err_handler.check_program_status(config_options, mpi_config)

    # Close the NetCDF file
    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.err_out(config_options)

