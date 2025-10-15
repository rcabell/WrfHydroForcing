try:
   import ESMF
except ImportError:
   import esmpy as ESMF

import glob
import os
import sys
import time
import traceback

import numpy as np
from core import err_handler, ioMod, regrid, time_handling


PRODUCT_NAME = "MPAS"
CYCLE_FREQ = 60
GRIB_VARS_IN = ['u10', 'v10', 'lwdnb', 't2m', 'q2', 'surface_pressure', 'swdnb', 'rainc', 'rainnc']
GRIB_LEVELS_IN = None
GRIB_MSG_INDEX = None
NETCDF_VARS = GRIB_VARS_IN
INPUT_MAP_TO_OUTPUTS = [0,1,2,4,5,6,7,3]
FORECAST_HORIZONS = None

def find_neighbors(input_forcings, config_options, d_current, mpi_config):
    # TODO: support sub-hourly cycling
    current_input_cycle, prev_input_forecast_hour, next_input_forecast_hour = time_handling.calculate_forcing_time(input_forcings, config_options, d_current, mpi_config)

    # TODO: This is brittle and may not span over 00Z correctly... use input forecast datetime
    pattern1 = f"{input_forcings.inDir}/diag_sfc.{current_input_cycle.strftime('%Y-%m-%d')}_{str(prev_input_forecast_hour).zfill(2)}.00.00.nc"
    files1 = glob.glob(pattern1)

    if len(files1) > 0:
        tmp_file1 = files1[0]
        if mpi_config.rank == 0:
            config_options.statusMsg = "Previous input file being used: " + tmp_file1
            err_handler.log_msg(config_options, mpi_config)
    else:
        if mpi_config.rank == 0:
            config_options.errMsg = f"Next input file {pattern1} not found"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    pattern1 = f"{input_forcings.inDir}/diag_sfc.{current_input_cycle.strftime('%Y-%m-%d')}_{str(next_input_forecast_hour).zfill(2)}.00.00.nc"
    files2 = glob.glob(pattern1)

    if len(files2) > 0:
        tmp_file2 = files2[0]
        if mpi_config.rank == 0:
            config_options.statusMsg = "Next input file being used: " + tmp_file2
            err_handler.log_msg(config_options, mpi_config)
    else:
        if mpi_config.rank == 0:
            config_options.errMsg = f"Next input file {pattern1} not found"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Check to see if files are already set. If not, then reset, grids and
    # regridding objects to communicate things need to be re-established.
    if input_forcings.file_in1 != tmp_file1 or input_forcings.file_in2 != tmp_file2 or (config_options.output_freq <= input_forcings.cycleFreq <= 60):
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
                input_forcings.fcst_min2 = input_forcings.fcst_min1
            else:
                # The input window has shifted. Reset fields 2 to
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
                config_options.errMsg = "Expected input file: " + input_forcings.file_in2 + " not found."
                err_handler.log_critical(config_options, mpi_config)
            else:
                config_options.statusMsg = "Expected input file: " + input_forcings.file_in2 + " not found. " \
                                                                                                   "Will not use in " \
                                                                                                   "final layering."
                err_handler.log_warning(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # If the file is missing, set the local slab of arrays to missing.
    if not os.path.exists(input_forcings.file_in2):
        if input_forcings.regridded_forcings2 is not None:
            input_forcings.regridded_forcings2[:, :, :] = config_options.globalNdv

def regrid_inputs(input_forcings, config_options, wrf_hydro_geo_meta, mpi_config):
    if not os.path.isfile(input_forcings.file_in2):
        return

    if input_forcings.regridComplete:
        if mpi_config.rank == 0:
            config_options.statusMsg = "No MPAS regridding required for this timestep."
            err_handler.log_msg(config_options, mpi_config)
        return

    id_tmp = ioMod.open_netcdf_forcing(input_forcings.file_in2, config_options, mpi_config,
                                                         open_on_all_procs=True)

    for force_count, nc_var in enumerate(input_forcings.grib_vars):
        if mpi_config.rank == 0:
            config_options.statusMsg = "Processing MPAS Variable: " + nc_var
            err_handler.log_msg(config_options, mpi_config)
        calc_regrid_flag = regrid.check_regrid_status(id_tmp, force_count, input_forcings,
                                               config_options, wrf_hydro_geo_meta, mpi_config)

        if calc_regrid_flag:
            calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, fill=True)

        # TODO: get height from native MPAS mesh to support downscaling etc

        var_tmp = None
        if mpi_config.rank == 0:
            try:
                var_tmp = id_tmp.variables[input_forcings.netcdf_var_names[force_count]][0, :]
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract: " + input_forcings.netcdf_var_names[force_count] + \
                                        " from: " + input_forcings.tmpFile + " (" + str(err) + ")"
                err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        var_sub_tmp = mpi_config.scatter_array(input_forcings, var_tmp, config_options)
        err_handler.check_program_status(config_options, mpi_config)

        try:
            input_forcings.esmf_field_in.data[:] = var_sub_tmp

        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to place input MPAS data into ESMF field: " + str(err)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if mpi_config.rank == 0:
            config_options.statusMsg = "Regridding input MPAS Field: " + input_forcings.netcdf_var_names[force_count]
            err_handler.log_msg(config_options, mpi_config)
        try:
            input_forcings.esmf_field_out = input_forcings.regridObj(input_forcings.esmf_field_in,
                                                                     input_forcings.esmf_field_out)
        except ValueError as ve:
            config_options.errMsg = "Unable to regrid input MPA forcing data: " + str(ve)
            err_handler.log_critical(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)

        if nc_var == 'rainnc':
            force_count -= 1
            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] += \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)
        else:
            try:
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :] = \
                    input_forcings.esmf_field_out.data
            except (ValueError, KeyError, AttributeError) as err:
                config_options.errMsg = "Unable to extract regridded HRRR forcing data from the ESMF field: " + str(err)
                err_handler.log_critical(config_options, mpi_config)
            err_handler.check_program_status(config_options, mpi_config)

        # If we are on the first timestep, set the previous regridded field to be
        # the latest as there are no states for time 0.
        if config_options.current_output_step == 1:
            input_forcings.regridded_forcings1[input_forcings.input_map_output[force_count], :, :] = \
                input_forcings.regridded_forcings2[input_forcings.input_map_output[force_count], :, :]

    if mpi_config.rank == 0:
        try:
            id_tmp.close()
        except OSError:
            config_options.errMsg = "Unable to close NetCDF file: " + input_forcings.tmpFile
            err_handler.log_critical(config_options, mpi_config)


def calculate_weights(id_tmp, force_count, input_forcings, config_options, mpi_config, fill=False):
    # read the ESMF Unstructured mesh file

    try:
        input_forcings.esmf_grid_in = ESMF.Mesh(filename=config_options.grid_meta, filetype=ESMF.FileFormat.ESMFMESH)
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create source ESMF mesh from MPAS ESMFMESH file: " + \
                                input_forcings.tmpFile + " (" + str(esmf_error) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    if mpi_config.rank == 0:
        try:
            input_forcings.nx_global = id_tmp.variables[input_forcings.netcdf_var_names[force_count]].shape[1]
        except (ValueError, KeyError, AttributeError) as err:
            config_options.errMsg = "Unable to extract cell coun t from: " + \
                                    input_forcings.netcdf_var_names[force_count] + " from: " + \
                                    input_forcings.tmpFile + " (" + str(err) + ")"
            err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # Broadcast the global nx/ny values
    input_forcings.nx_global = mpi_config.broadcast_parameter(input_forcings.nx_global,
                                                              config_options, param_type=int)
    err_handler.check_program_status(config_options, mpi_config)
    input_forcings.ny_global = 1

    # Save the local values for convienience
    input_forcings.nx_local = input_forcings.esmf_grid_in.size[1]
    input_forcings.ny_local = 1

    if input_forcings.nx_local < 2 :
        config_options.errMsg = f"You have either specified too many cores for: {input_forcings.productName}, " + \
                                "or your input forcing mesh is too small to process. Local mesh must " \
                                "have local size >= 2."
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # TODO: handle lat / lons to (potentially) allow for compatibility with gridded BC/Downscaling

    # create a field on the mesh
    try:
        input_forcings.esmf_field_in = ESMF.Field(input_forcings.esmf_grid_in, meshloc=ESMF.MeshLoc.ELEMENT,
                                                  name=input_forcings.productName + "_NATIVE")
        input_forcings.esmf_field_in.get_area()
    except ESMF.ESMPyException as esmf_error:
        config_options.errMsg = "Unable to create ESMF field object: " + str(esmf_error)
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)

    # set field to 1 for testing
    input_forcings.esmf_field_in.data[:] = 1

    try:
        input_forcings.x_lower_bound = input_forcings.esmf_field_in.lower_bounds[0]
        input_forcings.x_upper_bound = input_forcings.esmf_field_in.upper_bounds[0]
        input_forcings.y_lower_bound = 0
        input_forcings.y_upper_bound = 0
        input_forcings.nx_local = input_forcings.x_upper_bound - input_forcings.x_lower_bound
        input_forcings.ny_local = 1
    except (ValueError, KeyError, AttributeError) as err:
        config_options.errMsg = "Unable to extract local X/Y boundaries from global grid from temporary " + \
                                "file: " + input_forcings.tmpFile + " (" + str(err) + ")"
        err_handler.log_critical(config_options, mpi_config)
    err_handler.check_program_status(config_options, mpi_config)


    # ## CALCULATE WEIGHT ## #
    # Try to find a pre-existing weight file, if available

    weight_file = None
    if config_options.weightsDir is not None:
        grid_key = input_forcings.productName
        border = 0
        weight_file = os.path.join(config_options.weightsDir, "ESMF_weight_{}_b{}.nc4".format(grid_key, border))
        # check if file exists:
        if os.path.exists(weight_file):
            # read the data
            try:
                if mpi_config.rank == 0:
                    config_options.statusMsg = "Loading cached ESMF weight object for " + input_forcings.productName + \
                                               " from " + weight_file
                    err_handler.log_msg(config_options, mpi_config)
                err_handler.check_program_status(config_options, mpi_config)

                begin = time.monotonic()
                input_forcings.regridObj = ESMF.RegridFromFile(input_forcings.esmf_field_in,
                                                               input_forcings.esmf_field_out,
                                                               weight_file)
                end = time.monotonic()

                if mpi_config.rank == 0:
                    config_options.statusMsg = "Finished loading weight object with ESMF, took {} seconds".format(
                        end - begin)
                    err_handler.log_msg(config_options, mpi_config)

            except (IOError, ValueError, ESMF.ESMPyException) as esmf_error:
                config_options.errMsg = "Unable to load cached ESMF weight file: " + str(esmf_error)
                err_handler.log_warning(config_options, mpi_config)

    if input_forcings.regridObj is None:
        if mpi_config.rank == 0:
            config_options.statusMsg = "Creating weight object from ESMF"
            err_handler.log_msg(config_options, mpi_config)
        err_handler.check_program_status(config_options, mpi_config)
        try:
            extrap_method = ESMF.ExtrapMethod.CREEP_FILL if fill else ESMF.ExtrapMethod.NONE
            regrid_method = (ESMF.RegridMethod.BILINEAR, ESMF.RegridMethod.NEAREST_STOD)[input_forcings.regridOpt - 1]
            begin = time.monotonic()
            input_forcings.regridObj = ESMF.Regrid(input_forcings.esmf_field_in,
                                                   input_forcings.esmf_field_out,
                                                   src_mask_values=np.array([0, config_options.globalNdv]),
                                                   regrid_method=regrid_method,
                                                   unmapped_action=ESMF.UnmappedAction.IGNORE,
                                                   extrap_method=extrap_method,
                                                   filename=weight_file)
            end = time.monotonic()

            if mpi_config.rank == 0:
                config_options.statusMsg = "Finished generating weight object with ESMF, took {} seconds".format(
                        end - begin)
                err_handler.log_msg(config_options, mpi_config)
        except (RuntimeError, ImportError, ESMF.ESMPyException) as esmf_error:
            config_options.errMsg = "Unable to regrid input data from ESMF: " + str(esmf_error)
            err_handler.log_critical(config_options, mpi_config)
            etype, value, tb = sys.exc_info()
            traceback.print_exception(etype, value, tb)

        err_handler.check_program_status(config_options, mpi_config)
