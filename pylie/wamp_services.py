# -*- coding: utf-8 -*-

"""
file: wamp_services.py

WAMP service methods the module exposes.
"""

import dill
import os
import sys

from pandas import (read_csv, read_json, read_excel, read_table, concat, DataFrame)
from pylie import (LIEMDFrame, LIEDataFrame, pylie_config)
from pylie.filters.filtersplines import FilterSplines
from pylie.filters.filtergaussian import FilterGaussian
from pylie.methods.adan import (ad_residue_decomp, ad_dene, ad_dene_yrange, parse_gromacs_decomp)
from mdstudio.api.endpoint import endpoint
from mdstudio.component.session import ComponentSession

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


PANDAS_IMPORTERS = {'csv': read_csv, 'json': read_json, 'xlsx': read_excel, 'tbl': read_table}
PANDAS_EXPORTERS = {'csv': 'to_csv', 'json': 'to_json', 'xlsx': 'to_excel', 'tbl': 'to_string'}


class PylieWampApi(ComponentSession):
    """
    Pylie WAMP methods.

    Defines `require_config` to retrieve system and database configuration
    upon WAMP session setup
    """
    def authorize_request(self, uri, claims):
        return True

    def _get_config(self, config, name):

        ref_config = pylie_config.get(name).dict()
        ref_config.update(config)

        return ref_config

    def _import_to_dataframe(self, infile, ext='csv'):

        df = PANDAS_IMPORTERS[ext](infile)
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']

        return df

    def _export_dataframe(self, df, outfile, file_format='csv'):

        if file_format not in PANDAS_EXPORTERS:
            self.log.error('Unsupported file format: {0}'.format(file_format))
            return False

        if hasattr(df, PANDAS_EXPORTERS[file_format]):
            method = getattr(df, PANDAS_EXPORTERS[file_format])

            # Export to file
            with open(outfile, 'w') as outf:
                method(outf)

            return True
        return False

    def _check_file_status(self, filepath):
        if os.path.isfile(filepath):
            return 'completed'
        else:
            self.log.error("File: {} does not exist!".format(filepath))
            return 'failed'

    @endpoint('liedeltag', 'liedeltag_request', 'liedeltag_response')
    def calculate_lie_deltag(self, request, claims):
        """
        For a detailed input description see:
          pylie/schemas/endpoints/liedeltag_request.json

        For a detailed output description see:
          pylie/schemas/endpoints/liedeltag_response.json
        """

        alpha_beta_gamma = request['alpha_beta_gamma']

        # Filter DataFrame
        file_string = StringIO(request['dataframe']['content'])
        dfobject = LIEDataFrame(self._import_to_dataframe(file_string))
        dg_calc = dfobject.liedeltag(params=alpha_beta_gamma, kBt=request['kBt'])

        # Create workdir to save file
        workdir = os.path.abspath(request['workdir'])

        # Save dataframe
        file_format = request['fileformat']
        filepath = os.path.join(workdir, 'liedeltag.{0}'.format(file_format))
        if self._export_dataframe(dg_calc, filepath, file_format=file_format):
            results = dg_calc.to_dict()
        else:
            return None

        return {'liedeltag_file': encoder(filepath), 'liedeltag': results}

    @endpoint('concat_dataframes', 'concat_dataframes_request', 'concat_dataframes_response')
    def concat_dataframes(self, request, claims):
        """
        Combine multiple tabular DataFrames into one new DataFrame using
        the Python Pandas library.

        For a detailed input description see:
          pylie/schemas/endpoints/concat_dataframes_request.json

        For a detailed output description see:
          pylie/schemas/endpoints/concat_dataframes_response.json
        """

        # Import all files
        dfs = []
        for dataframe in request['dataframes']:
            file_string = StringIO(dataframe['content'])
            dfobject = self._import_to_dataframe(file_string)
            if isinstance(dfobject, DataFrame):
                dfs.append(dfobject)

        # Concatenate dataframes
        if len(dfs) > 1:
            concat_df = concat(
                dfs, ignore_index=request['ignore_index'],
                axis=request['axis'], join=request['join'])

            # Create workdir to save file
            workdir = os.path.abspath(request['workdir'])

            file_format = request['file_format']
            filepath = os.path.join(workdir, 'joined.{0}'.format(file_format))
            if self._export_dataframe(concat_df, filepath, file_format=file_format):
                concat_mdframe = filepath
        else:
            return None

        return {'concat_mdframe': encoder(concat_mdframe)}

    @endpoint('calculate_lie_average', 'calculate_lie_average_request', 'calculate_lie_average_response')
    def calculate_lie_average(self, request, claims):
        """
        Calculate LIE electrostatic and Van der Waals energy averages from
        a MDFrame.

        For a detailed input description see:
          pylie/schemas/endpoints/calculate_lie_average_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/calculate_lie_average_response.v1.json
        """

        mdframe = request['mdframe']

        # Create workdir to save file
        workdir = os.path.abspath(request['workdir'])

        # Import CSV file and run spline fitting filter
        file_string = StringIO(mdframe['content'])
        liemdframe = LIEMDFrame(read_csv(file_string))
        if 'Unnamed: 0' in liemdframe.columns:
            del liemdframe['Unnamed: 0']

        ave = liemdframe.inliers(method=request['inlierFilterMethod']).get_average()
        filepath = os.path.join(workdir, 'averaged.csv')
        ave.to_csv(filepath)

        return {'averaged': encoder(filepath)}

    @endpoint('gaussian_filter', 'gaussian_filter_request', 'gaussian_filter_response')
    def filter_gaussian(self, request, claims):
        """
        Use multivariate Gaussian Distribution analysis to
        filter VdW/Elec values

        For a detailed input description see:
          pylie/schemas/endpoints/gaussian_filter_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/gaussian_filter_response.v1.json
        """

        # Filter DataFrame
        file_string = StringIO(request['dataframe']['content'])
        dfobject = LIEDataFrame(self._import_to_dataframe(file_string))
        gaussian = FilterGaussian(dfobject, confidence=request["confidence"])
        filtered = gaussian.filter()
        self.log.info("Filter detected {0} outliers.".format(len(filtered.outliers.cases)))

        # Create workdir to save file
        workdir = os.path.abspath(request['workdir'])

        # Plot results
        if request['plot']:
            outp = os.path.join(workdir, 'gauss_filter.pdf')
            p = gaussian.plot()
            p.savefig(outp)

        # Save filtered dataframe
        file_format = request['file_format']
        filepath = os.path.join(workdir, 'gauss_filter.{0}'.format(file_format))
        if not self._export_dataframe(filtered, filepath, file_format=file_format):
            return None

        return {'gauss_filter': encoder(filepath)}

    @endpoint('filter_stable_trajectory', 'filter_stable_trajectory_request', 'filter_stable_trajectory_response')
    def filter_stable_trajectory(self, request, claims):
        """
        Use FFT and spline-based filtering to detect and extract stable regions
        in the MD energy trajectory

        For a detailed input description see:
          pylie/schemas/endpoints/filter_stable_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/filter_stable_response.v1.json
        """

        mdframe = request['mdframe']

        # Create workdir to save file
        workdir = os.path.abspath(request['workdir'])

        # Import CSV file and run spline fitting filter
        file_string = StringIO(mdframe['content'])
        liemdframe = LIEMDFrame(read_csv(file_string))

        if 'Unnamed: 0' in liemdframe.columns:
            del liemdframe['Unnamed: 0']

        splines = FilterSplines(liemdframe, **request['FilterSplines'])
        liemdframe = splines.filter()

        output = {}
        # Report the selected stable regions
        filtered = liemdframe.inliers()

        for pose in filtered.poses:
            stable = filtered.get_stable(pose)
            if stable:
                output['stable_pose_{0}'.format(pose)] = stable

        # Create plots
        if request['do_plot']:
            if os.path.exists(workdir):
                currpath = os.getcwd()
                os.chdir(workdir)
                paths = splines.plot(tofile=True, filetype=request['plotFileType'])
                for image_paths in paths:
                    output[image_paths] = encoder(os.path.join(workdir, image_paths))
                os.chdir(currpath)
            else:
                self.log.error('Working directory does not exist: {0}'.format(workdir))

        # Filter the mdframe
        if request['do_filter']:
            filepath = os.path.join(workdir, 'mdframe_splinefiltered.csv')
            filtered.to_csv(filepath)

        output['filtered_mdframe'] = encoder(filepath)
        return output

    @endpoint('collect_energy_trajectories', 'collect_energy_trajectories_request',
              'collect_energy_trajectories_response')
    def import_mdene_files(self, request, claims):
        """
        Import GROMACS MD trajectory energy files into a LIEMDFrame.

        The constructed LIEMDFrame should represents simulations for the same
        system with one simulation for the unbound state of the ligand and one
        or more simulations for the bound system with the ligand in potentially
        multiple binding poses.

        For a detailed input description see:
          pylie/schemas/endpoints/collect_energy_trajectories_request.v1.json

        For a detailed output description see:
          pylie/schemas/endpoints/collect_energy_trajectories_response.v1.json
        """

        # Use absolute path to save file
        workdir = os.path.abspath(request["workdir"])

        # Collect trajectories
        mdframe = LIEMDFrame()
        vdw_header = request['lie_vdw_header']
        ele_header = request['lie_ele_header']
        for pose, trj in enumerate(request['bound_trajectory']):
            mdframe.from_file(
                trj['content'], {
                    vdw_header: 'vdw_bound_{0}'.format(pose + 1),
                    ele_header: 'coul_bound_{0}'.format(pose + 1)},
                filetype=request['filetype'])
            self.log.debug('Import file: {0}, pose: {1}'.format(trj, pose))

        mdframe.from_file(
            request['unbound_trajectory']['content'], {vdw_header: 'vdw_unbound', ele_header: 'coul_unbound'},
            filetype=request['filetype'])
        self.log.debug('Import unbound file: {0}'.format(request['unbound_trajectory']['path']))

        # Set the case ID
        mdframe.case = request['case']

        # Store to file
        filepath = os.path.join(workdir, 'mdframe.csv')
        mdframe.to_csv(filepath)

        return {'mdframe': encoder(filepath)}

    @endpoint('adan_residue_decomp', 'adan_residue_decomp_request', 'adan_residue_decomp_response')
    def adan_residue_decomp(self, request, claims):
        """
        For a detailed input description see:
          pylie/schemas/endpoints/adan_residue_decomp_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/adan_residue_decomp_response.v1.json
        """

        # Load the model
        binary = request['model_pkl']['content']
        model = dill.loads(binary)

        # Parse gromacs residue decomposition energy files to DataFrame
        decomp_dfs = []
        for dcfileobj in request['decompose_files']:
            dcfile = StringIO(dcfileobj['content'])
            decomp_dfs.append(parse_gromacs_decomp(dcfile))

        # Run AD test
        ene = ad_residue_decomp(
            decomp_dfs, model['AD']['decVdw'], model['AD']['decEle'], cases=request['cases'])

        # Use absolute path to save file
        workdir = os.path.abspath(request['workdir'])

        filepath = os.path.join(workdir, 'adan_residue_decomp.csv')
        ene.to_csv(filepath)

        return {'decomp': ene.to_dict()}

    @endpoint('adan_dene', 'adan_dene_request', 'adan_dene_response')
    def adan_dene(self, request, claims):
        """
        For a detailed input description see:
          pylie/schemas/endpoints/adan_dene_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/adan_dene_response.v1.json
        """

        # Load the model
        binary = request['model_pkl']['content']
        model = dill.loads(binary)

        # Parse gromacs residue decomposition energy files to DataFrame
        file_string = StringIO(request['dataframe']['content'])
        dfobject = self._import_to_dataframe(file_string)

        # Run AD test
        ene = ad_dene(
            dfobject, model['AD']['Dene']['CovMatrix'],
            center=request['center'], ci_cutoff=request['ci_cutoff'])

        # Use absolute path to save file
        workdir = os.path.abspath(request['workdir'])

        filepath = os.path.join(workdir, 'adan_dene.csv')
        ene.to_csv(filepath)

        return {'decomp': ene.to_dict()}

    @endpoint('adan_dene_yrange', 'adan_dene_yrange_request', 'adan_dene_yrange_response')
    def adan_dene_yrange(self, request, claims):
        """
        For a detailed input description see:
          pylie/schemas/endpoints/adan_dene_yrange_request.v1.json

        For a detailed output description see:
          pydlie/schemas/endpoints/adan_dene_yrange_response.v1.json
        """

        # Parse gromacs residue decomposition energy files to DataFrame
        file_string = StringIO(request['dataframe']['content'])
        dfobject = self._import_to_dataframe(file_string)

        # Run AD test
        ene = ad_dene_yrange(dfobject, request['ymin'], request['ymax'])

        # Use absolute path to save file
        workdir = os.path.abspath(request['workdir'])

        filepath = os.path.join(workdir, 'adan_dene_yrange.csv')
        ene.to_csv(filepath)

        return {'decomp': ene.to_dict()}


def encoder(file_path, inline_content=True):
    """
    Encode the content of `file_path` into a simple dict.

    Set `inline_content` to False to prevent serialization of file content
    as part of WAMP return value. This is done for images for instance.
    The file path needs to be available on the receiver side.
    
    :param file_path:       path to local file
    :type file_path:        :py:str
    :param inline_content:  include content of file inline (serialize in
                            WAMP message)
    :type inline_content:   :py:bool

    :return:                WAMP message file object
    :rtype:                 :py:dict
    """

    extension = os.path.splitext(file_path)[1]

    content = None
    if inline_content:
        with open(file_path, 'r') as f:
            content = f.read()

    return {"path": file_path, "extension": extension,
            "content": content, "encoding": "utf8"}


def encode_file(val):
    if not os.path.isfile(val):
        return val
    else:
        return encoder(val)
