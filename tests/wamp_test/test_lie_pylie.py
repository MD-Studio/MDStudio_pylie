from mdstudio.deferred.chainable import chainable
from mdstudio.component.session import ComponentSession
from mdstudio.runner import main
from os.path import join

import numpy as np
import os
import pandas as pd
import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


file_path = os.path.realpath(__file__)
root = os.path.split(file_path)[0]


def create_path_file_obj(path, encoding='utf8'):
    """
    Encode the input files
    """
    extension = os.path.splitext(path)[1]
    with open(path, 'r') as f:
        content = f.read()

    return {
        u'path': path, u'encoding': encoding,
        u'content': str(content), u'extension': extension}


def create_workdir(name, path="/tmp/mdstudio/lie_pylie"):
    """Create temporal workdir dir"""
    workdir = join(path, name)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    return workdir


def compare_csv_files(str_1, str_2):
    """check if two csv files are the same"""
    f1 = StringIO(str_1)
    f2 = StringIO(str_2)
    df1 = pd.read_csv(f1).sort_index(axis=1)
    df2 = pd.read_csv(f2).sort_index(axis=1)

    return df1.equals(df2)


def compare_dictionaries(d1, d2):
    """Compare two dictionaries with nested numerical results """
    df1 = pd.DataFrame(d1).sort_index(axis=1)
    df2 = pd.DataFrame(d2).sort_index(axis=1)
    r = df1 - df2

    return abs(r.sum(axis=1).iloc[0]) < 1e-8


path_unbound = join(root, "files/trajectory/unbound_trajectory.ene")
path_bound = join(root, "files/trajectory/bound_trajectory.ene")
dict_trajectory = {
    u"unbound_trajectory": [create_path_file_obj(path_unbound)],
    u"bound_trajectory": [create_path_file_obj(path_bound)],
    u"lie_vdw_header": u"Ligand-Ligenv-vdw",
    u"lie_ele_header": u"Ligand-Ligenv-ele",
    u"workdir": u"/tmp"}

path_mdframe = join(root, "files/stable/mdframe.csv")
dict_stable = {u"mdframe": create_path_file_obj(path_mdframe),
               u"workdir": u"/tmp",
               u"FilterSplines": {u"minlength": 45}}

path_splinefiltered = join(root, "files/average/mdframe_splinefiltered.csv")
dict_average = {u"mdframe": create_path_file_obj(path_splinefiltered),
                u"workdir": u"/tmp"}

path_averaged = join(root, "files/deltag/averaged.csv")
dict_deltag = {
    u"alpha_beta_gamma": [0.5937400744224419,  0.31489794216038647, 0.0],
    u"workdir": u"/tmp",
    u"dataframe": create_path_file_obj(path_averaged)}

path_decompose = join(root, "files/adan_residue_deco/decompose_dataframe.ene")
path_params = join(root, "files/adan_residue_deco/params.pkl")
dict_adan_residue = {
    u"workdir": u"/tmp",
    u"decompose_files": [create_path_file_obj(path_decompose)],
    u"model_pkl": create_path_file_obj(path_params)}

path_liedeltag = join(root, "files/adan_dene_yrange/liedeltag.csv")
dict_adan_yrange = {
    u"workdir": u"/tmp",
    u"dataframe": create_path_file_obj(path_liedeltag),
    u"ymin": -42.59,
    u"ymax": -10.79,
    u"liedeltag": {
          u"case": {
            u"0": 1.0
          },
          u"prob-1": {
            u"0": 1.0
          },
          u"w_d1": {
            u"0": 1.0
          },
          u"w_coul": {
            u"0": 8.451392745098037
          },
          u"ref_affinity": {
            u"0": np.NaN
          },
          u"dg_calc": {
            u"0": -24.76012625865582
          },
          u"beta": {
            u"0": 0.31489794216038647
          },
          u"w_vdw": {
            u"0": -46.18427090196078
          },
          u"error": {
            u"0": np.NaN
          },
          u"alpha": {
            u"0": 0.5937400744224419
          },
          u"gamma": {
            u"0": 0.0
          }
        }
}

expected_adan_yrange_results = {
    u"case": {u"0": 1.0},
    u"CI": {u"0": 1},
    u"prob-1": {u"0": 1.0},
    u"w_d1": {u"0": 1.0},
    u"w_coul": {u"0": 8.451392745098037},
    u"ref_affinity": {u"0": np.NaN},
    u"dg_calc": {u"0": -24.76012625865582},
    u"beta": {u"0": 0.3148979421603865},
    u"w_vdw": {u"0": -46.18427090196078},
    u"error": {u"0": np.NaN},
    u"alpha": {u"0": 0.5937400744224419},
    u"gamma": {u"0": 0.0}
}

dict_adan_dene = {
    u"ci_cutoff": 13.690708685318436,
    u"workdir": u"/tmp",
    u"liedeltag": dict_adan_yrange["liedeltag"],
    u"model_pkl": dict_adan_residue["model_pkl"],
    u"dataframe": dict_adan_yrange["dataframe"],
    u"center": [-53.11058012546337, 21.656883661248937]}

expected_adan_results = {
    u"case": {u"0": 1.0},
    u"CI": {u"0": 0},
    u"mahal": {u"0": 0.5355241558090091},
    u"prob-1": {u"0": 1.0},
    u"w_d1": {u"0": 1.0},
    u"w_coul": {u"0": -13.2054909161509},
    u"ref_affinity": {u"0": np.NaN},
    u"dg_calc": {u"0": -24.76012625865582},
    u"beta": {u"0": 0.3148979421603865},
    u"w_vdw": {u"0": 6.926309223502592},
    u"error": {u"0": np.NaN},
    u"alpha": {u"0": 0.5937400744224419},
    u"gamma": {u"0": 0.0}
}


class Run_pylie(ComponentSession):

    def authorize_request(self, uri, claims):
        return True

    @chainable
    def on_run(self):
        result_collect = yield self.call(
            "mdgroup.lie_pylie.endpoint.collect_energy_trajectories",
            dict_trajectory)
        assert compare_csv_files(
            result_collect["mdframe"]['content'], dict_stable["mdframe"]['content'])
        print("method collect_energy_trajectories succeeded")

        result_stable = yield self.call(
            "mdgroup.lie_pylie.endpoint.filter_stable_trajectory",
            dict_stable)
        # assert compare_csv_files(
        #     result_stable["filtered_mdframe"]['content'], dict_average["mdframe"]['content'])
        print("method filter_stable_trajectory succeeded!")

        result_average = yield self.call(
            "mdgroup.lie_pylie.endpoint.calculate_lie_average", dict_average)
        assert compare_csv_files(
            result_average["averaged"]['content'], dict_deltag["dataframe"]['content'])
        print("method calculate_lie_average succeeded!")

        result_liedeltag = yield self.call(
            "mdgroup.lie_pylie.endpoint.liedeltag", dict_deltag)
        assert compare_csv_files(
            result_liedeltag["liedeltag_file"]['content'], dict_adan_yrange["dataframe"]['content'])
        print("method liedeltag succeeded!")

        result_adan_yrange = yield self.call(
            "mdgroup.lie_pylie.endpoint.adan_dene_yrange", dict_adan_yrange)
        assert compare_dictionaries(result_adan_yrange["decomp"], expected_adan_yrange_results)
        print("method adan_dene_yrange succeeded!")

        result_adan_dene = yield self.call(
            "mdgroup.lie_pylie.endpoint.adan_dene", dict_adan_dene)
        assert compare_dictionaries(result_adan_dene["decomp"], expected_adan_results)
        print("method adan_dene succeeded!")


if __name__ == "__main__":
    main(Run_pylie)
