#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

matplotlib.use('pdf')
# n_core = free_cores()
# n_thread = 2 * free_cores()

n_core = 4
n_thread = 8
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class OscillatingPlateTurek(Problem):
    def get_name(self):
        return 'oscillating_plate_turek'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/oscillating_plate_turek.py' + backend

        # length = [1., 2., 3., 4.]
        # height = [0.1]
        # pfreq = 500

        # Base case info
        self.case_info = {
            'etvf_N_25_alpha_1_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=1.0,
                N=25,
                clamp=None,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 1'),

            'etvf_N_25_alpha_2_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 2'),

            'etvf_N_25_alpha_3_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=3.0,
                clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 3'),

            'gtvf_N_25_alpha_2_clamped': (dict(
                pst='gtvf',
                uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                no_continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                clamp=None,
                N=25,
                pfreq=1000,
                tf=1.3,
                ), 'GTVF N 25 Alpha 2'),

            'gray_N_25_alpha_2_clamped': (dict(
                pst='gray',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                no_uhat_cont=None,
                no_continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'GRAY N 25 Alpha 2'),

            'etvf_N_25_alpha_1_not_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=1.0,
                N=25,
                no_clamp=None,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 1'),

            'etvf_N_25_alpha_2_not_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                no_clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 2'),

            'etvf_N_25_alpha_3_not_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=3.0,
                no_clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 3'),

            'gtvf_N_25_alpha_2_not_clamped': (dict(
                pst='gtvf',
                uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                no_continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                no_clamp=None,
                N=25,
                pfreq=1000,
                tf=1.3,
                ), 'GTVF N 25 Alpha 2'),

            'gray_N_25_alpha_2_not_clamped': (dict(
                pst='gray',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                no_uhat_cont=None,
                no_continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                no_clamp=None,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'GRAY N 25 Alpha 2'),

            'etvf_N_50_alpha_2_not_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                N=50,
                no_clamp=None,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 50 Alpha 2'),

            'etvf_N_100_alpha_2_not_clamped': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                N=100,
                no_clamp=None,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 100 Alpha 2'),
            # 'etvf_N_50': (dict(
            #     pst='sun2019',
            #     no_uhat_velgrad=None,
            #     no_shear_stress_tvf_correction=None,
            #     no_edac=None,
            #     no_surf_p_zero=None,
            #     uhat_cont=None,
            #     continuity_tvf_correction=None,
            #     N=50,
            #     pfreq=1000,
            #     ), 'ETVF N 50'),
            # 'etvf_N_100': (dict(
            #     pst='sun2019',
            #     no_uhat_velgrad=None,
            #     no_shear_stress_tvf_correction=None,
            #     no_edac=None,
            #     no_surf_p_zero=None,
            #     uhat_cont=None,
            #     continuity_tvf_correction=None,
            #     N=100,
            #     pfreq=1000,
            #     ), 'ETVF N 100'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_figures()

    def plot_figures(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        t_fem = data['etvf_N_25_alpha_2_not_clamped']['t_fem']
        y_amplitude_fem = data['etvf_N_25_alpha_2_not_clamped']['y_amplitude_fem']

        # sort the lists
        idx  = np.argsort(t_fem)
        list1 = np.array(t_fem)[idx]
        list2 = np.array(y_amplitude_fem)[idx]

        plt.plot(list1, list2, '--', label='FEM')

        for name in self.case_info:
            if name == 'etvf_N_25_alpha_2_not_clamped':
                t = data[name]['t']
                y_amplitude = data[name]['y_amplitude']

                plt.plot(t, y_amplitude, label=self.case_info[name][1])

            if name == 'etvf_N_50_alpha_2_not_clamped':
                t = data[name]['t']
                y_amplitude = data[name]['y_amplitude']

                plt.plot(t, y_amplitude, label=self.case_info[name][1])

            if name == 'etvf_N_100_alpha_2_not_clamped':
                t = data[name]['t']
                y_amplitude = data[name]['y_amplitude']

                plt.plot(t, y_amplitude, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('Y - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('etvf_diff_resolutions_y_amplitude.pdf'))
        plt.clf()
        plt.close()


class ElasticDamBreak2D(Problem):
    def get_name(self):
        return 'elastic_dam_break_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/elastic_dam_break_2d.py' + backend

        # length = [1., 2., 3., 4.]
        # height = [0.1]
        # pfreq = 500

        # Base case info
        self.case_info = {
            'etvf': (dict(
                scheme='etvf',
                pfreq=300), 'ETVF'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class FSIBM1(Problem):
    def get_name(self):
        return 'fsi_bm_1'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/fsi_bm_1.py' + backend

        # length = [1., 2., 3., 4.]
        # height = [0.1]
        # pfreq = 500

        # Base case info
        self.case_info = {
            'etvf_gate_rho_500_with_gravity': (dict(
                scheme='etvf',
                pst='sun2019',
                gate_rho=500,
                structure_gravity=None,
                pfreq=300), 'ETVF'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('pdf')

    PROBLEMS = [
        ElasticDamBreak2D, OscillatingPlateTurek, FSIBM1

    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    # task = FileCommandTask(
    #   'latexmk manuscript/paper.tex -pdf -outdir=manuscript',
    #   ['manuscript/paper.pdf']
    # )
    # automator.add_task(task, name='pdf', post_proc=True)

    automator.run()
