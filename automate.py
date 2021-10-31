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

n_core = 16
n_thread = 32
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


class Hwang2014StaticCantileverBeam(Problem):
    def get_name(self):
        return 'hwang_2014_static_cantilever_beam'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hwang_2014_static_cantilever_beam.py' + backend

        # Base case info
        self.case_info = {
            'n_5': (dict(
                pst='sun2019',
                solid_stress_bc=None,
                solid_velocity_bc=None,
                no_edac=None,
                continuity_correction=None,
                no_clamp=None,
                artificial_vis_alpha=2.0,
                N=5,
                distributed=None,
                gradual_force=None,
                gradual_force_time=0.1,
                no_wall_pst=None,
                no_two_layer=None,
                no_sandwich=None,
                pfreq=500,
                tf=0.3,
                ), 'N 5 '),

            'n_10': (dict(
                pst='sun2019',
                solid_stress_bc=None,
                solid_velocity_bc=None,
                no_edac=None,
                continuity_correction=None,
                no_clamp=None,
                artificial_vis_alpha=2.0,
                N=10,
                distributed=None,
                gradual_force=None,
                gradual_force_time=0.1,
                no_wall_pst=None,
                pfreq=500,
                tf=0.3,
                ), 'N 10'),

            # 'n_10_rk2': (dict(
            #     pst='sun2019',
            #     solid_stress_bc=None,
            #     solid_velocity_bc=None,
            #     no_edac=None,
            #     continuity_correction=None,
            #     no_uhat_vgrad=None,
            #     integrator="rk2",
            #     no_clamp=None,
            #     artificial_vis_alpha=2.0,
            #     N=10,
            #     distributed=None,
            #     gradual_force=None,
            #     gradual_force_time=0.1,
            #     no_wall_pst=None,
            #     no_two_layer=None,
            #     no_sandwich=None,
            #     pfreq=700,
            #     tf=0.3,
            #     ), 'N 10 RK2'),
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

        amplitude_analytical = data['n_5']['amplitude_analytical']
        t_analytical = data['n_5']['t_analytical']
        amplitude_khayyer = data['n_5']['amplitude_khayyer']
        t_khayyer = data['n_5']['t_analytical']

        plt.plot(t_analytical, amplitude_analytical, '--', label='Analytical')
        plt.plot(t_khayyer, amplitude_khayyer, '-', label='Khayyer')

        for name in self.case_info:
            t = data[name]['t_ctvf']
            amplitude_ctvf = data[name]['amplitude_ctvf']

            plt.plot(t, amplitude_ctvf, label=self.case_info[name][1])
            # if name == 'n_5':
            #     t = data[name]['t_ctvf']
            #     amplitude_ctvf = data[name]['amplitude_ctvf']

            #     plt.plot(t, amplitude_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('Y - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('homogenous_unclamped_plate_udl.pdf'))
        plt.clf()
        plt.close()


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

            'etvf_N_25_alpha_2_clamped_clamp_factor_10': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                clamp=None,
                clamp_factor=10,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 2 Clamp factor 10'),

            'etvf_N_25_alpha_2_clamped_clamp_factor_8': (dict(
                pst='sun2019',
                no_uhat_velgrad=None,
                no_shear_stress_tvf_correction=None,
                no_edac=None,
                no_surf_p_zero=None,
                uhat_cont=None,
                continuity_tvf_correction=None,
                artificial_vis_alpha=2.0,
                clamp=None,
                clamp_factor=8,
                N=25,
                pfreq=1000,
                tf=5.,
                ), 'ETVF N 25 Alpha 2 Clamp factor 8'),

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


class Ng2020HydrostaticWaterColumnOnElasticPlate(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'ng_2020_hydrostatic_water_column_on_elastic_plate'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/ng_2020_hydrostatic_water_column_on_elastic_plate.py' + backend

        # Base case info
        self.case_info = {
            'ctvf': (dict(
                scheme='substep',
                pfreq=500,
                tf=1.,
                no_rogers_eqns=None,
                d0=1e-2,
                ), 'CTVF'),

            # 'rogers': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=1.,
            #     rogers_eqns=None
            #     ), 'Rogers Scheme'),
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

        t_analytical = data['ctvf']['t_analytical']
        y_analytical = data['ctvf']['y_analytical']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_analytical, y_analytical, "-", label='Analytical')
        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            y_ctvf = data[name]['y_ctvf']

            plt.plot(t_ctvf, y_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('y - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Ng2020ElasticDamBreak(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'ng_2020_elastic_dam_break'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/ng_2020_elastic_dam_break.py' + backend

        # Base case info
        self.case_info = {
            'ctvf': (dict(
                scheme='ctvf',
                pfreq=500,
                tf=0.4,
                no_rogers_eqns=None
                ), 'CTVF'),

            # 'rogers': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=0.4,
            #     rogers_eqns=None
            #     ), 'Rogers Scheme'),
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

        txant = data['ctvf']['txant']
        xdant = data['ctvf']['xdant']
        txkha = data['ctvf']['txkha']
        xdkha = data['ctvf']['xdkha']
        txyan = data['ctvf']['txyan']
        xdyan = data['ctvf']['xdyan']
        txng = data['ctvf']['txng']
        xdng = data['ctvf']['xdng']
        txwcsph = data['ctvf']['txwcsph']
        xdwcsph = data['ctvf']['xdwcsph']

        tyant = data['ctvf']['tyant']
        ydant = data['ctvf']['ydant']
        tykha = data['ctvf']['tykha']
        ydkha = data['ctvf']['ydkha']
        tyyan = data['ctvf']['tyyan']
        ydyan = data['ctvf']['ydyan']
        tyng = data['ctvf']['tyng']
        ydng = data['ctvf']['ydng']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(txant, xdant, "o", label='Antoci 2008, Experiment')
        plt.plot(txkha, xdkha, "^", label='Khayyer 2018, ISPH-SPH')
        plt.plot(txyan, xdyan, "+", label='Yang 2012, SPH-FEM')
        plt.plot(txng, xdng, "v", label='Ng 2020, SPH-VCPM')
        plt.plot(txwcsph, xdwcsph, "*", label='WCSPH PySPH')
        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            x_ctvf = data[name]['x_ctvf']

            plt.plot(t_ctvf, x_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('x - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('x_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

        # ==================================
        # Plot y amplitude
        # ==================================
        plt.plot(tyant, ydant, "o", label='Antoci 2008, Experiment')
        plt.plot(tykha, ydkha, "^", label='Khayyer 2018, ISPH-SPH')
        plt.plot(tyyan, ydyan, "+", label='Yang 2012, SPH-FEM')
        plt.plot(tyng, ydng, "v", label='Ng 2020, SPH-VCPM')
        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            y_ctvf = data[name]['y_ctvf']

            plt.plot(t_ctvf, y_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('y - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot y amplitude
        # ==================================


class Sun2019DamBreakingFlowImpactingAnElasticGate(Problem):
    def get_name(self):
        return 'sun_2019_dam_breaking_flow_impacting_an_elastic_gate'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/sun_2019_dam_breaking_flow_impacting_an_elastic_gate.py' + backend

        # Base case info
        self.case_info = {
            'ctvf': (dict(
                scheme='ctvf',
                pfreq=200,
                case="db2",
                tf=0.003,
                no_rogers_eqns=None
                ), 'CTVF'),

            # 'rogers': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=0.4,
            #     rogers_eqns=None
            #     ), 'Rogers Scheme'),
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


class Zhang2021HighSpeedWaterEntryOfAnElasticWedge(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'zhang_2021_high_speed_water_entry_of_an_elastic_wedge'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/zhang_2021_high_speed_water_entry_of_an_elastic_wedge.py' + backend

        # Base case info
        self.case_info = {
            'ctvf': (dict(
                scheme='ctvf',
                pfreq=200,
                tf=0.003,
                no_rogers_eqns=None
                ), 'CTVF'),

            # 'rogers': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=0.4,
            #     rogers_eqns=None
            #     ), 'Rogers Scheme'),
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


class Hwang2014RollingTankWithEmbeddedHangingElasticBeam(Problem):
    def get_name(self):
        return 'hwang_2014_rolling_tank_with_embedded_hanging_elastic_beam'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hwang_2014_rolling_tank_with_embedded_hanging_elastic_beam.py' + backend

        # Base case info
        self.case_info = {
            'ctvf': (dict(
                scheme='ctvf',
                pfreq=500,
                tf=1.,
                no_rogers_eqns=None
                ), 'CTVF'),

            # 'rogers': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=0.4,
            #     rogers_eqns=None
            #     ), 'Rogers Scheme'),
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


class Nasar2019NonLinearQuasiStaticDefectionOf2DCantilieverBeam(Problem):
    def get_name(self):
        return 'nasar_2019_non_linear_quasi_static_deflection_of_2d_cantilever_beam'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/nasar_2019_non_linear_quasi_static_deflection_of_2d_cantilever_beam.py' + backend

        # Base case info
        self.case_info = {
            'n_40': (dict(
                Nx=40,
                gradual_force=None,
                gradual_force_time=0.1,
                pfreq=500,
                tf=0.3,
                ), 'N 40'),
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
        # Static cases
        Hwang2014StaticCantileverBeam,
        OscillatingPlateTurek,

        Ng2020HydrostaticWaterColumnOnElasticPlate,
        Ng2020ElasticDamBreak,
        Sun2019DamBreakingFlowImpactingAnElasticGate,
        Zhang2021HighSpeedWaterEntryOfAnElasticWedge,
        Hwang2014RollingTankWithEmbeddedHangingElasticBeam,

        # vector dem cases
        # Nasar2019NonLinearQuasiStaticDefectionOf2DCantilieverBeam
    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    automator.run()
