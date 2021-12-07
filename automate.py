#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name, mdict, dprod, opts2path, filter_cases
from automan.jobs import free_cores

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

# n_core = free_cores()
# n_thread = 2 * free_cores()

n_core = 24
n_thread = 24 * 2
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

        self.opts = mdict(N=[10, 15, 20])

        self.cases = []
        for kw in self.opts:
            name = opts2path(kw)
            folder_name = self.input_path(name)
            self.cases.append(
                Simulation(folder_name, cmd,
                           job_info=dict(n_core=n_core,
                                         n_thread=n_thread), cache_nnps=None,
                           pst='sun2019',
                           solid_stress_bc=None,
                           solid_velocity_bc=None,
                           damping=None,
                           damping_coeff=0.02,
                           artificial_vis_alpha=2.0,
                           no_clamp=None,
                           distributed=None,
                           gradual_force=None,
                           gradual_force_time=0.1,
                           no_wall_pst=None,
                           pfreq=500,
                           tf=0.3,
                           **kw))

    def run(self):
        self.make_output_dir()
        self.plot_disp(fname='homogenous')

    def plot_disp(self, fname):
        max_t = 0.
        # print("condition is", condition)
        t_analytical = np.array([])

        for case in self.cases:
            data = np.load(case.input_path('results.npz'))

            t = data['t_ctvf']
            max_t = max(max_t, max(t))
            y_ctvf = data['y_ctvf']

            label = opts2path(case.params, keys=['N'])

            plt.plot(t, y_ctvf, label=label.replace('_', ' = '))

            t_analytical = np.linspace(0., max_t, 1000)
            amplitude_analytical = data['amplitude_analytical'][0] * np.ones_like(t_analytical)

        if len(t_analytical) > 0:
            plt.plot(t_analytical, amplitude_analytical, label='Analytical')

        plt.xlabel('time')
        plt.ylabel('Y - amplitude')
        plt.legend()
        plt.savefig(self.output_path(fname))
        plt.clf()
        plt.close()


class Sun2019OscillatingPlateTurek(Problem):
    def get_name(self):
        return 'sun_2019_oscillating_plate_turek'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/oscillating_plate_turek.py' + backend

        self.opts = mdict(N=[10, 15, 20])

        self.cases = []
        for kw in self.opts:
            name = opts2path(kw)
            folder_name = self.input_path(name)
            self.cases.append(
                Simulation(folder_name, cmd,
                           job_info=dict(n_core=n_core,
                                         n_thread=n_thread), cache_nnps=None,
                           pst='sun2019',
                           solid_stress_bc=None,
                           solid_velocity_bc=None,
                           no_damping=None,
                           artificial_vis_alpha=2.0,
                           no_clamp=None,
                           wall_pst=None,
                           pfreq=1000,
                           tf=5,
                           **kw))

    def run(self):
        self.make_output_dir()
        self.plot_disp(fname='homogenous')

    def plot_disp(self, fname):
        for case in self.cases:
            data = np.load(case.input_path('results.npz'))

            t = data['t_ctvf']
            amplitude_ctvf = data['amplitude_ctvf']

            label = opts2path(case.params, keys=['N'])

            plt.plot(t, amplitude_ctvf, label=label.replace('_', ' = '))

            t_fem = data['t_fem']
            amplitude_fem = data['amplitude_fem']

        # sort the fem data before plotting
        p = t_fem.argsort()
        t_fem_new = t_fem[p]
        amplitude_fem_new = amplitude_fem[p]
        plt.plot(t_fem_new, amplitude_fem_new, label='FEM')

        plt.xlabel('time')
        plt.ylabel('Y - amplitude')
        plt.legend()
        plt.savefig(self.output_path(fname))
        plt.clf()
        plt.close()


class DamBreak2D(Problem):
    def get_name(self):
        return 'dam_break_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_2d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pst='sun2019',
                no_edac=None,
                no_cont_vc_bc=None,
                dx=1e-3,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),

            'wcsph_1': (dict(
                pst='sun2019',
                edac=None,
                no_cont_vc_bc=None,
                dx=0.05,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),

            'wcsph_2': (dict(
                pst='sun2019',
                edac=None,
                no_clamp_p=None,
                cont_vc_bc=None,
                dx=0.05,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),
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


class HydrostaticTank(Problem):
    def get_name(self):
        return 'hydrostatic_tank'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hydrostatic_tank.py' + backend + ' --detailed '

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pst='sun2019',
                no_edac=None,
                no_cont_vc_bc=None,
                dx=1e-3,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),

            'wcsph_1': (dict(
                pst='sun2019',
                edac=None,
                no_cont_vc_bc=None,
                dx=0.05,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),

            'wcsph_2': (dict(
                pst='sun2019',
                edac=None,
                no_clamp_p=None,
                cont_vc_bc=None,
                dx=0.05,
                tf=1.,
                pfreq=200,
                alpha=0.05
            ), 'WCSPH'),
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
            'N_10': (dict(
                scheme='wcsph',
                solid_velocity_bc=None,
                alpha_fluid=2.0,
                no_wall_pst=None,
                damping=None,
                damping_coeff=0.002,
                pfreq=500,
                tf=3.,
                d0=0.55 * 1e-2,
                ), 'N=10'),

            'N_15': (dict(
                scheme='wcsph',
                solid_velocity_bc=None,
                alpha_fluid=2.0,
                no_wall_pst=None,
                damping=None,
                damping_coeff=0.002,
                pfreq=700,
                tf=3.,
                d0=1e-2 * 0.35,
                ), 'N=15'),

            'N_20': (dict(
                scheme='wcsph',
                solid_velocity_bc=None,
                alpha_fluid=2.0,
                no_wall_pst=None,
                damping=None,
                damping_coeff=0.002,
                pfreq=1500,
                tf=3.,
                d0=1e-2 * 0.25,
                ), 'N=20'),
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

        rand_case = (list(data.keys())[0])
        t_analytical = data[rand_case]['t_analytical']
        y_analytical = data[rand_case]['y_analytical']

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
            # 'ctvf': (dict(
            #     scheme='ctvf',
            #     pfreq=500,
            #     tf=0.4,
            #     ), 'CTVF')

            'wcsph_fluids': (dict(
                scheme='wcsph_fluids',
                pfreq=500,
                tf=0.4,
                ), 'wcsph_fluids'),

            'wcsph': (dict(
                scheme='wcsph',
                pfreq=500,
                tf=0.4,
                ), 'wcsph')
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

        rand_case = (list(data.keys())[0])

        txant = data[rand_case]['txant']
        xdant = data[rand_case]['xdant']
        txkha = data[rand_case]['txkha']
        xdkha = data[rand_case]['xdkha']
        txyan = data[rand_case]['txyan']
        xdyan = data[rand_case]['xdyan']
        txng = data[rand_case]['txng']
        xdng = data[rand_case]['xdng']
        txwcsph = data[rand_case]['txwcsph']
        xdwcsph = data[rand_case]['xdwcsph']

        tyant = data[rand_case]['tyant']
        ydant = data[rand_case]['ydant']
        tykha = data[rand_case]['tykha']
        ydkha = data[rand_case]['ydkha']
        tyyan = data[rand_case]['tyyan']
        ydyan = data[rand_case]['ydyan']
        tyng = data[rand_case]['tyng']
        ydng = data[rand_case]['ydng']

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


class Sun2019DamBreakingFlowImpactingAnElasticPlate(Problem):
    def get_name(self):
        return 'sun_2019_dam_breaking_flow_impacting_an_elastic_plate'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/sun_2019_dam_breaking_flow_impacting_an_elastic_plate.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                scheme='wcsph',
                pfreq=300,
                tf=0.7,
                ), 'WCSPH'),

            'wcsph_fluids': (dict(
                scheme='wcsph_fluids',
                pfreq=300,
                tf=0.7,
                ), 'WCSPH fluids'),
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

        rand_case = (list(data.keys())[0])

        txsun = data[rand_case]['txsun']
        xdsun = data[rand_case]['xdsun']
        txbogaers = data[rand_case]['txbogaers']
        xdbogaers = data[rand_case]['xdbogaers']
        txidelsohn = data[rand_case]['txidelsohn']
        xdidelsohn = data[rand_case]['xdidelsohn']
        txliu = data[rand_case]['txliu']
        xdliu = data[rand_case]['xdliu']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        plt.plot(txsun, xdsun, "o", label='Sun et al 2019, MPS-DEM')
        plt.plot(txbogaers, xdbogaers, "^", label='Bogaers 2016, QN-LS')
        plt.plot(txidelsohn, xdidelsohn, "+", label='Idelsohn 20108, PFEM')
        plt.plot(txliu, xdliu, "v", label='Liu 2013, SPH')

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
            'wcsph': (dict(
                scheme='wcsph',
                pfreq=300,
                tf=0.0025,
                ), 'WCSPH'),

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

        rand_case = (list(data.keys())[0])

        tyatbanalytical = data[rand_case]['tyatbanalytical']
        yatbanalytical = data[rand_case]['yatbanalytical']
        tyatbfourey = data[rand_case]['tyatbfourey']
        yatbfourey = data[rand_case]['yatbfourey']
        tyatbli = data[rand_case]['tyatbli']
        yatbli = data[rand_case]['yatbli']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        plt.plot(tyatbanalytical, yatbanalytical, "o-", label='Analytical')
        plt.plot(tyatbfourey, yatbfourey, "^-", label='Fourey 2010, SPH-FEM')
        plt.plot(tyatbli, yatbli, "+-", label='Li 2015, SPH-FEM')

        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            yatbctvf = data[name]['yatbctvf']

            plt.plot(t_ctvf, yatbctvf, label=self.case_info[name][1])

        plt.xlabel('t')
        plt.ylabel('Displacement')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('b_displacement_with_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


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
        # stuctures cases
        Hwang2014StaticCantileverBeam,
        Sun2019OscillatingPlateTurek,

        # fluids cases
        DamBreak2D,
        HydrostaticTank,

        Ng2020HydrostaticWaterColumnOnElasticPlate,
        Ng2020ElasticDamBreak,
        Sun2019DamBreakingFlowImpactingAnElasticPlate,
        Zhang2021HighSpeedWaterEntryOfAnElasticWedge,
        Hwang2014RollingTankWithEmbeddedHangingElasticBeam,

        # vector dem cases
        # Nasar2019NonLinearQuasiStaticDefectionOf2DCantilieverBeam
    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    automator.run()
