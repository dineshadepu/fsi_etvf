from pysph.sph.integrator import Integrator


class EDACIntegrator(Integrator):
    def initial_acceleration(self, t, dt):
        pass

    def one_timestep(self, t, dt):
        self.compute_accelerations(0, update_nnps=False)
        self.stage1()
        self.do_post_stage(dt, 1)
        self.update_domain()

        self.compute_accelerations(1)

        self.stage2()
        self.do_post_stage(dt, 2)
