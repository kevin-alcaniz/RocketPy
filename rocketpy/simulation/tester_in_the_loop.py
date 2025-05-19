import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau
from functools import cached_property

from ..mathutils.function import funcify_method
from ..mathutils.vector_matrix import Matrix, Vector
from .flight import Flight
from ..tools import (
    calculate_cubic_hermite_coefficients,
    euler313_to_quaternions,
    find_root_linear_interpolation,
    find_roots_cubic_function,
)

ODE_SOLVER_MAP = {
    "RK23": RK23,
    "RK45": RK45,
    "DOP853": DOP853,
    "Radau": Radau,
    "BDF": BDF,
    "LSODA": LSODA,
}

class TesterInTheLoop:
    """Has a method to simulate time steps of the simulation to provide
    data in real time for the avionics.

    Attributes
    ----------
    TesterInTheLoop.rocket : Rocket
        Rocket class describing rocket. See Rocket class for more
        details.
    TesterInTheLoop.env : Environment
        Environment object describing rail length, elevation, gravity and
        weather condition. See Environment class for more details.
    TesterInTheLoop.parachutes : Parachute
        Direct link to parachutes of the Rocket. See Rocket class
        for more details.
    TesterInTheLoop.time_iterator : function
        Helper iterator function to generate time discretization points.
    TesterInTheLoop.rail_length : float, int
        Launch rail length in meters.
    TesterInTheLoop.effective_1rl : float
        Original rail length minus the distance measured from nozzle exit
        to the upper rail button. It assumes the nozzle to be aligned with
        the beginning of the rail.
    TesterInTheLoop.name: str
        Name of the simulation.
    TesterInTheLoop.max_time_step : int, float
        Maximum time step to use during numerical integration in seconds.
    TesterInTheLoop.min_time_step : int, float
        Minimum time step to use during numerical integration in seconds.
    TesterInTheLoop.rtol : int, float
        Maximum relative error tolerance to be tolerated in the
        numerical integration scheme.
    TesterInTheLoop.atol : int, float
        Maximum absolute error tolerance to be tolerated in the
        integration scheme.
    TesterInTheLoop.solver : scipy.integrate.LSODA
        Scipy LSODA integration scheme.
    TesterInTheLoop.inclination : int, float
        Launch rail inclination angle relative to ground, given in degrees.
    TesterInTheLoop.heading : int, float
        Launch heading angle relative to north given in degrees.
    TesterInTheLoop.initial_solution : list
        List defines initial condition - [tInit, x_init,
        y_init, z_init, vx_init, vy_init, vz_init, e0_init, e1_init,
        e2_init, e3_init, w1_init, w2_init, w3_init]
    TesterInTheLoop.t_initial : int, float
        Initial simulation time in seconds. Usually 0.
    TesterInTheLoop.solution : list
        Solution array which keeps results from each numerical
        integration.
    TesterInTheLoop.t : float
        Current integration time.
    TesterInTheLoop.y_sol : list
        Current integration state vector u.
    TesterInTheLoop.post_processed : bool
        Defines if solution data has been post processed.
    TesterInTheLoop.initial_solution : list
        List defines initial condition - [tInit, x_init,
        y_init, z_init, vx_init, vy_init, vz_init, e0_init, e1_init,
        e2_init, e3_init, w1_init, w2_init, w3_init]
    TesterInTheLoop.out_of_rail_time : int, float
        Time, in seconds, in which the rocket completely leaves the
        rail.
    TesterInTheLoop.out_of_rail_state : list
        State vector u corresponding to state when the rocket
        completely leaves the rail.
    TesterInTheLoop.apogee_state : array
        State vector u corresponding to state when the rocket's
        vertical velocity is zero in the apogee.
    TesterInTheLoop.apogee_time : int, float
        Time, in seconds, in which the rocket's vertical velocity
        reaches zero in the apogee.
    TesterInTheLoop.apogee_x : int, float
        X coordinate (positive east) of the center of mass of the
        rocket when it reaches apogee.
    TesterInTheLoop.apogee_y : int, float
        Y coordinate (positive north) of the center of mass of the
        rocket when it reaches apogee.
    TesterInTheLoop.apogee : int, float
        Z coordinate, or altitude, of the center of mass of the
        rocket when it reaches apogee.
    TesterInTheLoop.x_impact : int, float
        X coordinate (positive east) of the center of mass of the
        rocket when it impacts ground.
    TesterInTheLoop.y_impact : int, float
        Y coordinate (positive east) of the center of mass of the
        rocket when it impacts ground.
    TesterInTheLoop.impact_velocity : int, float
        Velocity magnitude of the center of mass of the rocket when
        it impacts ground.
    TesterInTheLoop.impact_state : array
        State vector u corresponding to state when the rocket
        impacts the ground.
    TesterInTheLoop.function_evaluations : array
        List that stores number of derivative function evaluations
        during numerical integration in cumulative manner.
    TesterInTheLoop.x : Function
        Rocket's X coordinate (positive east) as a function of time.
    TesterInTheLoop.y : Function
        Rocket's Y coordinate (positive north) as a function of time.
    TesterInTheLoop.z : Function
        Rocket's z coordinate (positive up) as a function of time.
    TesterInTheLoop.vx : Function
        Velocity of the rocket's center of dry mass in the X (East) direction of
        the inertial frame as a function of time.
    TesterInTheLoop.vy : Function
        Velocity of the rocket's center of dry mass in the Y (North) direction of
        the inertial frame as a function of time.
    TesterInTheLoop.vz : Function
        Velocity of the rocket's center of dry mass in the Z (Up) direction of
        the inertial frame as a function of time.
    TesterInTheLoop.ax : Function
        Acceleration of the rocket's center of dry mass along the X (East)
        axis in the inertial frame as a function of time.
    TesterInTheLoop.ay : Function
        Acceleration of the rocket's center of dry mass along the Y (North)
        axis in the inertial frame as a function of time.
    TesterInTheLoop.az : Function
        Acceleration of the rocket's center of dry mass along the Z (Up)
        axis in the inertial frame as a function of time.
    TesterInTheLoop.e0 : Function
        Rocket's Euler parameter 0 as a function of time.
    TesterInTheLoop.e1 : Function
        Rocket's Euler parameter 1 as a function of time.
    TesterInTheLoop.e2 : Function
        Rocket's Euler parameter 2 as a function of time.
    TesterInTheLoop.e3 : Function
        Rocket's Euler parameter 3 as a function of time.
    TesterInTheLoop.w1 : Function
        Angular velocity of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch rate (q).
    TesterInTheLoop.w2 : Function
        Angular velocity of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw rate (r).
    TesterInTheLoop.w3 : Function
        Angular velocity of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll rate (p).
    TesterInTheLoop.alpha1 : Function
        Angular acceleration of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration.
    TesterInTheLoop.alpha2 : Function
        Angular acceleration of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration.
    TesterInTheLoop.alpha3 : Function
        Angular acceleration of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll acceleration.
    TesterInTheLoop.altitude : Function
        Altitude of the rocket above the ground level (AGL) in meters.
    Flight.latitude: Function
        Rocket's latitude coordinates (positive North) as a function of time.
        The Equator has a latitude equal to 0, by convention.
    Flight.longitude: Function
        Rocket's longitude coordinates (positive East) as a function of time.
        Greenwich meridian has a longitude equal to 0, by convention.
    """

    def __init__(
        self,
        rocket,
        environment,
        rail_length,
        sampling_rate,
        inclination=80.0,
        heading=90.0,
        initial_solution=None,
        max_time_step=None,
        min_time_step=0,
        rtol=1e-6,
        atol=None,
        verbose=False,
        name="Tester-In-The-Loop",
        equations_of_motion="standard",
        ode_solver="LSODA",
        burning_delay=0,
    ):
        """Set up the simulation.

        Parameters
        ----------
        rocket : Rocket
            Rocket to simulate.
        environment : Environment
            Environment to run simulation on.
        rail_length : int, float
            Length in which the rocket will be attached to the rail, only
            moving along a fixed direction, that is, the line parallel to the
            rail. Currently, if the an initial_solution is passed, the rail
            length is not used.
        sampling_rate : int, float
            Frecuency at which the simulation has to be performed. It will
            be used to calculate the time step. In hertz.
        inclination : int, float, optional
            Rail inclination angle relative to ground, given in degrees.
            Default is 80.
        heading : int, float, optional
            Heading angle relative to north given in degrees.
            Default is 90, which points in the x (east) direction.
        initial_solution : array, Flight, optional
            Initial solution array to be used. Format is:

            .. code-block:: python

                initial_solution = [
                    self.t_initial,
                    x_init, y_init, z_init,
                    vx_init, vy_init, vz_init,
                    e0_init, e1_init, e2_init, e3_init,
                    w1_init, w2_init, w3_init
                ]

            If a Flight object is used, the last state vector will be
            used as initial solution. If None, the initial solution will start
            with all null values, except for the euler parameters which will be
            calculated based on given values of inclination and heading.
            Default is None.
        max_time_step : int, float, optional
            Maximum time step to use during integration in seconds.
            Default is 0.01.
        min_time_step : int, float, optional
            Minimum time step to use during integration in seconds.
            Default is 0.01.
        rtol : float, array, optional
            Maximum relative error tolerance to be tolerated in the
            integration scheme. Can be given as array for each
            state space variable. Default is 1e-3.
        atol : float, optional
            Maximum absolute error tolerance to be tolerated in the
            integration scheme. Can be given as array for each
            state space variable. Default is 6*[1e-3] + 4*[1e-6] + 3*[1e-3].
        verbose : bool, optional
            If true, verbose mode is activated. Default is False.
        name : str, optional
            Name of the flight. Default is "Flight".
        equations_of_motion : str, optional
            Type of equations of motion to use. Can be "standard" or
            "solid_propulsion". Default is "standard". Solid propulsion is a
            more restricted set of equations of motion that only works for
            solid propulsion rockets. Such equations were used in RocketPy v0
            and are kept here for backwards compatibility.
        ode_solver : str, ``scipy.integrate.OdeSolver``, optional
            Integration method to use to solve the equations of motion ODE.
            Available options are: 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF',
            'LSODA' from ``scipy.integrate.solve_ivp``.
            Default is 'LSODA', which is recommended for most flights.
            A custom ``scipy.integrate.OdeSolver`` can be passed as well.
            For more information on the integration methods, see the scipy
            documentation.
        burning_delay : int, float, optional
            Time spent between the ignition signal is sent and the motor actually
            provides thrust. In seconds.


        Returns
        -------
        None
        """
        # Save attributes
        self.rocket = rocket
        self.env = environment
        self.rail_length = rail_length
        if self.rail_length <= 0:  # pragma: no cover
            raise ValueError("Rail length must be a positive value.")
        self.heading = heading
        self.inclination = inclination
        self.sampling_rate = sampling_rate
        self.initial_solution = initial_solution
        self.rtol = rtol
        self.atol = atol or 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3]
        self.max_time_step = max_time_step or 1 / sampling_rate
        self.min_time_step = min_time_step
        self.equations_of_motion = equations_of_motion
        self.verbose = verbose
        self.ode_solver = ode_solver
        self.name = name
        self.burning_delay = burning_delay

        # More attributes
        self.parachutes = self.rocket.parachutes

        # Simulation initialization
        self.__init_devices(self.parachutes)
        self.__init_solution_monitors()
        self.__init_equations_of_motion()
        self.__init_solver_monitors()

    def simulate_one_time_step(self):
        """Go ahead in the simulation for one time step"""

        # Calculate the max time for simulation
        max_time = self.t + 1 / self.sampling_rate

        # Check if parachutes are deployed
        parachute = self.parachute_index
        if parachute is not None and self.t > self.parachute_delay[parachute]:
            self.parachute_cd_s = self.parachutes[parachute].cd_s
            self.derivative = self.u_dot_parachute
        
        # Select the integration method
        self.solver = self._solver(
            self.derivative,
            t0=self.t,
            y0=self.y_sol,
            t_bound=max_time,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_time_step,
            min_step=self.min_time_step,
        )

        # Initialize time nodes
        self.time_nodes = Flight.TimeNodes()
        # Add first time node to the time_nodes list
        self.time_nodes.add_node(self.t, [], [], [])
        # Add last time node to the time_nodes list
        self.time_nodes.add_node(max_time, [], [], [])
        # Organize time nodes with sort() and merge()
        self.time_nodes.sort()
        self.time_nodes.merge()

        # Iterate through time nodes
        for node_index, node in self.time_iterator(self.time_nodes):
            # Determine time bound for this time node
            node.time_bound = self.time_nodes[node_index + 1].t
            self.solver.t_bound = node.time_bound
            if self.__is_lsoda:
                self.solver._lsoda_solver._integrator.rwork[0] = (
                    self.solver.t_bound
                )
                self.solver._lsoda_solver._integrator.call_args[4] = (
                    self.solver._lsoda_solver._integrator.rwork
                )
            self.solver.status = "running"

            # Step through simulation
            while self.solver.status == "running":
                # Execute solver step, log solution and function evaluations
                self.solver.step()

                # Update time and state
                self.t = self.solver.t
                self.y_sol = self.solver.y
                if self.verbose:
                    print(f"Current Simulation Time: {self.t:3.4f} s", end="\r")
                self.solution += [self.store_solution()]
                self.function_evaluations.append(self.solver.nfev)

                # Check for first out of rail event
                if len(self.out_of_rail_state) == 1 and (
                    self.y_sol[0] ** 2
                    + self.y_sol[1] ** 2
                    + (self.y_sol[2] - self.env.elevation) ** 2
                    >= self.effective_1rl**2
                ):
                    # Check exactly when it went out using root finding
                    # Disconsider elevation
                    self.solution[-2][3] -= self.env.elevation
                    self.solution[-1][3] -= self.env.elevation
                    # Get points
                    y0 = (
                        sum(self.solution[-2][i] ** 2 for i in [1, 2, 3])
                        - self.effective_1rl**2
                    )
                    yp0 = 2 * sum(
                        self.solution[-2][i] * self.solution[-2][i + 3]
                        for i in [1, 2, 3]
                    )
                    t1 = self.solution[-1][0] - self.solution[-2][0]
                    y1 = (
                        sum(self.solution[-1][i] ** 2 for i in [1, 2, 3])
                        - self.effective_1rl**2
                    )
                    yp1 = 2 * sum(
                        self.solution[-1][i] * self.solution[-1][i + 3]
                        for i in [1, 2, 3]
                    )
                    # Put elevation back
                    self.solution[-2][3] += self.env.elevation
                    self.solution[-1][3] += self.env.elevation
                    # Cubic Hermite interpolation (ax**3 + bx**2 + cx + d)
                    a, b, c, d = calculate_cubic_hermite_coefficients(
                        0,
                        float(self.solver.step_size),
                        y0,
                        yp0,
                        y1,
                        yp1,
                    )
                    a += 1e-5  # TODO: why??
                    # Find roots
                    t_roots = find_roots_cubic_function(a, b, c, d)
                    # Find correct root
                    valid_t_root = [
                        t_root.real
                        for t_root in t_roots
                        if 0 < t_root.real < t1 and abs(t_root.imag) < 0.001
                    ]
                    if len(valid_t_root) > 1:  # pragma: no cover
                        raise ValueError(
                            "Multiple roots found when solving for rail exit time."
                        )
                    if len(valid_t_root) == 0:  # pragma: no cover
                        raise ValueError(
                            "No valid roots found when solving for rail exit time."
                        )
                    # Determine final state when upper button is going out of rail
                    self.t = valid_t_root[0] + self.solution[-2][0]
                    interpolator = self.solver.dense_output()
                    self.y_sol = interpolator(self.t)
                    self.solution[-1] = self.store_solution()
                    self.out_of_rail_time = self.t
                    self.out_of_rail_time_index = len(self.solution) - 1
                    self.out_of_rail_state = self.y_sol
                    # Prepare to leave loops and start new flight phase
                    self.time_nodes.flush_after(node_index)
                    self.time_nodes.add_node(self.t, [], [], [])
                    self.solver.status = "finished"
                    self.derivative = self.u_dot_generalized

                # Check for apogee event
                # TODO: negative vz doesn't really mean apogee. Improve this.
                if len(self.apogee_state) == 1 and self.y_sol[5] < 0:
                    # Assume linear vz(t) to detect when vz = 0
                    t0, vz0 = self.solution[-2][0], self.solution[-2][6]
                    t1, vz1 = self.solution[-1][0], self.solution[-1][6]
                    t_root = find_root_linear_interpolation(t0, t1, vz0, vz1, 0)
                    # Fetch state at t_root
                    interpolator = self.solver.dense_output()
                    self.apogee_state = interpolator(t_root)
                    # Store apogee data
                    self.apogee_time = t_root
                    self.apogee_x = self.apogee_state[0]
                    self.apogee_y = self.apogee_state[1]
                    self.apogee = self.apogee_state[2]

                    # adding the apogee state to solution increases accuracy
                    # we can only do this if the apogee is not the first state
                    u_dot = self.derivative(t_root, self.apogee_state)
                    self.solution.insert(-1, [
                        t_root, 
                        *self.apogee_state[:6].flatten(),
                        *u_dot[3:6],
                        *self.apogee_state[6:].flatten(),
                        *u_dot[-3:],
                    ])

                # Check for impact event
                if self.y_sol[2] < self.env.elevation:
                    # Check exactly when it happened using root finding
                    # Cubic Hermite interpolation (ax**3 + bx**2 + cx + d)
                    a, b, c, d = calculate_cubic_hermite_coefficients(
                        x0=0,  # t0
                        x1=float(self.solver.step_size),  # t1 - t0
                        y0=float(self.solution[-2][3] - self.env.elevation),  # z0
                        yp0=float(self.solution[-2][6]),  # vz0
                        y1=float(self.solution[-1][3] - self.env.elevation),  # z1
                        yp1=float(self.solution[-1][6]),  # vz1
                    )
                    # Find roots
                    t_roots = find_roots_cubic_function(a, b, c, d)
                    # Find correct root
                    t1 = self.solution[-1][0] - self.solution[-2][0]
                    valid_t_root = [
                        t_root.real
                        for t_root in t_roots
                        if abs(t_root.imag) < 0.001 and 0 < t_root.real < t1
                    ]
                    if len(valid_t_root) > 1:  # pragma: no cover
                        raise ValueError(
                            "Multiple roots found when solving for impact time."
                        )
                    # Determine impact state at t_root
                    self.t = self.t_final = valid_t_root[0] + self.solution[-2][0]
                    interpolator = self.solver.dense_output()
                    self.y_sol = self.impact_state = interpolator(self.t)
                    # Roll back solution
                    self.solution[-1] = self.store_solution()
                    # Save impact state
                    self.x_impact = self.impact_state[0]
                    self.y_impact = self.impact_state[1]
                    self.z_impact = self.impact_state[2]
                    self.impact_velocity = self.impact_state[5]
                    # Prepare to leave loops and start new flight phase
                    self.time_nodes.flush_after(node_index)
                    self.time_nodes.add_node(self.t, [], [], [])
                    self.solver.status = "finished"

        if self.verbose:
            print(f"\n>>> Simulation Completed at Time: {self.t:3.4f} s")

    def __init_solution_monitors(self):
        # Initialize solution monitors
        self.out_of_rail_time = 0
        self.out_of_rail_time_index = 0
        self.out_of_rail_state = np.array([0])
        self.apogee_state = np.array([0])
        self.apogee_time = 0
        self.x_impact = 0
        self.y_impact = 0
        self.impact_velocity = 0
        self.impact_state = np.array([0])
        self.post_processed = False
        self.__post_processed_variables = []

    def __init_equations_of_motion(self):
        """Initialize equations of motion."""
        if self.equations_of_motion == "solid_propulsion":
            # NOTE: The u_dot is faster, but only works for solid propulsion
            self.u_dot_generalized = self.u_dot

    def __init_solver_monitors(self):
        # Initialize solver monitors
        self.function_evaluations = []
        # Initialize solution state
        self.solution = []
        self.__init_flight_state()

        self.t_initial = self.initial_solution[0]
        self.t = self.initial_solution[0]
        self.y_sol = self.initial_solution[1:]
        self.solution += [self.store_solution()]

        self.__set_ode_solver(self.ode_solver)

    def time_iterator(self, node_list):
        i = 0
        while i < len(node_list) - 1:
            yield i, node_list[i]
            i += 1
    
    def __init_flight_state(self):
        """Initialize flight state variables."""
        if self.initial_solution is None:
            # Initialize time and state variables
            self.t_initial = 0
            x_init, y_init, z_init = 0, 0, self.env.elevation
            vx_init, vy_init, vz_init = 0, 0, 0
            w1_init, w2_init, w3_init = 0, 0, 0
            # Initialize attitude
            # Precession / Heading Angle
            self.psi_init = np.radians(-self.heading)
            # Nutation / Attitude Angle
            self.theta_init = np.radians(self.inclination - 90)
            # Spin / Bank Angle
            self.phi_init = 0

            # Consider Rail Buttons position, if there is rail buttons
            try:
                self.phi_init += (
                    self.rocket.rail_buttons[0].component.angular_position_rad
                    if self.rocket._csys == 1
                    else 2 * np.pi
                    - self.rocket.rail_buttons[0].component.angular_position_rad
                )
            except IndexError:
                pass

            # 3-1-3 Euler Angles to Euler Parameters
            e0_init, e1_init, e2_init, e3_init = euler313_to_quaternions(
                self.phi_init, self.theta_init, self.psi_init
            )
            # Store initial conditions
            self.initial_solution = [
                self.t_initial,
                x_init,
                y_init,
                z_init,
                vx_init,
                vy_init,
                vz_init,
                e0_init,
                e1_init,
                e2_init,
                e3_init,
                w1_init,
                w2_init,
                w3_init,
            ]
            # Set initial derivative for rail phase
            self.derivative = self.udot_rail1
            self.motor_on = False
        elif isinstance(self.initial_solution, Flight):
            # Initialize time and state variables based on last solution of
            # previous flight
            self.initial_solution = self.initial_solution.solution[-1]
            # Set unused monitors
            self.out_of_rail_state = self.initial_solution[1:]
            self.out_of_rail_time = self.initial_solution[0]
            self.out_of_rail_time_index = 0
            # Set initial derivative for 6-DOF flight phase
            self.derivative = self.u_dot_generalized
        else:
            # Initial solution given, ignore rail phase
            # TODO: Check if rocket is actually out of rail. Otherwise, start at rail
            self.out_of_rail_state = self.initial_solution[1:]
            self.out_of_rail_time = self.initial_solution[0]
            self.out_of_rail_time_index = 0
            self.derivative = self.u_dot_generalized

    def __set_ode_solver(self, solver):
        """Sets the ODE solver to be used in the simulation.

        Parameters
        ----------
        solver : str, ``scipy.integrate.OdeSolver``
            Integration method to use to solve the equations of motion ODE,
            or a custom ``scipy.integrate.OdeSolver``.
        """
        if isinstance(solver, OdeSolver):
            self._solver = solver
        else:
            try:
                self._solver = ODE_SOLVER_MAP[solver]
            except KeyError as e:  # pragma: no cover
                raise ValueError(
                    f"Invalid ``ode_solver`` input: {solver}. "
                    f"Available options are: {', '.join(ODE_SOLVER_MAP.keys())}"
                ) from e

        self.__is_lsoda = issubclass(self._solver, LSODA)

    def udot_rail1(self, t, u):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 1 DOF motion in the rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].

        Return
        ------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].

        """
        # Retrieve integration data
        _, _, z, vx, vy, vz, e0, e1, e2, e3, _, _, _ = u

        # Motor burning time
        if self.motor_on:
            burning_time = t - self.burning_delay
        else:
            burning_time = 0
        
        # Retrieve important quantities
        # Mass
        total_mass_at_t = self.rocket.total_mass.get_value_opt(burning_time)

        # Get freestream speed
        free_stream_speed = (
            (self.env.wind_velocity_x.get_value_opt(z) - vx) ** 2
            + (self.env.wind_velocity_y.get_value_opt(z) - vy) ** 2
            + (vz) ** 2
        ) ** 0.5
        free_stream_mach = free_stream_speed / self.env.speed_of_sound.get_value_opt(z)
        drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)

        # Calculate Forces
        pressure = self.env.pressure.get_value_opt(z)
        net_thrust = max(
            self.rocket.motor.thrust.get_value_opt(burning_time)
            + self.rocket.motor.pressure_thrust(pressure),
            0,
        )
        rho = self.env.density.get_value_opt(z)
        R3 = -0.5 * rho * (free_stream_speed**2) * self.rocket.area * (drag_coeff)

        # Calculate Linear acceleration
        a3 = (R3 + net_thrust) / total_mass_at_t - (
            e0**2 - e1**2 - e2**2 + e3**2
        ) * self.env.gravity.get_value_opt(z)
        if a3 > 0:
            ax = 2 * (e1 * e3 + e0 * e2) * a3
            ay = 2 * (e2 * e3 - e0 * e1) * a3
            az = (1 - 2 * (e1**2 + e2**2)) * a3
        else:
            ax, ay, az = 0, 0, 0

        self.__post_processed_variables.append([t, self.air_brake_deployment_level])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    def u_dot(self, t, u):  # pylint: disable=too-many-locals,too-many-statements
        """Calculates derivative of u state vector with respect to time
        when rocket is flying in 6 DOF motion during ascent out of rail
        and descent without parachute.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].

        Returns
        -------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].
        """

        # Retrieve integration data
        _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
        # Determine lift force and moment
        R1, R2, M1, M2, M3 = 0, 0, 0, 0, 0
        # Motor burning time
        if self.motor_on:
            burning_time = t - self.burning_delay
        else:
            burning_time = 0
        # Thrust correction parameters
        pressure = self.env.pressure.get_value_opt(z)
        # Determine current behavior
        if self.rocket.motor.burn_start_time < burning_time < self.rocket.motor.burn_out_time:
            # Motor burning
            # Retrieve important motor quantities
            # Inertias
            motor_I_33_at_t = self.rocket.motor.I_33.get_value_opt(burning_time)
            motor_I_11_at_t = self.rocket.motor.I_11.get_value_opt(burning_time)
            motor_I_33_derivative_at_t = self.rocket.motor.I_33.differentiate(
                burning_time, dx=1e-6
            )
            motor_I_11_derivative_at_t = self.rocket.motor.I_11.differentiate(
                burning_time, dx=1e-6
            )
            # Mass
            mass_flow_rate_at_t = self.rocket.motor.mass_flow_rate.get_value_opt(burning_time)
            propellant_mass_at_t = self.rocket.motor.propellant_mass.get_value_opt(burning_time)
            # Thrust
            net_thrust = max(
                self.rocket.motor.thrust.get_value_opt(burning_time)
                + self.rocket.motor.pressure_thrust(pressure),
                0,
            )
            # Off center moment
            M1 += self.rocket.thrust_eccentricity_y * net_thrust
            M2 -= self.rocket.thrust_eccentricity_x * net_thrust
        else:
            # Motor stopped
            # Inertias
            (
                motor_I_33_at_t,
                motor_I_11_at_t,
                motor_I_33_derivative_at_t,
                motor_I_11_derivative_at_t,
            ) = (0, 0, 0, 0)
            # Mass
            mass_flow_rate_at_t, propellant_mass_at_t = 0, 0
            # thrust
            net_thrust = 0

        # Retrieve important quantities
        # Inertias
        rocket_dry_I_33 = self.rocket.dry_I_33
        rocket_dry_I_11 = self.rocket.dry_I_11
        # Mass
        rocket_dry_mass = self.rocket.dry_mass  # already with motor's dry mass
        total_mass_at_t = propellant_mass_at_t + rocket_dry_mass
        mu = (propellant_mass_at_t * rocket_dry_mass) / (
            propellant_mass_at_t + rocket_dry_mass
        )
        # Geometry
        # b = -self.rocket.distance_rocket_propellant
        b = (
            -(
                self.rocket.center_of_propellant_position.get_value_opt(0)
                - self.rocket.center_of_dry_mass_position
            )
            * self.rocket._csys
        )
        c = self.rocket.nozzle_to_cdm
        nozzle_radius = self.rocket.motor.nozzle_radius
        # Prepare transformation matrix
        a11 = 1 - 2 * (e2**2 + e3**2)
        a12 = 2 * (e1 * e2 - e0 * e3)
        a13 = 2 * (e1 * e3 + e0 * e2)
        a21 = 2 * (e1 * e2 + e0 * e3)
        a22 = 1 - 2 * (e1**2 + e3**2)
        a23 = 2 * (e2 * e3 - e0 * e1)
        a31 = 2 * (e1 * e3 - e0 * e2)
        a32 = 2 * (e2 * e3 + e0 * e1)
        a33 = 1 - 2 * (e1**2 + e2**2)
        # Transformation matrix: (123) -> (XYZ)
        K = Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        Kt = K.transpose

        # Calculate Forces and Moments
        # Get freestream speed
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        speed_of_sound = self.env.speed_of_sound.get_value_opt(z)
        free_stream_speed = (
            (wind_velocity_x - vx) ** 2 + (wind_velocity_y - vy) ** 2 + (vz) ** 2
        ) ** 0.5
        free_stream_mach = free_stream_speed / speed_of_sound

        # Determine aerodynamics forces
        # Determine Drag Force
        if burning_time < self.rocket.motor.burn_out_time:
            drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)
        else:
            drag_coeff = self.rocket.power_off_drag.get_value_opt(free_stream_mach)
        rho = self.env.density.get_value_opt(z)
        R3 = -0.5 * rho * (free_stream_speed**2) * self.rocket.area * drag_coeff
        for air_brakes in self.rocket.air_brakes:
            if self.air_brake_deployment_level > 0:
                air_brakes_cd = air_brakes.drag_coefficient.get_value_opt(
                    self.air_brake_deployment_level, free_stream_mach
                )
                air_brakes_force = (
                    -0.5
                    * rho
                    * (free_stream_speed**2)
                    * air_brakes.reference_area
                    * air_brakes_cd
                )
                if air_brakes.override_rocket_drag:
                    R3 = air_brakes_force  # Substitutes rocket drag coefficient
                else:
                    R3 += air_brakes_force
        # Off center moment
        M1 += self.rocket.cp_eccentricity_y * R3
        M2 -= self.rocket.cp_eccentricity_x * R3
        # Get rocket velocity in body frame
        vx_b = a11 * vx + a21 * vy + a31 * vz
        vy_b = a12 * vx + a22 * vy + a32 * vz
        vz_b = a13 * vx + a23 * vy + a33 * vz
        # Calculate lift and moment for each component of the rocket
        velocity_in_body_frame = Vector([vx_b, vy_b, vz_b])
        w = Vector([omega1, omega2, omega3])
        for aero_surface, _ in self.rocket.aerodynamic_surfaces:
            # Component cp relative to CDM in body frame
            comp_cp = self.rocket.surfaces_cp_to_cdm[aero_surface]
            # Component absolute velocity in body frame
            comp_vb = velocity_in_body_frame + (w ^ comp_cp)
            # Wind velocity at component altitude
            comp_z = z + (K @ comp_cp).z
            comp_wind_vx = self.env.wind_velocity_x.get_value_opt(comp_z)
            comp_wind_vy = self.env.wind_velocity_y.get_value_opt(comp_z)
            # Component freestream velocity in body frame
            comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
            comp_stream_velocity = comp_wind_vb - comp_vb
            comp_stream_speed = abs(comp_stream_velocity)
            comp_stream_mach = comp_stream_speed / speed_of_sound
            # Reynolds at component altitude
            # TODO: Reynolds is only used in generic surfaces. This calculation
            # should be moved to the surface class for efficiency
            comp_reynolds = (
                self.env.density.get_value_opt(comp_z)
                * comp_stream_speed
                * aero_surface.reference_length
                / self.env.dynamic_viscosity.get_value_opt(comp_z)
            )
            # Forces and moments
            X, Y, Z, M, N, L = aero_surface.compute_forces_and_moments(
                comp_stream_velocity,
                comp_stream_speed,
                comp_stream_mach,
                rho,
                comp_cp,
                w,
                comp_reynolds,
            )
            R1 += X
            R2 += Y
            R3 += Z
            M1 += M
            M2 += N
            M3 += L
        # Off center moment
        M3 += self.rocket.cp_eccentricity_x * R2 - self.rocket.cp_eccentricity_y * R1

        # Calculate derivatives
        # Angular acceleration
        alpha1 = (
            M1
            - (
                omega2
                * omega3
                * (
                    rocket_dry_I_33
                    + motor_I_33_at_t
                    - rocket_dry_I_11
                    - motor_I_11_at_t
                    - mu * b**2
                )
                + omega1
                * (
                    (
                        motor_I_11_derivative_at_t
                        + mass_flow_rate_at_t
                        * (rocket_dry_mass - 1)
                        * (b / total_mass_at_t) ** 2
                    )
                    - mass_flow_rate_at_t
                    * ((nozzle_radius / 2) ** 2 + (c - b * mu / rocket_dry_mass) ** 2)
                )
            )
        ) / (rocket_dry_I_11 + motor_I_11_at_t + mu * b**2)
        alpha2 = (
            M2
            - (
                omega1
                * omega3
                * (
                    rocket_dry_I_11
                    + motor_I_11_at_t
                    + mu * b**2
                    - rocket_dry_I_33
                    - motor_I_33_at_t
                )
                + omega2
                * (
                    (
                        motor_I_11_derivative_at_t
                        + mass_flow_rate_at_t
                        * (rocket_dry_mass - 1)
                        * (b / total_mass_at_t) ** 2
                    )
                    - mass_flow_rate_at_t
                    * ((nozzle_radius / 2) ** 2 + (c - b * mu / rocket_dry_mass) ** 2)
                )
            )
        ) / (rocket_dry_I_11 + motor_I_11_at_t + mu * b**2)
        alpha3 = (
            M3
            - omega3
            * (
                motor_I_33_derivative_at_t
                - mass_flow_rate_at_t * (nozzle_radius**2) / 2
            )
        ) / (rocket_dry_I_33 + motor_I_33_at_t)
        # Euler parameters derivative
        e0dot = 0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3)
        e1dot = 0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3)
        e2dot = 0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3)
        e3dot = 0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2)

        # Linear acceleration
        L = [
            (
                R1
                - b * propellant_mass_at_t * (omega2**2 + omega3**2)
                - 2 * c * mass_flow_rate_at_t * omega2
            )
            / total_mass_at_t,
            (
                R2
                + b * propellant_mass_at_t * (alpha3 + omega1 * omega2)
                + 2 * c * mass_flow_rate_at_t * omega1
            )
            / total_mass_at_t,
            (R3 - b * propellant_mass_at_t * (alpha2 - omega1 * omega3) + net_thrust)
            / total_mass_at_t,
        ]
        ax, ay, az = K @ Vector(L)
        az -= self.env.gravity.get_value_opt(z)  # Include gravity

        # Create u_dot
        u_dot = [
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            e0dot,
            e1dot,
            e2dot,
            e3dot,
            alpha1,
            alpha2,
            alpha3,
        ]

        self.__post_processed_variables.append([t, self.air_brake_deployment_level])

        return u_dot

    def u_dot_generalized(self, t, u):  # pylint: disable=too-many-locals,too-many-statements
        """Calculates derivative of u state vector with respect to time when the
        rocket is flying in 6 DOF motion in space and significant mass variation
        effects exist. Typical flight phases include powered ascent after launch
        rail.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, q0, q1,
            q2, q3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self variables
            such as self.angle_of_attack, by default False.

        Returns
        -------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0_dot, e1_dot, e2_dot, e3_dot, alpha1, alpha2, alpha3].
        """
        # Retrieve integration data
        _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

        # Create necessary vectors
        # r = Vector([x, y, z])               # CDM position vector
        v = Vector([vx, vy, vz])  # CDM velocity vector
        e = [e0, e1, e2, e3]  # Euler parameters/quaternions
        w = Vector([omega1, omega2, omega3])  # Angular velocity vector

        # Motor burning time
        if self.motor_on:
            burning_time = t - self.burning_delay
        else:
            burning_time = 0

        # Retrieve necessary quantities
        ## Rocket mass
        total_mass = self.rocket.total_mass.get_value_opt(burning_time)
        total_mass_dot = self.rocket.total_mass_flow_rate.get_value_opt(burning_time)
        total_mass_ddot = self.rocket.total_mass_flow_rate.differentiate_complex_step(burning_time)
        ## CM position vector and time derivatives relative to CDM in body frame
        r_CM_z = self.rocket.com_to_cdm_function
        r_CM_t = r_CM_z.get_value_opt(burning_time)
        r_CM = Vector([0, 0, r_CM_t])
        r_CM_dot = Vector([0, 0, r_CM_z.differentiate_complex_step(burning_time)])
        r_CM_ddot = Vector([0, 0, r_CM_z.differentiate(burning_time, order=2)])
        ## Nozzle position vector
        r_NOZ = Vector([0, 0, self.rocket.nozzle_to_cdm])
        ## Nozzle gyration tensor
        S_nozzle = self.rocket.nozzle_gyration_tensor
        ## Inertia tensor
        inertia_tensor = self.rocket.get_inertia_tensor_at_time(burning_time)
        ## Inertia tensor time derivative in the body frame
        I_dot = self.rocket.get_inertia_tensor_derivative_at_time(burning_time)

        # Calculate the Inertia tensor relative to CM
        H = (r_CM.cross_matrix @ -r_CM.cross_matrix) * total_mass
        I_CM = inertia_tensor - H

        # Prepare transformation matrices
        K = Matrix.transformation(e)
        Kt = K.transpose

        # Compute aerodynamic forces and moments
        R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

        ## Drag force
        rho = self.env.density.get_value_opt(z)
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)
        wind_velocity = Vector([wind_velocity_x, wind_velocity_y, 0])
        free_stream_speed = abs((wind_velocity - Vector(v)))
        speed_of_sound = self.env.speed_of_sound.get_value_opt(z)
        free_stream_mach = free_stream_speed / speed_of_sound

        if self.rocket.motor.burn_start_time < burning_time < self.rocket.motor.burn_out_time:
            pressure = self.env.pressure.get_value_opt(z)
            net_thrust = max(
                self.rocket.motor.thrust.get_value_opt(burning_time)
                + self.rocket.motor.pressure_thrust(pressure),
                0,
            )
            drag_coeff = self.rocket.power_on_drag.get_value_opt(free_stream_mach)
        else:
            net_thrust = 0
            drag_coeff = self.rocket.power_off_drag.get_value_opt(free_stream_mach)
        R3 += -0.5 * rho * (free_stream_speed**2) * self.rocket.area * drag_coeff
        for air_brakes in self.rocket.air_brakes:
            if self.air_brake_deployment_level > 0:
                air_brakes_cd = air_brakes.drag_coefficient.get_value_opt(
                    self.air_brake_deployment_level, free_stream_mach
                )
                air_brakes_force = (
                    -0.5
                    * rho
                    * (free_stream_speed**2)
                    * air_brakes.reference_area
                    * air_brakes_cd
                )
                if air_brakes.override_rocket_drag:
                    R3 = air_brakes_force  # Substitutes rocket drag coefficient
                else:
                    R3 += air_brakes_force
        # Get rocket velocity in body frame
        velocity_in_body_frame = Kt @ v
        # Calculate lift and moment for each component of the rocket
        for aero_surface, _ in self.rocket.aerodynamic_surfaces:
            # Component cp relative to CDM in body frame
            comp_cp = self.rocket.surfaces_cp_to_cdm[aero_surface]
            # Component absolute velocity in body frame
            comp_vb = velocity_in_body_frame + (w ^ comp_cp)
            # Wind velocity at component altitude
            comp_z = z + (K @ comp_cp).z
            comp_wind_vx = self.env.wind_velocity_x.get_value_opt(comp_z)
            comp_wind_vy = self.env.wind_velocity_y.get_value_opt(comp_z)
            # Component freestream velocity in body frame
            comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
            comp_stream_velocity = comp_wind_vb - comp_vb
            comp_stream_speed = abs(comp_stream_velocity)
            comp_stream_mach = comp_stream_speed / speed_of_sound
            # Reynolds at component altitude
            # TODO: Reynolds is only used in generic surfaces. This calculation
            # should be moved to the surface class for efficiency
            comp_reynolds = (
                self.env.density.get_value_opt(comp_z)
                * comp_stream_speed
                * aero_surface.reference_length
                / self.env.dynamic_viscosity.get_value_opt(comp_z)
            )
            # Forces and moments
            X, Y, Z, M, N, L = aero_surface.compute_forces_and_moments(
                comp_stream_velocity,
                comp_stream_speed,
                comp_stream_mach,
                rho,
                comp_cp,
                w,
                comp_reynolds,
            )
            R1 += X
            R2 += Y
            R3 += Z
            M1 += M
            M2 += N
            M3 += L

        # Off center moment
        M1 += (
            self.rocket.cp_eccentricity_y * R3
            + self.rocket.thrust_eccentricity_y * net_thrust
        )
        M2 -= (
            self.rocket.cp_eccentricity_x * R3
            + self.rocket.thrust_eccentricity_x * net_thrust
        )
        M3 += self.rocket.cp_eccentricity_x * R2 - self.rocket.cp_eccentricity_y * R1

        weight_in_body_frame = Kt @ Vector(
            [0, 0, -total_mass * self.env.gravity.get_value_opt(z)]
        )

        T00 = total_mass * r_CM
        T03 = 2 * total_mass_dot * (r_NOZ - r_CM) - 2 * total_mass * r_CM_dot
        T04 = (
            Vector([0, 0, net_thrust])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (r_NOZ - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = (
            ((w ^ T00) ^ w)
            + (w ^ T03)
            + T04
            + weight_in_body_frame
            + Vector([R1, R2, R3])
        )

        T21 = (
            ((inertia_tensor @ w) ^ w)
            + T05 @ w
            - (weight_in_body_frame ^ r_CM)
            + Vector([M1, M2, M3])
        )

        # Angular velocity derivative
        w_dot = I_CM.inverse @ (T21 + (T20 ^ r_CM))

        # Velocity vector derivative
        v_dot = K @ (T20 / total_mass - (r_CM ^ w_dot))

        # Euler parameters derivative
        e_dot = [
            0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
            0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
            0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
            0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
        ]

        # Position vector derivative
        r_dot = [vx, vy, vz]

        # Create u_dot
        u_dot = [*r_dot, *v_dot, *e_dot, *w_dot]

        self.__post_processed_variables.append([t, self.air_brake_deployment_level])

        return u_dot
    
    def u_dot_parachute(self, t, u):
        """Calculates derivative of u state vector with respect to time
        when rocket is flying under parachute. A 3 DOF approximation is
        used.

        Parameters
        ----------
        t : float
            Time in seconds
        u : list
            State vector defined by u = [x, y, z, vx, vy, vz, e0, e1,
            e2, e3, omega1, omega2, omega3].
        post_processing : bool, optional
            If True, adds flight data information directly to self
            variables such as self.angle_of_attack. Default is False.

        Return
        ------
        u_dot : list
            State vector defined by u_dot = [vx, vy, vz, ax, ay, az,
            e0dot, e1dot, e2dot, e3dot, alpha1, alpha2, alpha3].

        """
        # Get relevant state data
        z, vx, vy, vz = u[2:6]

        # Get atmospheric data
        rho = self.env.density.get_value_opt(z)
        wind_velocity_x = self.env.wind_velocity_x.get_value_opt(z)
        wind_velocity_y = self.env.wind_velocity_y.get_value_opt(z)

        # Get Parachute data
        cd_s = self.parachute_cd_s

        # Get the mass of the rocket
        mp = self.rocket.dry_mass

        # Define constants
        ka = 1  # Added mass coefficient (depends on parachute's porosity)
        R = 1.5  # Parachute radius
        # to = 1.2
        # eta = 1
        # Rdot = (6 * R * (1 - eta) / (1.2**6)) * (
        #     (1 - eta) * t**5 + eta * (to**3) * (t**2)
        # )
        # Rdot = 0

        # Calculate added mass
        ma = ka * rho * (4 / 3) * np.pi * R**3

        # Calculate freestream speed
        freestream_x = vx - wind_velocity_x
        freestream_y = vy - wind_velocity_y
        freestream_z = vz
        free_stream_speed = (freestream_x**2 + freestream_y**2 + freestream_z**2) ** 0.5

        # Determine drag force
        pseudo_drag = -0.5 * rho * cd_s * free_stream_speed
        # pseudo_drag = pseudo_drag - ka * rho * 4 * np.pi * (R**2) * Rdot
        Dx = pseudo_drag * freestream_x
        Dy = pseudo_drag * freestream_y
        Dz = pseudo_drag * freestream_z
        ax = Dx / (mp + ma)
        ay = Dy / (mp + ma)
        az = (Dz - 9.8 * mp) / (mp + ma)

        self.__post_processed_variables.append([t, self.air_brake_deployment_level])

        return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]

    @cached_property
    def effective_1rl(self):
        """Original rail length minus the distance measured from nozzle exit
        to the upper rail button. It assumes the nozzle to be aligned with
        the beginning of the rail."""
        nozzle = self.rocket.nozzle_position
        try:
            rail_buttons = self.rocket.rail_buttons[0]
            upper_r_button = (
                rail_buttons.component.buttons_distance * self.rocket._csys
                + rail_buttons.position.z
            )
        except IndexError:  # No rail buttons defined
            upper_r_button = nozzle
        effective_1rl = self.rail_length - abs(nozzle - upper_r_button)
        return effective_1rl

    def __init_devices(self, parachutes):
        """Initializes parameters for the motor, the airbrake and the parachutes"""
        self.motor_triggered_at = None
        self.air_brake_deployment_level = 0
        self.parachute_triggered_at = [None] * len(parachutes)
        self.parachute_index = None
        self.parachute_delay = []
        for parachute in parachutes:
            self.parachute_delay.append(parachute.lag)

    def store_solution(self):
        """Does a list with the solution of the simulation"""

        u_dot = self.derivative(self.t, self.y_sol)

        return [
            self.t,         # flight time
            self.y_sol[0],  # x
            self.y_sol[1],  # y
            self.y_sol[2],  # z
            self.y_sol[3],  # vx
            self.y_sol[4],  # vy
            self.y_sol[5],  # vz
            u_dot[3],       # ax
            u_dot[4],       # ay
            u_dot[5],       # az
            self.y_sol[6],  # e0
            self.y_sol[7],  # e1
            self.y_sol[8],  # e2
            self.y_sol[9],  # e3
            self.y_sol[10], # omega1
            self.y_sol[11], # omega2
            self.y_sol[12], # omega3
            u_dot[10],      # alpha1
            u_dot[11],      # alpha2
            u_dot[12],      # alpha3
        ]

    def apply_input_data(self, input):
        """Takes the data from avionics and process it"""

        self.motor_burning_signal = input[0]
        self.air_brake_deployment_level = input[1]
        self.parachute_deployment_signal = input[2]

        # Check changes in status
        if self.motor_burning_signal and self.motor_triggered_at is None:
            self.motor_on = True
            self.burning_delay += self.t
            self.motor_triggered_at = self.t
        
        for parachute, deployment_signal in enumerate(self.parachute_deployment_signal):
            if deployment_signal and self.parachute_triggered_at[parachute] is None:
                self.parachute_index = parachute
                self.parachute_delay[parachute] += self.t
                self.parachute_triggered_at[parachute] = self.t

    def generate_output_data(self):
        # TODO: Add sensors and generate data
        a=1
    
    # Functions for the solution
    @funcify_method("Time (s)", "X (m)", "spline", "constant")
    def x(self):
        """Rocket x position relative to the launch pad as a Function of
        time."""
        return np.array(self.solution)[:, [0, 1]]

    @funcify_method("Time (s)", "Y (m)", "spline", "constant")
    def y(self):
        """Rocket y position relative to the lauch pad as a Function of
        time."""
        return np.array(self.solution)[:, [0, 2]]

    @funcify_method("Time (s)", "Z (m)", "spline", "constant")
    def z(self):
        """Rocket z position relative to the launch pad as a Function of
        time."""
        return np.array(self.solution)[:, [0, 3]]

    @funcify_method("Time (s)", "Altitude AGL (m)", "spline", "constant")
    def altitude(self):
        """Rocket altitude above ground level as a Function of time. Ground
        level is defined by the environment elevation."""
        return self.z - self.env.elevation

    @funcify_method("Time (s)", "Vx (m/s)", "spline", "zero")
    def vx(self):
        """Velocity of the rocket's center of dry mass in the X (East) direction
        of the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 4]]

    @funcify_method("Time (s)", "Vy (m/s)", "spline", "zero")
    def vy(self):
        """Velocity of the rocket's center of dry mass in the Y (North)
        direction of the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 5]]

    @funcify_method("Time (s)", "Vz (m/s)", "spline", "zero")
    def vz(self):
        """Velocity of the rocket's center of dry mass in the Z (Up) direction of
        the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 6]]
    
    @funcify_method("Time (s)", "Ax (m/s²)", "spline", "zero")
    def ax(self):
        """Acceleration of the rocket's center of dry mass along the X (East)
        axis in the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 7]]

    @funcify_method("Time (s)", "Ay (m/s²)", "spline", "zero")
    def ay(self):
        """Acceleration of the rocket's center of dry mass along the Y (North)
        axis in the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 8]]

    @funcify_method("Time (s)", "Az (m/s²)", "spline", "zero")
    def az(self):
        """Acceleration of the rocket's center of dry mass along the Z (Up)
        axis in the inertial frame as a function of time."""
        return np.array(self.solution)[:, [0, 9]]

    @funcify_method("Time (s)", "e0", "spline", "constant")
    def e0(self):
        """Rocket quaternion e0 as a Function of time."""
        return np.array(self.solution)[:, [0, 10]]

    @funcify_method("Time (s)", "e1", "spline", "constant")
    def e1(self):
        """Rocket quaternion e1 as a Function of time."""
        return np.array(self.solution)[:, [0, 11]]

    @funcify_method("Time (s)", "e2", "spline", "constant")
    def e2(self):
        """Rocket quaternion e2 as a Function of time."""
        return np.array(self.solution)[:, [0, 12]]

    @funcify_method("Time (s)", "e3", "spline", "constant")
    def e3(self):
        """Rocket quaternion e3 as a Function of time."""
        return np.array(self.solution)[:, [0, 13]]

    @funcify_method("Time (s)", "ω1 (rad/s)", "spline", "zero")
    def w1(self):
        """Angular velocity of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch rate (q)."""
        return np.array(self.solution)[:, [0, 14]]

    @funcify_method("Time (s)", "ω2 (rad/s)", "spline", "zero")
    def w2(self):
        """Angular velocity of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw rate (r)."""
        return np.array(self.solution)[:, [0, 15]]

    @funcify_method("Time (s)", "ω3 (rad/s)", "spline", "zero")
    def w3(self):
        """Angular velocity of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll rate (p)."""
        return np.array(self.solution)[:, [0, 16]]
    
    @funcify_method("Time (s)", "α1 (rad/s²)", "spline", "zero")
    def alpha1(self):
        """Angular acceleration of the rocket in the x direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        pitch acceleration."""
        return np.array(self.solution)[:, [0, 17]]

    @funcify_method("Time (s)", "α2 (rad/s²)", "spline", "zero")
    def alpha2(self):
        """Angular acceleration of the rocket in the y direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        yaw acceleration."""
        return np.array(self.solution)[:, [0, 18]]

    @funcify_method("Time (s)", "α3 (rad/s²)", "spline", "zero")
    def alpha3(self):
        """Angular acceleration of the rocket in the z direction of the rocket's
        body frame as a function of time, in rad/s. Sometimes referred to as
        roll acceleration."""
        return np.array(self.solution)[:, [0, 19]]
    
    @cached_property
    def time(self):
        """Returns time array from solution."""
        return np.array(self.solution)[:, 0]

    @funcify_method(
        "Time (s)", "Horizontal Distance to Launch Point (m)", "spline", "constant"
    )
    def drift(self):
        """Rocket horizontal distance to tha launch point, in meters, as a
        Function of time."""
        return np.column_stack(
            (self.time, (self.x[:, 1] ** 2 + self.y[:, 1] ** 2) ** 0.5)
        )

    @funcify_method("Time (s)", "Bearing (°)", "spline", "constant")
    def bearing(self):
        """Rocket bearing compass, in degrees, as a Function of time."""
        x, y = self.x[:, 1], self.y[:, 1]
        bearing = (2 * np.pi - np.arctan2(-x, y)) * (180 / np.pi)
        return np.column_stack((self.time, bearing))
    
    @funcify_method("Time (s)", "Latitude (°)", "linear", "constant")
    def latitude(self):
        """Rocket latitude coordinate, in degrees, as a Function of
        time.
        """
        lat1 = np.deg2rad(self.env.latitude)  # Launch lat point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        latitude = np.rad2deg(
            np.arcsin(
                np.sin(lat1) * np.cos(self.drift[:, 1] / self.env.earth_radius)
                + np.cos(lat1)
                * np.sin(self.drift[:, 1] / self.env.earth_radius)
                * np.cos(np.deg2rad(self.bearing[:, 1]))
            )
        )
        return np.column_stack((self.time, latitude))

    # TODO: haversine should be defined in tools.py so we just invoke it in here.
    @funcify_method("Time (s)", "Longitude (°)", "linear", "constant")
    def longitude(self):
        """Rocket longitude coordinate, in degrees, as a Function of
        time.
        """
        lat1 = np.deg2rad(self.env.latitude)  # Launch lat point converted to radians
        lon1 = np.deg2rad(self.env.longitude)  # Launch lon point converted to radians

        # Applies the haversine equation to find final lat/lon coordinates
        longitude = np.rad2deg(
            lon1
            + np.arctan2(
                np.sin(np.deg2rad(self.bearing[:, 1]))
                * np.sin(self.drift[:, 1] / self.env.earth_radius)
                * np.cos(lat1),
                np.cos(self.drift[:, 1] / self.env.earth_radius)
                - np.sin(lat1) * np.sin(np.deg2rad(self.latitude[:, 1])),
            )
        )

        return np.column_stack((self.time, longitude))
    
    @funcify_method("Time (s)", "Deployment (-)", "linear", "zero")
    def airbrake_deployment(self):
        """Deployment level of the airbrake"""
        return np.array(self.__post_processed_variables)[:, [0, 1]]

# TODO: Add sensors to this class