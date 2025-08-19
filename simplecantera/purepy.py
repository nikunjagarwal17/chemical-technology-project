"""
Pure-Python core implementation for the SimpleCantera MVP.

Includes:
- Thermodynamics: ideal-gas, constant cp enthalpy/entropy
- Reaction: reversible A <=> B, mass-action kinetics
- Reactor: Well-mixed (constant-volume), CSTR, discretized PFR
- ReactorNetwork: combine reactors in series/parallel
- run_simulation_from_dict: high-level entrypoint using a spec dictionary
"""
from math import log
import csv
from typing import List, Tuple


class Thermodynamics:
    """Ideal gas thermodynamics with constant heat capacity.

    enthalpy(T) = cp * T
    entropy(T) = cp * ln(T) (simplified)
    """
    R = 8.31446261815324  # J/mol/K

    def __init__(self, cp: float = 29.1):
        self.cp = float(cp)

    def enthalpy(self, T: float) -> float:
        return self.cp * float(T)

    def entropy(self, T: float) -> float:
        return self.cp * log(float(T)) if T > 0 else float('nan')


class Reaction:
    """Simple reversible reaction A <=> B with mass-action kinetics.

    rate = kf*[A] - kr*[B]
    """
    def __init__(self, kf: float, kr: float):
        self.kf = float(kf)
        self.kr = float(kr)

    def rate(self, conc: List[float]) -> float:
        # conc: [A, B]
        return self.kf * conc[0] - self.kr * conc[1]


class ReactionMulti:
    """General reversible reaction with stoichiometry.

    reactants and products are dicts mapping species index to stoichiometric coeff.
    rate = kf * product(conc[reactant]**nu) - kr * product(conc[product]**nu)
    """
    def __init__(self, kf: float, kr: float, reactants: dict, products: dict):
        self.kf = float(kf)
        self.kr = float(kr)
        # reactants/products: dict {species_idx: stoich}
        self.reactants = {int(k): int(v) for k, v in reactants.items()}
        self.products = {int(k): int(v) for k, v in products.items()}

    def rate(self, conc: List[float]) -> float:
        f = 1.0
        for idx, nu in self.reactants.items():
            f *= conc[idx] ** nu if conc[idx] > 0 else 0.0
        r = 1.0
        for idx, nu in self.products.items():
            r *= conc[idx] ** nu if conc[idx] > 0 else 0.0
        return self.kf * f - self.kr * r


class MultiReactor:
    """Reactor for N species and multiple reactions using RK4 integration.

    State: concentrations list of length N.
    """
    def __init__(self, thermo: Thermodynamics, reactions: List[ReactionMulti],
                 species: List[str], T: float = 300.0, conc0: List[float] = None):
        self.thermo = thermo
        self.reactions = reactions
        self.species = list(species)
        self.T = float(T)
        self.N = len(self.species)
        if conc0 is None:
            self.conc = [0.0] * self.N
        else:
            self.conc = [float(x) for x in conc0]

    def _dcdt(self, conc: List[float]) -> List[float]:
        # initialize derivative
        d = [0.0] * self.N
        for rxn in self.reactions:
            rate = rxn.rate(conc)
            # reactants: decrease
            for idx, nu in rxn.reactants.items():
                d[idx] -= nu * rate
            # products: increase
            for idx, nu in rxn.products.items():
                d[idx] += nu * rate
        return d

    def step(self, dt: float):
        y0 = self.conc
        k1 = self._dcdt(y0)
        y1 = [y0[i] + 0.5 * dt * k1[i] for i in range(self.N)]
        k2 = self._dcdt(y1)
        y2 = [y0[i] + 0.5 * dt * k2[i] for i in range(self.N)]
        k3 = self._dcdt(y2)
        y3 = [y0[i] + dt * k3[i] for i in range(self.N)]
        k4 = self._dcdt(y3)
        for i in range(self.N):
            self.conc[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            if self.conc[i] < 0:
                self.conc[i] = 0.0

    def run(self, time_span: float, time_step: float):
        nsteps = int(max(1, round(time_span / time_step)))
        times = [0.0]
        traj = [list(self.conc)]
        for i in range(nsteps):
            self.step(time_step)
            times.append((i + 1) * time_step)
            traj.append(list(self.conc))
        return times, traj

    def run_adaptive(self, time_span: float, dt_init: float = 1e-3, atol: float = 1e-6, rtol: float = 1e-6):
        # Step-doubling adaptive RK4 (take dt and two dt/2 steps)
        def rk4_step(y, dt):
            N = len(y)
            def deriv(state):
                d = [0.0] * N
                for rxn in self.reactions:
                    rate = rxn.rate(state)
                    for idx, nu in rxn.reactants.items():
                        d[int(idx)] -= nu * rate
                    for idx, nu in rxn.products.items():
                        d[int(idx)] += nu * rate
                return d
            k1 = deriv(y)
            y2 = [y[i] + 0.5 * dt * k1[i] for i in range(N)]
            k2 = deriv(y2)
            y3 = [y[i] + 0.5 * dt * k2[i] for i in range(N)]
            k3 = deriv(y3)
            y4 = [y[i] + dt * k3[i] for i in range(N)]
            k4 = deriv(y4)
            ynew = [y[i] + (dt/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) for i in range(N)]
            return ynew

        t = 0.0
        y = list(self.conc)
        out_times = [0.0]
        out_traj = [list(y)]
        dt = float(dt_init)
        while t < time_span:
            if t + dt > time_span:
                dt = time_span - t
            # one step
            y_big = rk4_step(y, dt)
            # two half steps
            y_half = rk4_step(y, dt*0.5)
            y_two = rk4_step(y_half, dt*0.5)
            # error estimate
            errnorm = 0.0
            for i in range(self.N):
                sc = atol + rtol * max(abs(y[i]), abs(y_two[i]))
                e = (y_two[i] - y_big[i]) / sc if sc != 0 else 0.0
                errnorm += e*e
            errnorm = (errnorm / self.N) ** 0.5
            if errnorm <= 1.0:
                t += dt
                y = [max(0.0, v) for v in y_two]
                out_times.append(t)
                out_traj.append(list(y))
            # adjust dt
            safety = 0.9
            minscale = 0.2
            maxscale = 2.0
            if errnorm == 0.0:
                errnorm = 1e-16
            # RK4 step-doubling: error ~ dt^5 so exponent is 1/5
            scale = safety * errnorm ** (-0.2)
            scale = max(minscale, min(maxscale, scale))
            dt *= scale
            if dt < 1e-12:
                dt = 1e-12
        self.conc = list(y)
        return out_times, out_traj


class WellMixedReactor:
    """Constant-volume, isothermal reactor using forward Euler integration.

    State: concentrations [A, B]
    
    Constructor is flexible:
      - WellMixedReactor(thermo, reaction, T=..., conc0=(A0,B0))  # full form
      - WellMixedReactor(reaction, A0=..., B0=...)               # convenient short form
    """
    def __init__(self, *args, T: float = 300.0, volume: float = 1.0, conc0: Tuple[float, float] = (1.0, 0.0), **kwargs):
        # Backwards-compatible and convenient constructors
        if len(args) == 1 and isinstance(args[0], Reaction):
            # Short form: (reaction, A0=..., B0=...)
            self.thermo = Thermodynamics()
            self.reaction = args[0]
        elif len(args) >= 2:
            # Full form: (thermo, reaction, ...)
            self.thermo = args[0]
            self.reaction = args[1]
        else:
            raise TypeError('Expected WellMixedReactor(reaction, ...) or WellMixedReactor(thermo, reaction, ...)')

        # Accept A0/B0 legacy keyword args for convenience
        if 'A0' in kwargs or 'B0' in kwargs:
            A0 = kwargs.get('A0', conc0[0])
            B0 = kwargs.get('B0', conc0[1])
            conc0 = (float(A0), float(B0))

        self.T = float(T)
        self.volume = float(volume)
        self.conc = [float(conc0[0]), float(conc0[1])]

    def step(self, dt: float):
        # RK4 for dy/dt where y = [A,B], dy/dt = [-r, r]
        A = self.conc[0]
        B = self.conc[1]
        def f(a, b):
            r = self.reaction.rate([a, b])
            return -r, r

        k1A, k1B = f(A, B)
        k2A, k2B = f(A + 0.5 * dt * k1A, B + 0.5 * dt * k1B)
        k3A, k3B = f(A + 0.5 * dt * k2A, B + 0.5 * dt * k2B)
        k4A, k4B = f(A + dt * k3A, B + dt * k3B)

        A += (dt / 6.0) * (k1A + 2.0 * k2A + 2.0 * k3A + k4A)
        B += (dt / 6.0) * (k1B + 2.0 * k2B + 2.0 * k3B + k4B)
        self.conc[0] = max(A, 0.0)
        self.conc[1] = max(B, 0.0)

    def run(self, time_span: float, time_step: float):
        nsteps = int(max(1, round(time_span / time_step)))
        times = [0.0]
        traj = [[self.conc[0], self.conc[1]]]
        for i in range(nsteps):
            self.step(time_step)
            times.append((i + 1) * time_step)
            traj.append([self.conc[0], self.conc[1]])
        return times, traj

    def run_adaptive(self, time_span: float, dt_init: float = 1e-3, atol: float = 1e-6, rtol: float = 1e-6):
        # Step-doubling adaptive RK4 for scalar system
        def rk4_step_scalar(state, dt):
            a, b = state
            def f(a, b):
                r = self.reaction.rate([a, b])
                return -r, r
            k1A, k1B = f(a, b)
            k2A, k2B = f(a + 0.5*dt*k1A, b + 0.5*dt*k1B)
            k3A, k3B = f(a + 0.5*dt*k2A, b + 0.5*dt*k2B)
            k4A, k4B = f(a + dt*k3A, b + dt*k3B)
            anew = a + (dt/6.0)*(k1A + 2.0*k2A + 2.0*k3A + k4A)
            bnew = b + (dt/6.0)*(k1B + 2.0*k2B + 2.0*k3B + k4B)
            return anew, bnew

        t = 0.0
        A = self.conc[0]
        B = self.conc[1]
        out_times = [0.0]
        out_traj = [[A, B]]
        dt = float(dt_init)
        while t < time_span:
            if t + dt > time_span:
                dt = time_span - t
            # one big step
            A_big, B_big = rk4_step_scalar((A, B), dt)
            # two half steps
            A_half, B_half = rk4_step_scalar((A, B), dt*0.5)
            A_two, B_two = rk4_step_scalar((A_half, B_half), dt*0.5)
            # error
            scA = atol + rtol * max(abs(A), abs(A_two))
            scB = atol + rtol * max(abs(B), abs(B_two))
            eA = (A_two - A_big) / scA if scA != 0 else 0.0
            eB = (B_two - B_big) / scB if scB != 0 else 0.0
            errnorm = (eA*eA + eB*eB) ** 0.5 / (2 ** 0.5)
            if errnorm <= 1.0:
                t += dt
                A, B = max(0.0, A_two), max(0.0, B_two)
                out_times.append(t); out_traj.append([A, B])
            safety = 0.9; minscale = 0.2; maxscale = 2.0
            if errnorm == 0.0: errnorm = 1e-16
            # RK4 step-doubling: error ~ dt^5 so exponent is 1/5
            scale = safety * (errnorm ** -0.2)
            scale = max(minscale, min(maxscale, scale))
            dt *= scale
            if dt < 1e-12: dt = 1e-12
        self.conc = [A, B]
        return out_times, out_traj


class CSTR(WellMixedReactor):
    """Continuous stirred tank reactor with inlet/outlet flow.

    dC/dt = -r(C) + (q/V)*(C_in - C)
    """
    def __init__(self, thermo: Thermodynamics, reaction: Reaction,
                 T: float = 300.0, volume: float = 1.0, conc0: Tuple[float, float] = (1.0, 0.0),
                 q: float = 0.0, conc_in: Tuple[float, float] = (0.0, 0.0)):
        super().__init__(thermo, reaction, T=T, volume=volume, conc0=conc0)
        self.q = float(q)
        self.conc_in = [float(conc_in[0]), float(conc_in[1])]

    def step(self, dt: float):
        # RK4 for combined ODE: dC/dt = reaction + flow
        A = self.conc[0]
        B = self.conc[1]

        def f(a, b):
            r = self.reaction.rate([a, b])
            ra = -r
            rb = r
            fa = ra + (self.q / self.volume) * (self.conc_in[0] - a)
            fb = rb + (self.q / self.volume) * (self.conc_in[1] - b)
            return fa, fb

        k1A, k1B = f(A, B)
        k2A, k2B = f(A + 0.5 * dt * k1A, B + 0.5 * dt * k1B)
        k3A, k3B = f(A + 0.5 * dt * k2A, B + 0.5 * dt * k2B)
        k4A, k4B = f(A + dt * k3A, B + dt * k3B)

        A += (dt / 6.0) * (k1A + 2.0 * k2A + 2.0 * k3A + k4A)
        B += (dt / 6.0) * (k1B + 2.0 * k2B + 2.0 * k3B + k4B)
        self.conc[0] = max(A, 0.0)
        self.conc[1] = max(B, 0.0)


class PFR:
    """Simple discretized plug-flow reactor implemented as N segments.

    Each segment is treated as a small CSTR; flow q moves material from segment i to i+1.
    """
    def __init__(self, thermo: Thermodynamics, reaction: Reaction,
                 T: float = 300.0, total_volume: float = 1.0, nseg: int = 10,
                 conc0: Tuple[float, float] = (1.0, 0.0), q: float = 1.0):
        self.thermo = thermo
        self.reaction = reaction
        self.T = float(T)
        self.total_volume = float(total_volume)
        self.nseg = max(1, int(nseg))
        self.seg_volume = self.total_volume / self.nseg
        # segments concentrations: list of [A,B]
        self.segs = [[float(conc0[0]), float(conc0[1])] for _ in range(self.nseg)]
        self.q = float(q)

    def step(self, dt: float):
        # compute reaction change in each segment using RK4 per segment for reaction term
        new_segs = [list(s) for s in self.segs]
        for i in range(self.nseg):
            A = self.segs[i][0]
            B = self.segs[i][1]
            def f(a, b):
                r = self.reaction.rate([a, b])
                return -r, r
            k1A, k1B = f(A, B)
            k2A, k2B = f(A + 0.5 * dt * k1A, B + 0.5 * dt * k1B)
            k3A, k3B = f(A + 0.5 * dt * k2A, B + 0.5 * dt * k2B)
            k4A, k4B = f(A + dt * k3A, B + dt * k3B)
            newA = A + (dt / 6.0) * (k1A + 2.0 * k2A + 2.0 * k3A + k4A)
            newB = B + (dt / 6.0) * (k1B + 2.0 * k2B + 2.0 * k3B + k4B)
            new_segs[i][0] = newA
            new_segs[i][1] = newB
        # flow between segments: simple upwind
        for i in range(self.nseg - 1, 0, -1):
            # amount transferred from i-1 to i during dt
            Cin = new_segs[i - 1]
            Cout = new_segs[i]
            flow = (self.q / self.seg_volume) * dt
            # explicit Euler exchange: C_i += flow*(C_{i-1} - C_i)
            for j in range(2):
                delta = flow * (Cin[j] - Cout[j])
                new_segs[i][j] += delta
                new_segs[i - 1][j] -= delta
        # clamp and assign
        for i in range(self.nseg):
            for j in range(2):
                if new_segs[i][j] < 0:
                    new_segs[i][j] = 0.0
        self.segs = new_segs

    def run(self, time_span: float, time_step: float):
        nsteps = int(max(1, round(time_span / time_step)))
        times = [0.0]
        traj = [[list(self.segs[0]), list(self.segs[-1])]]
        for i in range(nsteps):
            self.step(time_step)
            times.append((i + 1) * time_step)
            traj.append([[s[0] for s in self.segs][0:1][0], [self.segs[-1][0], self.segs[-1][1]]])
        # For compatibility return times and concentration per time as [A,B] of outlet
        outlet_traj = [[s[-2] if False else seg[0], seg[1]] for seg in self.segs]
        # Simpler: return times and outlet concentration history
        outlet = [[seg[0], seg[1]] for seg in self.segs]
        # Build simple outlet history from recorded traj
        out_history = []
        for row in traj:
            # row = [first_segment_list, last_segment_list]
            first, last = row
            out_history.append([last[0], last[1]])
        return times, out_history


class ReactorNetwork:
    """Support simple series and parallel networks of reactors.

    network_spec: {'type': 'series'|'parallel', 'reactors': [spec,...], 'flow': q}
    """
    def __init__(self, reactors: List[object], mode: str = 'series'):
        self.reactors = reactors
        self.mode = mode

    def run(self, time_span: float, time_step: float):
        nsteps = int(max(1, round(time_span / time_step)))
        times = [0.0]
        history = []
        # initialize history with initial concentrations of each reactor
        history.append([list(r.conc) if hasattr(r, 'conc') else list(r.segs[0]) for r in self.reactors])
        for i in range(nsteps):
            # step each reactor
            for r in self.reactors:
                if hasattr(r, 'step'):
                    r.step(time_step)
            times.append((i + 1) * time_step)
            history.append([list(r.conc) if hasattr(r, 'conc') else list(r.segs[0]) for r in self.reactors])
        return times, history


# Convenience builder and runner

def build_from_dict(spec: dict):
    """Create thermo/reaction/reactor from a spec dict.

    spec examples:
      { 'reaction': {'kf':1., 'kr':0.5}, 'initial':{'temperature':300,'conc':{'A':1,'B':0}},
        'sim':{'time_span':10,'time_step':0.01}, 'system': 'WellMixed' }

      for CSTR: 'system': 'CSTR', 'cstr': { 'q': 0.5, 'conc_in': {'A':1,'B':0} }
      for PFR: 'system': 'PFR', 'pfr': { 'nseg': 20, 'q': 1.0 }
      for network series: 'system': 'series', 'reactors': [ list of reactor specs ]
    """
    reaction = spec.get('reaction', {})
    initial = spec.get('initial', {})
    sim = spec.get('sim', {})

    kf = reaction.get('kf', 1.0)
    kr = reaction.get('kr', 0.5)
    T = initial.get('temperature', 300.0)
    conc = initial.get('conc', {})
    A0 = conc.get('A', 1.0)
    B0 = conc.get('B', 0.0)
    cp = spec.get('thermo', {}).get('cp', 29.1)

    thermo = Thermodynamics(cp=cp)
    rxn = Reaction(kf=kf, kr=kr)

    system = spec.get('system', 'WellMixed')
    # Multi-species support
    if 'species' in spec and 'reactions' in spec:
        species = spec.get('species', [])
        # reactions: list of dicts with kf, kr, reactants, products
        rxns = []
        for r in spec.get('reactions', []):
            kf = r.get('kf', 1.0)
            kr = r.get('kr', 0.0)
            reactants = r.get('reactants', {})
            products = r.get('products', {})
            # map species names to indices
            react_idx = {species.index(s): v for s, v in reactants.items()}
            prod_idx = {species.index(s): v for s, v in products.items()}
            rxns.append(ReactionMulti(kf=kf, kr=kr, reactants=react_idx, products=prod_idx))
        conc0_list = [spec.get('initial', {}).get('conc', {}).get(s, 0.0) for s in species]
        reactor = MultiReactor(thermo, rxns, species, T=T, conc0=conc0_list)
        return reactor, sim
    if system == 'WellMixed':
        reactor = WellMixedReactor(thermo, rxn, T=T, conc0=(A0, B0))
        return reactor, sim
    elif system == 'CSTR':
        cstr_spec = spec.get('cstr', {})
        q = cstr_spec.get('q', 0.0)
        conc_in = cstr_spec.get('conc_in', {'A': 0.0, 'B': 0.0})
        reactor = CSTR(thermo, rxn, T=T, conc0=(A0, B0), q=q, conc_in=(conc_in.get('A', 0.0), conc_in.get('B', 0.0)))
        return reactor, sim
    elif system == 'PFR':
        pfr_spec = spec.get('pfr', {})
        nseg = pfr_spec.get('nseg', 10)
        q = pfr_spec.get('q', 1.0)
        reactor = PFR(thermo, rxn, T=T, total_volume=pfr_spec.get('total_volume', 1.0), nseg=nseg, conc0=(A0, B0), q=q)
        return reactor, sim
    elif system == 'series':
        # build each reactor and return a ReactorNetwork
        reactors = []
        for r_spec in spec.get('reactors', []):
            r, _ = build_from_dict(r_spec)
            reactors.append(r)
        net = ReactorNetwork(reactors, mode='series')
        return net, sim
    else:
        # fallback to WellMixed
        reactor = WellMixedReactor(thermo, rxn, T=T, conc0=(A0, B0))
        return reactor, sim


def run_simulation_from_dict(spec: dict, csv_out: str = None, plot: bool = False):
    """High-level runner. Returns times and trajectory.

    For network returns times and a history per reactor.
    For single reactor returns times and concentrations [A,B] per time.
    """
    reactor, sim = build_from_dict(spec)
    time_span = sim.get('time_span', 10.0)
    time_step = sim.get('time_step', 0.01)

    if hasattr(reactor, 'run'):
        times, traj = reactor.run(time_span, time_step)
    else:
        # for safety assume well-mixed interface
        times, traj = reactor.run(time_span, time_step)

    # Normalize output for CSV: if network, flatten by reactor index
    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.writer(f)
            # header
            if isinstance(traj[0][0], list) or isinstance(traj[0][0], tuple):
                # network-like history: traj[t][reactor_index] == [A,B]
                    # ...existing code...
                nreact = len(traj[0])
                header = ['time']
                for i in range(nreact):
                    header += [f'A_r{i}', f'B_r{i}']
                writer.writerow(header)
                for t, row in zip(times, traj):
                    rline = [t]
                    for state in row:
                        rline += [state[0], state[1]]
                    writer.writerow(rline)
            else:
                writer.writerow(['time', 'A', 'B'])
                for t, (a, b) in zip(times, traj):
                    writer.writerow([t, a, b])

    if plot:
        try:
            import matplotlib.pyplot as plt
            if isinstance(traj[0][0], list) or isinstance(traj[0][0], tuple):
                # network: plot first reactor's A and B
                A = [row[0][0] for row in traj]
                B = [row[0][1] for row in traj]
            else:
                A = [p[0] for p in traj]
                B = [p[1] for p in traj]
            plt.plot(times, A, label='A')
            plt.plot(times, B, label='B')
            plt.xlabel('time')
            plt.ylabel('concentration')
            plt.legend()
            plt.show()
        except Exception:
            pass

    return times, traj


# Small benchmarking helper
def benchmark_multi_reactor(reactor: MultiReactor, time_span: float = 1.0, time_step: float = 0.001) -> float:
    """Run a short timed loop of the reactor and return elapsed seconds.

    This is a tiny helper for local microbenchmarks. It uses time.perf_counter.
    """
    import time
    t0 = time.perf_counter()
    reactor.run(time_span, time_step)
    t1 = time.perf_counter()
    return t1 - t0


# Backwards-compatible alias
run_simulation = run_simulation_from_dict
