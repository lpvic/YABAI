import itertools
from math import ceil, floor, log
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt

from .constants import ZH_L16, noaa_cns_equations, pw
from .tanks import Tank
from .exceptions import InterpolationError


@dataclass
class Parameters:
    last_stop_depth: float = 3
    stop_depth_incr: float = 3
    safety_stop_depth: float = 5
    safety_stop_duration: float = 3

    v_asc: float = 10
    v_desc: float = 20

    own_descent_sac: float = 20
    own_bottom_sac: float = 20
    own_ascent_sac: float = 17
    buddy_ascent_sac: float = 17

    gf_high: float = 1.
    gf_low: float = 1.

    calc_ascent: bool = True
    calc_descent: bool = True
    deco_stops: bool = True
    safety_stop: bool = True

    gas_switch: str = 'depth'  # 'depth' | 'stop' | 'manual'
    gas_switch_duration: float = 60

    dt: timedelta = timedelta(seconds=1)


class Waypoint:
    def __init__(self, depth: float = 0., duration: float | timedelta = None, runtime: float | timedelta = None,
                 tank: int = 0) -> None:
        self.depth: float = depth / 1.
        self.duration: timedelta
        self.runtime: timedelta
        self.tank = tank

        if duration is None:
            self.duration = timedelta(seconds=0)
        elif isinstance(duration, timedelta):
            self.duration = duration
        else:
            self.duration = timedelta(minutes=duration)

        if runtime is None:
            self.runtime = timedelta(seconds=0)
        elif isinstance(runtime, timedelta):
            self.runtime = runtime
        else:
            self.runtime = timedelta(minutes=runtime)

    def __str__(self) -> str:
        return ('Waypoint(depth={depth:.1f}, duration={duration}, runtime={runtime}, tank={tank})'
                .format(depth=self.depth, duration=self.duration, runtime=self.runtime, tank=self.tank))

    def __repr__(self) -> str:
        return ('Waypoint(depth={depth:.1f}, duration={duration}, runtime={runtime}, tank={tank})'
                .format(depth=self.depth, duration=self.duration, runtime=self.runtime, tank=self.tank))


class IntegrationPoint:
    def __init__(self, waypoint: Waypoint) -> None:
        self.waypoint = waypoint
        self.tank_pressure = []
        self.load_ig = {'N2': np.full(16, 0.79 * (1 - pw)), 'He': np.zeros(16)}
        self.ceilings = np.ones(16)
        self.otu = 0.
        self.otu_cum = 0.
        self.cns = 0.
        self.cns_cum = 0.

    @property
    def p_amb(self) -> float:
        return (self.waypoint.depth / 10.) + 1.

    @property
    def ceiling(self) -> float:
        if np.max(self.ceilings) > 0.:
            return np.max(self.ceilings)
        else:
            return 0.

    def __str__(self) -> str:
        return ('IntegrationPoint(depth={depth:.5f}, duration={duration}, runtime={runtime}, tank={tank},'
                ' ceiling={ceiling:.1f}, tank_pressure={tank_pressure}, cns={cns}%, otu={otu})'
                .format(depth=self.waypoint.depth, duration=self.waypoint.duration, runtime=self.waypoint.runtime,
                        tank=self.waypoint.tank, ceiling=self.ceiling,
                        tank_pressure=[int('{:.0f}'.format(p)) for p in self.tank_pressure],
                        cns=ceil(self.cns_cum * 100), otu=ceil(self.otu_cum)))


class Profile:
    def __init__(self, waypoints: list[Waypoint], tanks: list[Tank], params: Parameters = Parameters()) -> None:
        self._params: Parameters = params
        self._tanks: list[Tank] = tanks
        self._waypoints: list[Waypoint] = []
        self._integration_points: list[IntegrationPoint] = []

        self._complete_waypoints(waypoints, self._params.calc_descent)
        self._calculate_bottom()
        deco_dive = self._calculate_direct_ascent(0, self._integration_points[-1], False)[-1].ceiling > 0.
        if deco_dive:
            self._calculate_deco_ascent(0., self._integration_points[-1])
        else:
            if self._params.safety_stop:
                self._calculate_regular_ascent(0., self._integration_points[-1])
            else:
                self._calculate_direct_ascent(0., self._integration_points[-1])

    @property
    def waypoints(self) -> list[Waypoint]:
        return self._waypoints

    @property
    def integration_points(self) -> list[IntegrationPoint]:
        return self._integration_points

    def plot(self) -> None:
        depth = []
        ceiling = []
        runtime = []
        for ip in self._integration_points:
            depth.append(ip.waypoint.depth)
            ceiling.append(ip.ceiling)
            runtime.append(ip.waypoint.runtime.seconds / 60)
        plt.gca().invert_yaxis()
        plt.plot(runtime, depth, 'b-', markersize=2)
        plt.plot(runtime, ceiling, 'r-', markersize=2)
        plt.show()

    def plot_waypoints(self) -> None:
        depth = []
        runtime = []
        for wp in self._waypoints:
            depth.append(wp.depth)
            runtime.append(wp.runtime.seconds)
        plt.gca().invert_yaxis()
        plt.plot(runtime, depth, 'bo-')
        plt.show()

    def plot_integration_points(self) -> None:
        depth = []
        runtime = []
        for ip in self._integration_points:
            depth.append(ip.waypoint.depth)
            runtime.append(ip.waypoint.runtime.seconds)
        plt.gca().invert_yaxis()
        plt.plot(runtime, depth, 'b-', markersize=2)
        plt.show()

    def plot_compartment(self, gas: str, compartment: int) -> None:
        p_ig = []
        runtime = []
        for ip in self._integration_points:
            p_ig.append(ip.load_ig[gas][compartment - 1])
            runtime.append(ip.waypoint.runtime.seconds)
        plt.gca().invert_yaxis()
        plt.plot(runtime, p_ig, 'b-', markersize=2)
        plt.show()

    def plot_compartments(self, gas: str) -> None:
        colors = 'bgrcmkbgrcmkbgrc'
        for c in range(16):
            p_ig = []
            runtime = []
            for ip in self._integration_points:
                p_ig.append(ip.load_ig[gas][c])
                runtime.append(ip.waypoint.runtime.seconds)
            plt.plot(runtime, p_ig, colors[c] + '-', markersize=2)
        plt.gca().invert_yaxis()
        plt.show()

    def plot_ceiling(self) -> None:
        ceiling = []
        runtime = []
        for ip in self._integration_points:
            ceiling.append(ip.ceiling)
            runtime.append(ip.waypoint.runtime.seconds)
        plt.gca().invert_yaxis()
        plt.plot(runtime, ceiling, 'bo-', markersize=2)
        plt.show()

    def plot_ceilings(self) -> None:
        colors = 'bgrcmkbgrcmkbgrc'
        for c in range(16):
            ceilings = []
            runtime = []
            for ip in self._integration_points:
                ceilings.append(ip.ceilings[c])
                runtime.append(ip.waypoint.runtime.seconds)
            plt.plot(runtime, ceilings, colors[c] + '-', markersize=2)
        plt.gca().invert_yaxis()
        plt.show()

    def _complete_waypoints(self, waypoints: list[Waypoint], desc: bool = True) -> None:
        wps = [wp for wp in waypoints]
        if wps[0].depth != 0:
            time_to_bottom = timedelta(minutes=wps[0].depth / self._params.v_desc)
            if desc:
                self._waypoints.append(Waypoint(0, time_to_bottom, timedelta(seconds=0)))
                self._waypoints.append(Waypoint(wps[0].depth, wps[0].duration, self._waypoints[0].duration))
            else:
                self._waypoints.append(Waypoint(wps[0].depth, wps[0].duration, timedelta(seconds=0)))
        else:
            self._waypoints.append(Waypoint(wps[0].depth, wps[0].duration, timedelta(seconds=0)))

        if len(wps) == 1:
            wps.append(Waypoint(wps[0].depth, timedelta(seconds=0),
                                wps[0].runtime.seconds / 60 + wps[0].duration.seconds / 60))

        for idx, wp in enumerate(wps[1:], start=1):
            prev_wp = self._waypoints[-1]

            if wp.depth > prev_wp.depth:
                desc_time = timedelta(minutes=(wp.depth - prev_wp.depth) / self._params.v_desc)
                self._waypoints.append(Waypoint(prev_wp.depth, desc_time,
                                                prev_wp.runtime.seconds / 60 + prev_wp.duration.seconds / 60))
            elif wp.depth < prev_wp.depth:
                asc_time = timedelta(minutes=(prev_wp.depth - wp.depth) / self._params.v_asc)
                self._waypoints.append(Waypoint(prev_wp.depth, asc_time,
                                                prev_wp.runtime.seconds / 60 + prev_wp.duration.seconds / 60))

            prev_wp = self._waypoints[-1]
            if idx == (len(waypoints) - 1):
                duration = timedelta(seconds=0)
            else:
                duration = wp.duration
            self._waypoints.append(Waypoint(wp.depth, duration,
                                            prev_wp.runtime.seconds / 60 + prev_wp.duration.seconds / 60))

    def _calculate_bottom(self) -> None:
        t = 0
        for idx, wp in enumerate(self._waypoints):
            if t > (wp.runtime.seconds + wp.duration.seconds):
                continue
            else:
                while t <= (wp.runtime.seconds + wp.duration.seconds):
                    new_wp = Waypoint(self._interpolate_depth(timedelta(seconds=t)), self._params.dt,
                                      timedelta(seconds=t))
                    new_ip = IntegrationPoint(new_wp)
                    new_ip.tank_pressure = [x.start_pressure for x in self._tanks]

                    if self._integration_points:
                        prev_ip = self._integration_points[-1]
                        new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
                        if prev_ip.waypoint.depth < new_ip.waypoint.depth:
                            sac = self._params.own_descent_sac
                        else:
                            sac = self._params.own_bottom_sac
                        new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, sac)
                        new_ip.cns = self._calculate_cns(new_ip, prev_ip)
                        new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
                        new_ip.otu = self._calculate_otu(new_ip, prev_ip)
                        new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu

                    new_ip.ceilings = self._calculate_ceilings(new_ip)
                    self._integration_points.append(new_ip)
                    t = t + self._params.dt.seconds

    def _calculate_direct_ascent(self, depth: float, ip: IntegrationPoint, append: bool = True)\
            -> list[IntegrationPoint]:
        prev_ip = ip
        t = ip.waypoint.runtime.seconds
        out = []
        while prev_ip.waypoint.depth > depth:
            t = t + self._params.dt.seconds
            new_wp = Waypoint(depth=round(prev_ip.waypoint.depth - (self._params.v_asc * self._params.dt.seconds / 60),
                                          1),
                              duration=self._params.dt, runtime=timedelta(seconds=t), tank=prev_ip.waypoint.tank)
            new_ip = IntegrationPoint(new_wp)
            new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
            new_ip.ceilings = self._calculate_ceilings(new_ip)
            new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
            new_ip.cns = self._calculate_cns(new_ip, prev_ip)
            new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
            new_ip.otu = self._calculate_otu(new_ip, prev_ip)
            new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
            prev_ip = new_ip
            out.append(new_ip)

        if append:
            duration = out[-1].waypoint.runtime.seconds - out[0].waypoint.runtime.seconds + self._params.dt.seconds
            self._waypoints[-1].duration = timedelta(seconds=duration)
            self._waypoints.append(Waypoint(out[-1].waypoint.depth, 0, out[-1].waypoint.runtime, out[-1].waypoint.tank))
            self._integration_points = self._integration_points + out

        return out

    def _calculate_regular_ascent(self, depth: float, ip: IntegrationPoint, append: bool = True)\
            -> list[IntegrationPoint]:
        prev_ip = ip
        if depth > self._params.safety_stop_depth:
            segments = [self._calculate_direct_ascent(depth, prev_ip, False)]
        else:
            segments = [self._calculate_direct_ascent(self._params.safety_stop_depth, prev_ip, False)]

            prev_ip = segments[-1][-1]
            t = prev_ip.waypoint.runtime.seconds
            safety_stop_timer = 0.
            out = []
            while safety_stop_timer < timedelta(minutes=self._params.safety_stop_duration).seconds:
                t = t + self._params.dt.seconds
                new_wp = Waypoint(self._params.safety_stop_depth, self._params.dt, timedelta(seconds=t),
                                  prev_ip.waypoint.tank)
                new_ip = IntegrationPoint(new_wp)
                new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
                new_ip.ceilings = self._calculate_ceilings(new_ip)
                new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
                new_ip.cns = self._calculate_cns(new_ip, prev_ip)
                new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
                new_ip.otu = self._calculate_otu(new_ip, prev_ip)
                new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
                prev_ip = new_ip
                safety_stop_timer = safety_stop_timer + self._params.dt.seconds
                out.append(new_ip)
            segments.append(out)

            segments.append(self._calculate_direct_ascent(depth, segments[-1][-1], False))

        if append:
            for s in segments:
                duration = s[-1].waypoint.runtime.seconds - s[0].waypoint.runtime.seconds + self._params.dt.seconds
                self._waypoints[-1].duration = timedelta(seconds=duration)
                self._waypoints.append(Waypoint(s[-1].waypoint.depth, 0, s[-1].waypoint.runtime, s[-1].waypoint.tank))
                self._integration_points = self._integration_points + s

        return list(itertools.chain(*segments))

    def _calculate_deco_ascent(self, depth: float, ip: IntegrationPoint, append: bool = True) -> list[IntegrationPoint]:
        prev_ip = ip
        next_deco_stop = self._calculate_next_deco_stop(prev_ip.ceiling)
        next_gas_stop = self._calculate_next_gas_stop(prev_ip.waypoint.depth)
        next_stop = max([next_deco_stop, next_gas_stop])

        t = ip.waypoint.runtime.seconds

        segments = []
        while prev_ip.waypoint.depth > depth:
            out = []
            while prev_ip.waypoint.depth > next_stop:
                t = t + self._params.dt.seconds
                new_depth = round(prev_ip.waypoint.depth - (self._params.v_asc * self._params.dt.seconds / 60), 1)
                new_wp = Waypoint(depth=new_depth, duration=self._params.dt, runtime=timedelta(seconds=t),
                                  tank=prev_ip.waypoint.tank)
                new_ip = IntegrationPoint(new_wp)
                new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
                new_ip.ceilings = self._calculate_ceilings(new_ip)
                new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
                new_ip.cns = self._calculate_cns(new_ip, prev_ip)
                new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
                new_ip.otu = self._calculate_otu(new_ip, prev_ip)
                new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
                next_deco_stop = self._calculate_next_deco_stop(new_ip.ceiling)
                next_stop = max([next_deco_stop, next_gas_stop])

                prev_ip = new_ip

                out.append(new_ip)

            if out:
                segments.append(out)

            if (next_deco_stop > 0.) and (next_deco_stop > next_gas_stop):
                segments.append(self._add_deco_stop(next_deco_stop, segments[-1][-1]))
                next_deco_stop = self._calculate_next_deco_stop(segments[-1][-1].ceiling)
                next_stop = max([next_deco_stop, next_gas_stop])

            if (next_gas_stop > 0.) and (next_gas_stop > next_deco_stop):
                segments.append(self._add_gas_switch_stop(next_gas_stop, segments[-1][-1]))
                next_gas_stop = self._calculate_next_gas_stop(segments[-1][-1].waypoint.depth)
                next_stop = max([next_deco_stop, next_gas_stop])

            prev_ip = segments[-1][-1]
            t = prev_ip.waypoint.runtime.seconds

        if append:
            for s in segments:
                duration = s[-1].waypoint.runtime.seconds - s[0].waypoint.runtime.seconds + self._params.dt.seconds
                self._waypoints[-1].duration = timedelta(seconds=duration)
                self._waypoints.append(Waypoint(s[-1].waypoint.depth, 0, s[-1].waypoint.runtime, s[-1].waypoint.tank))
                self._integration_points = self._integration_points + s

        return list(itertools.chain(*segments))

    def _calculate_compartments(self, ip: IntegrationPoint, prev_ip: IntegrationPoint) -> dict[str, float]:
        out = {}
        p_amb = ip.p_amb

        for g in ['N2', 'He']:
            p0 = prev_ip.load_ig[g]
            f_ig = self._tanks[prev_ip.waypoint.tank].gas.fN2 if g == 'N2' else\
                self._tanks[prev_ip.waypoint.tank].gas.fHe
            pi = np.full(16, f_ig * (p_amb - pw))
            r = (((ip.waypoint.depth - prev_ip.waypoint.depth) / prev_ip.waypoint.duration.seconds / 60) * f_ig) / 10
            k = log(2) / ZH_L16['C'][g]['ht']

            # Schreiner equation
            p_ig = pi
            p_ig = p_ig + r * (prev_ip.waypoint.duration.seconds / 60 - (1 / k))
            p_ig = p_ig - (pi - p0 - (r / k)) * np.exp(-k * prev_ip.waypoint.duration.seconds / 60)

            out[g] = p_ig

        return out

    def _calculate_ceilings(self, ip: IntegrationPoint) -> np.ndarray:
        a_n2 = ZH_L16['C']['N2']['a']
        b_n2 = ZH_L16['C']['N2']['b']
        a_he = ZH_L16['C']['He']['a']
        b_he = ZH_L16['C']['He']['b']
        load_n2 = ip.load_ig['N2']
        load_he = ip.load_ig['He']
        p_amb = ip.p_amb

        p_max = 0.
        for wp in self._waypoints:
            p_max = wp.depth if wp.depth > p_max else p_max
        p_max = (p_max / 10) + 1

        a = ((a_n2 * load_n2 + a_he * load_he) / (load_n2 + load_he))
        b = ((b_n2 * load_n2 + b_he * load_he) / (load_n2 + load_he))
        p_comp = load_n2 + load_he
        gf = self._params.gf_high - self._params.gf_low
        gf = gf / (1. - p_max)
        gf = gf * (p_amb - 1.)
        gf = gf + self._params.gf_high

        out = (p_comp - (a * gf))
        out = out / ((gf / b) - gf + 1.)

        return (out - 1.) * 10

    def _calculate_next_deco_stop(self, ceiling: float) -> float:
        out = ceil(ceiling / self._params.stop_depth_incr) * self._params.stop_depth_incr
        if (((out <= self._params.last_stop_depth)
            or ((out - self._params.last_stop_depth) < self._params.stop_depth_incr))
                and (ceiling > 0)):
            if ceiling < self._params.last_stop_depth:
                out = self._params.last_stop_depth
            else:
                out = ((ceil(ceiling / self._params.stop_depth_incr) * self._params.stop_depth_incr) +
                       self._params.stop_depth_incr)

        return out

    def _add_deco_stop(self, depth: float, prev_ip: IntegrationPoint) -> list[IntegrationPoint]:
        t = prev_ip.waypoint.runtime.seconds
        out = []
        next_deco_stop = self._calculate_next_deco_stop(prev_ip.ceiling)

        stop_time = self._params.dt.seconds
        while prev_ip.ceiling < next_deco_stop:
            t = t + self._params.dt.seconds
            new_wp = Waypoint(depth=depth, duration=self._params.dt, runtime=timedelta(seconds=t),
                              tank=prev_ip.waypoint.tank)
            new_ip = IntegrationPoint(new_wp)
            new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
            new_ip.ceilings = self._calculate_ceilings(new_ip)
            if (((self._params.gas_switch == 'stop') or (self._params.gas_switch == 'depth')) and
                    (stop_time >= (self._params.gas_switch_duration - 1))):
                new_ip.waypoint.tank = self._select_tank(new_ip.waypoint.depth)
            new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
            new_ip.cns = self._calculate_cns(new_ip, prev_ip)
            new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
            new_ip.otu = self._calculate_otu(new_ip, prev_ip)
            new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
            prev_ip = new_ip
            next_deco_stop = self._calculate_next_deco_stop(prev_ip.ceiling)

            out.append(new_ip)

            if next_deco_stop < depth:
                break

            stop_time = stop_time + self._params.dt.seconds

        while (int(floor(stop_time)) % 60) != 0:
            t = t + self._params.dt.seconds
            new_wp = Waypoint(depth=depth, duration=self._params.dt, runtime=timedelta(seconds=t),
                              tank=prev_ip.waypoint.tank)
            new_ip = IntegrationPoint(new_wp)
            new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
            new_ip.ceilings = self._calculate_ceilings(new_ip)
            new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
            if (((self._params.gas_switch == 'stop') or (self._params.gas_switch == 'depth')) and
                    (stop_time >= (self._params.gas_switch_duration - 1))):
                new_ip.waypoint.tank = self._select_tank(new_ip.waypoint.depth)
            new_ip.cns = self._calculate_cns(new_ip, prev_ip)
            new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
            new_ip.otu = self._calculate_otu(new_ip, prev_ip)
            new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
            prev_ip = new_ip
            stop_time = stop_time + self._params.dt.seconds

            out.append(new_ip)

        return out

    def _calculate_tank_pressure(self, ip: IntegrationPoint, prev_ip: IntegrationPoint, sac: float) -> list[float]:
        rmv = sac * (ip.p_amb + prev_ip.p_amb) / 2
        bar_min = rmv / self._tanks[prev_ip.waypoint.tank].size
        consumption = bar_min * prev_ip.waypoint.duration.seconds / 60

        out = []
        for p in prev_ip.tank_pressure:
            out.append(p)
        out[prev_ip.waypoint.tank] = out[prev_ip.waypoint.tank] - consumption

        return out

    def _select_tank(self, depth: float, pp_o2: float = 1.6) -> int:
        tank = 0
        max_o2 = 0
        for t in self._tanks:
            if (t.gas.O2 > max_o2) and (t.gas.mod(pp_o2) >= depth):
                tank = self._tanks.index(t)

        return tank

    def _calculate_next_gas_stop(self, depth: float, pp_o2: float = 1.6) -> float:
        out = 0.
        for t in self._tanks:
            if depth > t.gas.mod(pp_o2):
                if t.gas.mod(pp_o2) > out:
                    out = t.gas.mod(pp_o2)
        return out if self._params == 'depth' else 0.

    def _add_gas_switch_stop(self, depth: float, prev_ip: IntegrationPoint) -> list[IntegrationPoint]:
        t = prev_ip.waypoint.runtime.seconds
        out = []

        stop_time = self._params.dt.seconds
        while stop_time <= self._params.gas_switch_duration:
            t = t + self._params.dt.seconds
            new_wp = Waypoint(depth=depth, duration=self._params.dt, runtime=timedelta(seconds=t),
                              tank=prev_ip.waypoint.tank)
            new_ip = IntegrationPoint(new_wp)
            new_ip.load_ig = self._calculate_compartments(new_ip, prev_ip)
            new_ip.ceilings = self._calculate_ceilings(new_ip)
            new_ip.tank_pressure = self._calculate_tank_pressure(new_ip, prev_ip, self._params.own_ascent_sac)
            new_ip.cns = self._calculate_cns(new_ip, prev_ip)
            new_ip.cns_cum = prev_ip.cns_cum + new_ip.cns
            new_ip.otu = self._calculate_otu(new_ip, prev_ip)
            new_ip.otu_cum = prev_ip.otu_cum + new_ip.otu
            prev_ip = new_ip
            stop_time = stop_time + self._params.dt.seconds

            out.append(new_ip)

        out[-1].waypoint.tank = self._select_tank(out[-1].waypoint.depth)

        return out

    def _calculate_otu(self, ip: IntegrationPoint, prev_ip: IntegrationPoint) -> float:
        segment_time = prev_ip.waypoint.duration.seconds / 60
        pp_o2_ini = self._tanks[prev_ip.waypoint.tank].gas.ppO2(prev_ip.waypoint.depth)
        pp_o2_end = self._tanks[ip.waypoint.tank].gas.ppO2(ip.waypoint.depth)

        if ip.waypoint.depth == prev_ip.waypoint.depth:
            otu = segment_time * pow(0.5 / (pp_o2_ini - 0.5), -5. / 6.) if pp_o2_ini >= 0.5 else 0.
        else:
            max_pp_o2 = max(pp_o2_ini, pp_o2_end)
            min_pp_o2 = min(pp_o2_ini, pp_o2_end)

            if max_pp_o2 < 0.5:
                return 0.

            if min_pp_o2 < 0.5:
                low_pp_o2 = 0.5
            else:
                low_pp_o2 = min_pp_o2

            exposure_time = segment_time * ((max_pp_o2 - low_pp_o2) / (max_pp_o2 - min_pp_o2))

            pp_o2_ini = max(0.5, pp_o2_ini)
            pp_o2_end = max(0.5, pp_o2_end)
            otu = pow((pp_o2_end - 0.5) / 0.5, 11. / 6.)
            otu = otu - pow((pp_o2_ini - 0.5) / 0.5, 11. / 6.)
            otu = 3. * exposure_time / 11. * otu
            otu = otu / (pp_o2_end - pp_o2_ini)

        return otu

    def _calculate_cns(self, ip: IntegrationPoint, prev_ip: IntegrationPoint) -> float:
        segment_time = prev_ip.waypoint.duration.seconds / 60
        pp_o2_ini = self._tanks[prev_ip.waypoint.tank].gas.ppO2(prev_ip.waypoint.depth)
        pp_o2_end = self._tanks[ip.waypoint.tank].gas.ppO2(ip.waypoint.depth)

        if (pp_o2_ini < 0.5) or (pp_o2_end < 0.5):
            return 0.

        m = 0.
        b = 0.
        for k, v in noaa_cns_equations.items():
            if (pp_o2_ini >= k[0]) and (pp_o2_ini <= k[1]):
                m = v[0]
                b = v[1]
                break

        t_lim = m * pp_o2_ini + b
        if ip.waypoint.depth == prev_ip.waypoint.depth:
            cns = segment_time / t_lim
        else:
            k = (pp_o2_end - pp_o2_ini) / segment_time
            cns = log(abs(t_lim + (m * k * segment_time)))
            cns = cns - log(abs(t_lim))
            cns = cns / (m * k)

        return cns

    def _interpolate_depth(self, runtime: timedelta) -> float:
        wp_0 = None
        wp_1 = None

        for idx, wp in enumerate(self._waypoints):
            if runtime.seconds == wp.runtime.seconds:
                return round(wp.depth, 1)
            elif runtime.seconds > wp.runtime.seconds:
                wp_0 = wp
                wp_1 = self._waypoints[idx + 1]
                continue

        if (wp_0 is None) or (wp_1 is None):
            raise InterpolationError

        out = (wp_1.depth - wp_0.depth) / (wp_1.runtime.seconds - wp_0.runtime.seconds)
        out = out * (runtime.seconds - wp_0.runtime.seconds)
        out = out + wp_0.depth

        return round(out, 1)
