# This file is part of YABAI.

# YABAI is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public
# License # as published by the Free Software Foundation, either version 3 of the License, or any later version.

# YABAI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.

# You should have received a copy of the Affero GNU General Public License along with YABAI.
# If not, see <https://www.gnu.org/licenses/>.

from datetime import timedelta
from src.yabai import Parameters, Tank, Gas, Waypoint, Profile

# Initialize setup object
params = Parameters(dt=timedelta(seconds=1), gf_low=1., gf_high=1., last_stop_depth=3, gas_switch_mode='stop')

# Create tanks
tanks = [Tank(size=15, gas=Gas(o2=21), start_pressure=200),
         Tank(size=10, gas=Gas(o2=50), start_pressure=200)]

# Create waypoints
waypoints = [Waypoint(45, 75)]  # , Waypoint(30, 2)]
# waypoints = [Waypoint(45, 7), Waypoint(45, (45 - 5) / params.v_asc),
#              Waypoint(5, 3), Waypoint(0, 0)]
# waypoints = [Waypoint(45, 25), Waypoint(15, 10), Waypoint(45, 0)]

# Create profile
profile = Profile(waypoints=waypoints, tanks=tanks, params=params)

for ip in profile.integration_points:
    pass
    print(ip)

for wp in profile.waypoints:
    print(wp)

profile.plot_waypoints()
profile.plot_integration_points()
profile.plot_compartments('N2')
profile.plot_ceilings()
profile.plot_ceiling()
profile.plot()
