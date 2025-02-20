# Final project
# Ajit Badwal
# Space shuttle simulator to different planets in our solar system

# imports all that is needed
import numpy as np
import matplotlib.pyplot as plt
import requests # needed for weather
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D


# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24  # Mass of the Earth (kg)
R = 6371000  # Radius of the Earth (m)
rocket_cross_sectional_area = 16.6  # m² (estimated cross-sectional area of the space shuttle)
burn_rate = 4989.5161  # Fuel consumption rate (kg/s)
vacuum_thrust_force = 213188 # oxygen and hydrogen force expelled
# Rocket-specific parameters
# space shuttle model
initial_fuel_mass = 1995806.428  # Initial fuel mass (kg)
rocket_mass = 2030000  # Initial mass of the rocket (kg)
latitude = 28.3922  # Launch latitude of cape canaveral
longitude = -80.605659  # Launch longitude cape canaveral
density_at_launch_fully_loaded = 4409245.244 / 2625  # in pounds/ft^3
# Time settings
air_density_at_sea_level = 1.225 # kg/m^3
t_max = 600  # Simulate for 10 minutes (600 seconds)
dt = 0.1  # Smaller time step for more precision at different times Important ^*
num_steps = int(t_max / dt) # number of steps in program so 6000 different intervals
# Constants
R_EARTH = 6371000  # Earth's radius in meters
OMEGA_EARTH = 7.2921e-5  # Earth's angular velocity in rad/s
EARTH_RADIUS = 6371000  # Radius of the Earth in meters (mean radius)
altitude = 3.05  # in meters above sea level
velocity = 447  # in meters per second
phi = np.radians(latitude) # turn into cartesian grid
phi2 = np.radians(longitude)
# Tangential velocity due to Earth's rotation, speed it has orbiting the planet
tangential_velocity = OMEGA_EARTH * R_EARTH * np.cos(phi)
fuel_mass = 1995806.428  # Total fuel mass (kg)
nozzlelength = 3.1 # in meters long
nozzle_diameter = 0.26 # meters
nozzle_throat = 2.30 # meters at exit
rocket_position = np.array([latitude, longitude, R + 3.05])
rocket_velocity = np.array([0, 0, 0])  # Starting velocity
fuel_mass = initial_fuel_mass

orbital_radius_earth = 1.50 * 10**8
orbital_radius_mars = 2.28 * 10**8
# Weather conditions check before simulation starts
# Set weather-based conditions for launch suitability
max_wind_speed = 21.6067  # m/s
min_temp = 0.555556  # Minimum temperature for launch (in Celsius)
max_temp = 37.2222  # Maximum temperature for launch (in Celsius)
max_rain = 0  # Maximum rain in mm per hour
max_snow = 0  # Maximum snow in mm per hour
max_clouds = 50

# Initialize lists to store positions and times
positions = []
times = []
# information of planets
planets = {
    "Earth": {"mass": 5.972e24, "radius": 6.371e6, "atmosphere_height": 100e3, "escape_velocity": 4.0270e4}, "orbital_radius": 1.5e8,
    "Mars": {"mass": 0.64171e24, "radius": 3.389e6, "atmosphere_height": 11e3, "escape_velocity": 5.03e3, "orbital_radius": 2.28e8},
    "Moon": {"mass": 0.0734767309e24 , "radius": 1.7374e6, "atmosphere_height": 0.1e-3, "escape_velocity": 8.6e3, "orbital_radius": 0.00385e8},
}
destination = "Mars"
if destination == "Mars":
    planet_data = planets[destination]
    rocket_mass = 11000000 # pounds
    fuel_mass = 3300000
destination = "Moon"
if destination == "moon":
    planet_data = planets[destination]
    rocket_mass = 13928134 # pounds
    fuel_mass = 7928134

# set conditions for area of launch
city_name = 'cape canaveral'
state_name = 'florida'
# Weather data fetch function (fetches weather data based on city and state)
def fetch_weather_data(city_name, state_name):

    api_key = "4f0142b007a72b3a3020c9ceb060bcf8"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name + "," + state_name + ",us"

    # Get response from the API
    response = requests.get(complete_url)

    # Analyze the JSON data
    data = response.json()

    if data["cod"] == 200:
        wind_speed = data["wind"]["speed"]  # Wind speed in m/s
        weather_description = data["weather"][0]["description"]  # Weather description
        temp = data["main"]["temp"] - 273.15  # Temperature in Celsius (convert from Kelvin)
        pressure = data["main"]["pressure"]  # Pressure in hPa
        humidity = data["main"]["humidity"]  # Humidity in percentage
        clouds = data["clouds"]["all"]  # Cloudiness in percentage
        rain = data.get("rain", {}).get("1h", 0)  # Rain volume (if any) in last hour (mm)
        snow = data.get("snow", {}).get("1h", 0)  # Snow volume (if any) in last hour (mm)
        lat = data["coord"]["lat"]  # Latitude
        lon = data["coord"]["lon"]  # Longitude
        cloud_base = data["clouds"].get("base", 0)  # Cloud base altitude in meters

        # Convert cloud base from meters to kilometers
        cloud_base_km = cloud_base / 1000.0  # Convert meters to kilometers

        return wind_speed, weather_description, temp, pressure, humidity, clouds, rain, snow, lat, lon, cloud_base_km
    else:
        print(f"Error: {data.get('message', 'City not found')}")
        return None, None, None, None, None, None, None, None, None, None, None


# Fetch weather data for the entered city and state
wind_speed, weather_description, temp, pressure, humidity, clouds, rain, snow, lat, lon, cloud_base_km = fetch_weather_data(
    city_name, state_name)

if wind_speed is None:
    print("Exiting simulation due to weather data error.")
    exit()  # Exit if weather data is not fetched successfully

# Print weather conditions
print(
    f"Initial weather: {weather_description}, Wind speed: {wind_speed} m/s, Temperature: {temp}°C, Pressure: {pressure} hPa, Cloud base: {cloud_base_km} km, Rain: {rain}mm, Snow: {snow}mm")

# Weather conditions check before simulation starts (printing warnings instead of exiting)
launch_suitable = True

if wind_speed > max_wind_speed:
    print(f"Warning: Launch conditions are not suitable due to high wind speed: {wind_speed} m/s")
    launch_suitable = False
if temp < min_temp or temp > max_temp:
    print(f"Warning: Launch conditions are not suitable due to temperature: {temp}°C")
    launch_suitable = False
if rain > max_rain or snow > max_snow:
    print(f"Warning: Launch conditions are not suitable due to precipitation: Rain: {rain} mm/h, Snow: {snow} mm/h")
    launch_suitable = False
if clouds > max_clouds:
    print(f"Warning: Launch conditions are not suitable due to cloudiness: {clouds}%")
    launch_suitable = False

# Ask the user whether to continue
if not launch_suitable:
    continue_launch = input(
        "The launch conditions are not ideal. Do you want to continue with the launch? (y/n): ").lower()
    if continue_launch != 'y':
        print("Simulation halted due to unsuitable launch conditions.")
        exit()


# Function to update the fuel mass
def update_fuel(fuel_mass, burn_rate, dt):
    return fuel_mass - burn_rate * dt


position = rocket_position
mass = rocket_mass
fuel = fuel_mass


# You can integrate this code into your simulation by updating the position at each step.

def escape_velocity(position):
    """Calculate escape velocity from a given position (m)."""
    r = np.linalg.norm(position)  # Use full 3D distance
    return np.sqrt(2 * G * M / r)

def generate_launch_path(earth_radius, orbit_radius, num_points=100):
    """Simulate a realistic curved path from Earth's surface to the orbit."""
    t = np.linspace(0, 1, num_points)  # Parameter for the trajectory (0 to 1) ((((start, stop))__
    r = earth_radius + (orbit_radius - earth_radius) * t**2  # Quadratic ascent
    theta = np.pi / 4 * t  # Gradually increase angle (45 degrees max)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0.01 * earth_radius * t  # Gradual upward motion
    return np.vstack((x, y, z)).T

def generate_circular_orbit(radius, num_points=200):
    """Generate a circular orbit in the ecliptic plane."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)  # Flat orbit in the ecliptic plane same as x array
    return np.vstack((x, y, z)).T # like matrix

def generate_hohmann_transfer(earth_radius, mars_radius, num_points=500):
    """
    Generate a 3D trajectory for a Hohmann transfer orbit.
    :parameter earth_radius: Radius of Earth's orbit (km).
    :parameter mars_radius: Radius of Mars' orbit (km).
    :parameter num_points: Number of points in the trajectory.
    :return: Numpy array of (x, y, z) points.
    """
    # Semi-major axis of the transfer ellipse
    semi_major_axis = (earth_radius + mars_radius) / 2

    # Eccentricity of the ellipse how flat or round the shape of the ellipse is
    eccentricity = (mars_radius - earth_radius) / (earth_radius + mars_radius)

    # True anomaly (angle in radians) from 0 to π (half orbit) to reach mars
    theta = np.linspace(0, np.pi, num_points)

    # Parametric equations for the ellipse in the ecliptic plane (z = 0)
    #r is the distance from the center of the central body (such as the Earth) to the object (such as a satellite or planet) at a given point in its orbit
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta) # position of rocket in x cartesian
    y = r * np.sin(theta)# y cartesian
    z = np.zeros_like(x)  # Assuming planar motion in the ecliptic plane for simple program
    return np.vstack((x, y, z)).T


# Function to compute the gravitational force
def gravitational_force(position):
    epsilon = 1e-6  # Small value to avoid division by zero
    r = np.linalg.norm(position)  # Use full 3D distance (magnitude of position)

    if r < epsilon:
        return np.array([0.0, 0.0, 0.0])  # Return a 3D vector

    elif r < R:
        gravity_force = np.zeros(3)  # No gravity if below Earth's surface
    else:
        gravity_force = -G * M * position / r ** 3  # Normal gravity force

    force_magnitude = G * M / r ** 2
    force_direction = -position / r  # Unit vector pointing towards the planet
    return force_magnitude * force_direction  # 3D vector for force


# Velocity components (3D in the global reference frame)
vx = tangential_velocity * np.cos(phi2)  # Eastward component
vy = tangential_velocity * np.sin(phi)  # Northward component


def air_density(new_altitude, temp, pressure):
    """Calculate air density at a given altitude (in meters) without capping at 8500m."""
    # Earth's average radius in meters
    earth_radius = 6371000  # in meters

    # Adjust the altitude to be relative to the Earth's surface
    effective_altitude = new_altitude

    # Avoid negative altitude (below the Earth's surface)
    if effective_altitude < 0:
        effective_altitude = 0

    kelvin = temp + 273.15  # Convert Celsius to Kelvin

    pressure1 = pressure * 100  # Convert hPa to Pa (only if pressure is in hPa)

    gas_constant = 287.058  # Specific gas constant for dry air (J/(kg·K))
    rho1 = pressure1 / (gas_constant * kelvin)  # Ideal gas law for air density at sea level

    H = 8500  # Scale height of the atmosphere in meters

    # Calculate air density using the scale height with the effective altitude
    air_density_at_altitude = rho1 * np.exp(-effective_altitude / H)  # Adjust for altitude

    return air_density_at_altitude


# Drag and lift coefficients based on angle of attack and flap configuration
# took a horrendious amount of time
def get_drag_coefficient(angle_of_attack, flap_configuration):
    """Return the drag coefficient based on angle of attack and flap configuration."""
    if flap_configuration == 0:
        if angle_of_attack == 0:
            return 0.13
        elif angle_of_attack == 5:
            return 0.14
        elif angle_of_attack == 10:
            return 0.185
        elif angle_of_attack == 15:
            return 0.38
    elif flap_configuration == 15:
        if angle_of_attack == 0:
            return 0.15
        elif angle_of_attack == 5:
            return 0.19
        elif angle_of_attack == 10:
            return 0.215
        elif angle_of_attack == 15:
            return 0.45
    elif flap_configuration == 40:
        if angle_of_attack == 0:
            return 0.202
        elif angle_of_attack == 5:
            return 0.240
        elif angle_of_attack == 10:
            return 0.293
        elif angle_of_attack == 15:
            return 0.0  # No data provided for 15 degrees at 40-degree flap
    # Default case for unsupported configurations
    return 0.0


def get_lift_coefficient(angle_of_attack, flap_configuration):
    """Return the lift coefficient based on angle of attack and flap configuration."""
    if flap_configuration == 0:
        if angle_of_attack == 0:
            return 0.00
        elif angle_of_attack == 5:
            return 0.45
        elif angle_of_attack == 10:
            return 0.93
        elif angle_of_attack == 15:
            return 0.75
    elif flap_configuration == 15:
        if angle_of_attack == 0:
            return 0.30
        elif angle_of_attack == 5:
            return 0.75
        elif angle_of_attack == 10:
            return 1.30
        elif angle_of_attack == 15:
            return 1.20
    elif flap_configuration == 40:
        if angle_of_attack == 0:
            return 0.50
        elif angle_of_attack == 5:
            return 0.93
        elif angle_of_attack == 10:
            return 1.58
        elif angle_of_attack == 15:
            return 1.20
    # Default case for unsupported configurations
    return 0.0
def adjust_flap_and_aoa(mach_number, altitude, velocity):
    """
    Adjust flap configuration and AoA based on altitude, Mach number, and velocity.
    Handles both scalar and array inputs for altitude and Mach number.
    """
    # Ensure altitude and mach_number are arrays
    altitude = np.atleast_1d(altitude)
    mach_number = np.atleast_1d(mach_number)
    # Initialize outputs
    flap_configurations = []
    angles_of_attack = []
    drag_coefficients = []
    lift_coefficients = []

    for alt, mach in zip(altitude, mach_number):
        if alt < 10000:  # Below 10 km
            flap_configuration, angle_of_attack = 0, 0
        elif 10000 <= alt < 20000:  # Between 10 km and 20 km
            flap_configuration, angle_of_attack = 5, 5
        elif alt > 20000:  # Above 20 km
            flap_configuration, angle_of_attack = 5, 10
        elif alt > 30000:
            flap_configuration, angle_of_attack = 10, 15
        else:
            flap_configuration, angle_of_attack = 15, 15

        # Get coefficients
        drag_coefficient = get_drag_coefficient(angle_of_attack, flap_configuration)
        lift_coefficient = get_lift_coefficient(angle_of_attack, flap_configuration)

        # Append results
        flap_configurations.append(flap_configuration)
        angles_of_attack.append(angle_of_attack)
        drag_coefficients.append(drag_coefficient)
        lift_coefficients.append(lift_coefficient)

    # Return all 4 values
    return flap_configurations[0], angles_of_attack[0], drag_coefficients[0], lift_coefficients[0]

# resistance force caused by air or other fluid
def drag_force1(velocity, altitude, rocket_cross_sectional_area, drag_coefficient, temp, pressure):
    # Ensure velocity is a NumPy array
    velocity = np.array(velocity)
    density = air_density(altitude, temp, pressure)
    # Calculate air density (use `density` directly or compute it based on altitude)
    rho = density  # Replace this with your actual calculation if needed

    # Calculate velocity magnitude
    velocity_magnitude = np.linalg.norm(velocity)

    # Calculate drag force
    drag_force = 0.5 * rho * velocity_magnitude ** 2 * drag_coefficient * rocket_cross_sectional_area
    return drag_force


def drag_force(velocity, altitude, cross_sectional_area, temp, pressure, drag_coefficients, mach_number, angle_of_attack, flap_configuration):
    """Calculate the drag force based on velocity, altitude, area, and flap configuration."""
    # Ensure velocity is a numpy array for vectorized operations
    velocity = np.array(velocity)

    # Adjust flap and angle of attack, unpack the tuple if necessary
    drag_coefficient, angle_of_attack, flap_configuration, _ = adjust_flap_and_aoa(mach_number, altitude, velocity)

    # Make sure drag_coefficient is a scalar
    drag_coefficient = float(drag_coefficient)

    # Get air density at the given altitude
    rho = air_density(altitude, temp, pressure)
    rho = float(rho)

    # Convert cross-sectional area to scalar
    cross_sectional_area = float(cross_sectional_area)

    # Compute the speed (magnitude of the velocity vector)
    v = np.linalg.norm(velocity)  # Calculate the magnitude of the velocity vector

    # If the velocity is not zero, calculate the drag force
    if v != 0:
        drag_magnitude = 0.5 * drag_coefficient * rho * cross_sectional_area * v ** 2
        drag_force_vector = -drag_magnitude * velocity  # Drag force acts opposite to the direction of velocity
    else:
        drag_force_vector = np.zeros(3)  # Ensure drag is a zero 3D vector when no velocity

    return drag_force_vector  # Return only the drag force vector

# Function to calculate engine efficiency (specific impulse)
def engine_efficiency(altitude):
    """Return specific impulse based on altitude (in meters)."""
    base_Isp = 366  # ssme calculation
    Isp_in_space = 452
    return base_Isp + (altitude / 10000)  # Example linear increase with altitude

# an apparent force caused by the earth's rotation coriolis effect
def coriolis_effect(velocity, latitude, longitude):
    omega = 7.2921159e-5  # Earth's angular velocity in rad/s

    # Ensure velocity is a numpy array with the correct shape
    velocity = np.asarray(velocity)
    # If rocket_velocity has 4 elements, make sure you extract the correct 3 elements:
    velocity = rocket_velocity[:3]  # Extract the first 3 elements for velocity

    # Ensure that velocity is a 1D array with 3 elements:
    if not isinstance(velocity, np.ndarray):
        velocity = np.array(velocity, dtype=float)
    if velocity.ndim != 1 or velocity.shape[0] != 3:
        raise ValueError(f"Velocity must be a 1D array with 3 elements. Current shape: {velocity.shape}")

        # Compute the Coriolis acceleration
    coriolis_acceleration = -2 * mass * (omega*velocity)

    return coriolis_acceleration

# Function to compute the exhaust velocity based on altitude
def exhaust_velocity(altitude):
    """Compute the exhaust velocity using the specific impulse.
    the speed at which exhaust gases leave a rocket's engine nozzle relative to the rocket
"""
    Isp = engine_efficiency(altitude)  # Specific impulse at current altitude
    g0 = 9.81  # Standard gravitational acceleration (m/s^2)
    return Isp * g0  # Exhaust velocity (m/s)

def thrust_force_vector(rocket_velocity, altitude, burn_rate):
    """Compute the thrust force vector based on exhaust velocity, altitude, and burn rate."""
    # Calculate the exhaust velocity based on altitude
    v_e = exhaust_velocity(altitude)
    p0 = 4350 # starting pressure
    free_stream_velocity = 7777.78 #m/s
    # Calculate the mass flow rate (burn rate)
    mass_flow_rate = 514.49  # kg/s
    pe= 5 # exit pressure
    A_e = 0.806 # estimated exit area of nozzle
    # Compute the thrust force
    thrust = mass_flow_rate * v_e - (mass_flow_rate * free_stream_velocity) + (pe - p0) * A_e

    # Calculate the thrust direction based on velocity (normalized)
    if np.linalg.norm(rocket_velocity) == 0:
        return np.array([0.0, 0.0, 0.0])  # No thrust if velocity is zero

        # Calculate the direction of the thrust vector (normalized rocket velocity)
    direction = rocket_velocity / np.linalg.norm(rocket_velocity)

    # The thrust force vector is the thrust magnitude multiplied by the direction
    thrust_vector = thrust * np.abs(rocket_velocity) / np.linalg.norm(rocket_velocity)

    return thrust_vector

def new_velocity_for_z(mach_number, velocity, rocket_position, rocket_velocity, temp, pressure, dt,
                       rocket_cross_sectional_area, rocket_mass, gravitational_force, vy, vx):
    # Ensure rocket_velocity is a numpy array with correct shape
    if isinstance(rocket_velocity, tuple):
        rocket_velocity = rocket_velocity[0]  # Extract the array from the tuple

    # If rocket_velocity is a scalar, initialize it as a 3D array if nothing in the set velocity
    if rocket_velocity.ndim == 0:
        rocket_velocity = np.array([0.0, 0.0, 0.0])

    # Ensure rocket_velocity is 3-dimensional, initialize with zeros if needed
    elif rocket_velocity.ndim == 1 and len(rocket_velocity) < 3:
        # adds a extra array if the len is less than 3 and makes sure it is always 3
        rocket_velocity = np.pad(rocket_velocity, (0, 3 - len(rocket_velocity)))

    # Adjust flap configuration and angle of attack
    flap_results = adjust_flap_and_aoa(mach_number, altitude, np.linalg.norm(rocket_velocity))
    flap_configuration = flap_results[0] # gets flap config
    angle_of_attack = flap_results[1] # gets aoa froms second agrument

    # Calculate drag force
    D = drag_force(velocity, rocket_position[0]+rocket_position[1]+rocket_position[2], rocket_cross_sectional_area, flap_configuration, angle_of_attack, temp,
                   pressure, mach_number, flap_configuration)

    p0 = 4350
    free_stream_velocity = 7777.78 #m/s
    # Calculate the mass flow rate (burn rate)
    mass_flow_rate = 514.49  # kg/s
    pe= 5
    A_e = 0.806 # estimated exit area of nozzle
    gamma = 1.4
    gas_constant_2 = .286 # kj/kg/k
    t = 3000 * (1+ (1.4-1)/2 * mach_number**2)**-1
    v_e = mass_flow_rate * np.sqrt(gamma*gas_constant_2*t)

    # Compute the thrust force
    thrust = mass_flow_rate * v_e - (mass_flow_rate * free_stream_velocity) + (pe - p0) * A_e
    # Calculate accelerations in all directions
    thrust_force_x = thrust * np.cos(angle_of_attack)  # Thrust in the x-direction
    thrust_force_y = thrust * np.sin(angle_of_attack)  # Thrust in the y-direction
    a_z = (thrust / rocket_mass) - gravitational_force(rocket_position)[2] - (D / rocket_mass)  # Z-acceleration
    a_x = thrust_force_x / rocket_mass  # Acceleration in the x-direction
    a_y = thrust_force_y / rocket_mass  # Acceleration in the y-direction
    # Ensure a_z is scalar
    if np.ndim(a_z) > 0:
        a_z = a_z[2]  # If a_z is an array, select the z-component

    # Update velocity components
    rocket_velocity[0] += a_x * dt  # a_x is an array and  want to use the first component
    rocket_velocity[1] += a_y * dt  # Update vy (y-direction)
    rocket_velocity[2] += a_z * dt  # Update vz (z-direction)

    # Return updated rocket velocity
    updated_velocity = rocket_velocity[0], rocket_velocity[1], rocket_velocity[2]  # rocket_velocity is already a numpy array
    print(altitude)
    return updated_velocity, rocket_velocity[2]  # Return the z-component (vz) along with updated velocity


# Assuming initial velocity and temperature are available
mach_number = 1.6  # start off launch

velocity_magnitude = np.linalg.norm(rocket_velocity)
speed_of_sound = 331 + (0.61 * temp)  # Based on temperature
mach_number = velocity_magnitude / speed_of_sound  # Update Mach number with initial velocity

# didnt use too complicated for mine
def hohmann_transfer_velocity_change(orbital_radius_current, orbital_radius_target):
    """Calculate the velocity change for a Hohmann transfer between two orbits."""
    # Semi-major axis of the elliptical transfer orbit
    a = (orbital_radius_current + orbital_radius_target) / 2

    # Velocity at periapsis (Earth orbit)
    v_periapsis = np.sqrt(G * M * (2 / orbital_radius_current - 1 / a))

    # Velocity at apoapsis (target planet orbit)
    v_apoapsis = np.sqrt(G * M * (2 / orbital_radius_target - 1 / a))

    # Delta-v for the transfer burn
    delta_v = v_periapsis - escape_velocity([orbital_radius_current, 0.0, 0.0])  # From circular to elliptical orbit
    return delta_v, v_periapsis, v_apoapsis


def runge_kutta_4(rocket_position, rocket_velocity, rocket_mass, fuel_mass, temp, pressure, dt, mach_number,
                  rocket_velocity_in):

    r = rocket_position
    rocket_velocity, _ = new_velocity_for_z(
        mach_number,
        rocket_velocity_in,
        rocket_position,
        rocket_velocity,
        temp,
        pressure,
        dt,
        rocket_cross_sectional_area,
        rocket_mass,
        gravitational_force,
        vy,
        vx
    )

    v = rocket_velocity

    # Ensure velocity is initialized as a 3-element array (x, y, z)
    if isinstance(v, (tuple, list)):
        velocity = np.array(v, dtype=float)
    if velocity.ndim != 1 or velocity.shape[0] != 3:
        raise ValueError(f"Velocity must be a 1D array with 3 elements. Current: {velocity}")

    # Ensure rocket_velocity is a 3-element numpy array
    if len(rocket_velocity) < 3:
        rocket_velocity = np.concatenate([rocket_velocity, np.zeros(3 - len(rocket_velocity))])

    # Now rocket_velocity is guaranteed to have 3 elements (x, y, z)
    rocket_velocity = rocket_velocity[:3]

    # Compute the Mach number
    speed_of_sound = 331.3 + (0.606 * temp)
    mach_number = np.linalg.norm(rocket_velocity) / speed_of_sound

    # Normalize velocity for calculations
    norm_v = np.linalg.norm(rocket_velocity)

    # Update flap configuration and coefficients
    flap_configuration, angle_of_attack, drag_coefficients, lift_coefficients = adjust_flap_and_aoa(
        mach_number, r, norm_v)

    def compute_forces(r, v, m, fuel_rate, temp, pressure, mach_number):
        gravity = gravitational_force(r)
        drag_raw = drag_force(v, altitude, rocket_cross_sectional_area, temp, pressure, drag_coefficients, mach_number, angle_of_attack, flap_configuration)

        if drag_raw is None:
            raise ValueError("Drag force calculation returned None")

        thrust = thrust_force_vector(v, r, fuel_rate)

        coriolis = coriolis_effect(v, latitude, longitude)
        if coriolis is None:
            coriolis = np.zeros(3)

        net_force = gravity + drag_raw + thrust + coriolis

        return net_force / m

    fuel_rate = burn_rate if fuel_mass > 0 else 0  # Ensure burn rate is zero when fuel runs out

    # Compute the Mach number
    mach_number = np.linalg.norm(v) / speed_of_sound
    acceleration = compute_forces(r, v, rocket_mass, fuel_rate, temp, pressure, mach_number)

    # Runge-Kutta coefficients (for each velocity component)
    k1_v = dt * acceleration
    k1_r = dt * rocket_velocity[0] * rocket_velocity[1] * rocket_velocity[2]

    # Compute intermediate forces for k2
    r_k2 = r + 0.5 * k1_r
    v_k2 = rocket_velocity + 0.5 * k1_v
    m_k2 = rocket_mass - 0.5 * fuel_rate * dt  # Intermediate mass at step
    acceleration_k2 = compute_forces(r_k2, v_k2, m_k2, fuel_rate, temp, pressure, mach_number)

    k2_v = dt * acceleration_k2
    k2_r = dt * v_k2

    # Compute intermediate forces for k3
    r_k3 = r + 0.5 * k2_r
    v_k3 = rocket_velocity + 0.5 * k2_v
    m_k3 = rocket_mass - 0.5 * fuel_rate * dt
    acceleration_k3 = compute_forces(r_k3, v_k3, m_k3, fuel_rate, temp, pressure, mach_number)

    k3_v = dt * acceleration_k3
    k3_r = dt * v_k3

    # Compute intermediate forces for k4
    r_k4 = r + k3_r
    v_k4 = rocket_velocity + k3_v
    m_k4 = rocket_mass - fuel_rate * dt  # Final mass after full burn
    acceleration_k4 = compute_forces(r_k4, v_k4, m_k4, fuel_rate, temp, pressure, mach_number)

    k4_v = dt * acceleration_k4
    k4_r = dt * v_k4

    # Update rocket velocity using the Runge-Kutta formula for each component (x, y, z)
    rocket_velocity += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6 # 1+2+2+1 = 6 /6 to normalize

    # Update fuel mass
    fuel_mass -= fuel_rate * dt
    if fuel_mass < 0:
        fuel_mass = 0

    print(rocket_velocity)
    return rocket_position, rocket_velocity, fuel_mass, mach_number, rocket_mass

def update_position(altitude, dt, mach_number, velocity, position, rocket_velocity, temp, pressure,
                    rocket_cross_sectional_area, rocket_mass, gravitational_force, vy, vx):

    # Update velocity in the z-direction using the new_velocity_for_z function
    vz, real = new_velocity_for_z(
        mach_number, velocity, position, rocket_velocity, temp, pressure, dt,
        rocket_cross_sectional_area, rocket_mass, gravitational_force, vy, vx
    )

    # Calculate new altitude based on the vertical velocity and time step (dt)
    new_altitude = altitude + real * dt

    return new_altitude


def simulate_rocket_flight(rocket_mass, initial_fuel_mass, num_steps, dt, city_name, state_name, rocket_position, rocket_velocity, mach_number):
    position = np.array(rocket_position)  # Launch position (assuming launch at Earth's surface)
    velocity = np.array(rocket_velocity)  # Initial velocity
    fuel_mass = initial_fuel_mass  # Initial fuel mass
    weather = fetch_weather_data(city_name, state_name)  # Standard temperature and pressure at sea level
    rocket_mass = rocket_mass  # Total mass of rocket

    positions = []  # List to store positions at each step
    times = []  # List to store time steps
    velocities = []  # List to store velocities
    fuel_masses = []  # List to store remaining fuel masses

    for t in range(num_steps):
        time = t * dt
        positions.append(position.copy())
        velocities.append(velocity.copy())
        fuel_masses.append(fuel_mass)
        times.append(time)

        # Recalculate Mach number at each step (velocity and temp are updated)
        speed_of_sound = 331.3 + (0.606 * temp)
        mach_number = np.linalg.norm(velocity) / speed_of_sound

        # Run the Runge-Kutta integration step
        position, velocity, fuel_mass, _, _ = runge_kutta_4(position, velocity, rocket_mass, fuel_mass, temp, pressure, dt, mach_number, velocity)

    return times, positions, velocities, fuel_masses
times, positions, velocities, fuel_masses = simulate_rocket_flight(
    rocket_mass, initial_fuel_mass, num_steps, dt, city_name, state_name, rocket_position, rocket_velocity, mach_number
)

# Print or process the results
print(times)
for t in range(num_steps):
    # Ensure mach_number is updated correctly before passing to the function
    position, velocity, mass, fuel, mach_number = runge_kutta_4(
        position,
        np.array(velocity, dtype=float),  # Ensure velocity is a NumPy array
        mass,
        fuel,
        temp,
        pressure,
        dt,
        mach_number,
        np.array(velocity, dtype=float)  # Pass velocity correctly
    )

    # Extract position and velocity components
    latitude, longitude, altitude = position

    # Update the rocket's position at each time step using the update_position function
    altitude += np.sqrt(rocket_velocity[0]**2 + rocket_velocity[1]**2 + rocket_velocity[2]**2) * dt
    print(altitude)

    # Append time and position data for plotting
    times.append(t * dt)
    # Calculate the updated velocity for the rocket in the z-direction
    rocket_velocity, _ = new_velocity_for_z(
        mach_number, velocity, position, rocket_velocity, temp, pressure, dt,
        rocket_cross_sectional_area, rocket_mass, gravitational_force, vy, vx
    )
    # Set updated rocket mass and fuel values
    mass = rocket_mass  # Current mass of the rocket (including fuel)
    fuel = fuel_mass  # Current fuel mass

    # Break if the rocket runs out of fuel or falls back to Earth
    if fuel <= 0 or position[2] < R:  # If altitude is below Earth's radius
        break

    # Print every 100 steps
    if t % 100 == 0:
        print(f"Step {t}: Time {t * dt:.1f} seconds, Position", f"Altitude: {altitude} meters, Velocity: {np.sqrt(rocket_velocity[0]**2 + rocket_velocity[1]**2 + rocket_velocity[2]**2):.1f} m/s, "        )

# Parameters for the simulation
time_interval = 10  # Time interval between each step (in seconds)
total_steps = 500  # Total number of steps for the simulation

x = R * np.cos(latitude) * np.cos(longitude)
y = R * np.cos(latitude) * np.sin(longitude)
z = np.sin(latitude)
# Initial conditions
position = np.array([x, y, z + altitude])  # This is a NumPy array
# Assuming rocket_velocity is a tuple like:
# (array([-8360.16321866, -8229.92822555, -2540.21949432]), np.float64(-2540.219494321917))

# Extract the first element (the array) and use it
velocity = np.array(rocket_velocity[0])  # This will be the velocity array

exit_time = None  # Variable to track the time the rocket exits the atmosphere

# Function to update the position
def update_position1(position, velocity, time_interval):
    # Simplified example of updating position based on velocity
    return position + velocity * time_interval


# Function to update the velocity (simple model for demonstration)
def update_velocity1(velocity, time_interval):
    # Example with gravity and constant thrust
    g = 9.81  # Gravitational acceleration (m/s^2)
    thrust = thrust_force_vector(rocket_velocity, altitude, burn_rate)

    # Update velocity in each direction
    return [
        velocity[0] + thrust[0],  # No thrust in x-direction
        velocity[1]+ thrust[1],
        velocity[2] + (thrust[2] + g) * time_interval  # z-direction velocity
    ]

def atmospheric_exit_simulation(rocket_position, rocket_velocity, planet_data, dt):
    trajectory = []
    inclinations = []
    distances = []

    while rocket_position[2] < planet_data["atmosphere_height"]:
        drag = drag_force(rocket_velocity, rocket_position[2], cross_sectional_area, temp, pressure,
                          drag_coefficients, mach_number, angle_of_attack, flap_configuration)
        # newtons grav constant with planets
        gravity = -planet_data["mass"] * 6.67430e-11 / (rocket_position[2] + planet_data["radius"])**2
        thrust = mass_flow_rate * v_e - (mass_flow_rate * free_stream_velocity) + (pe - p0) * A_e
        net_force = thrust - drag + gravity
        rocket_velocity += net_force * dt / rocket_mass
        rocket_position += rocket_velocity * dt

        # Store data for trajectory, inclinations, and distances
        trajectory.append(rocket_position.tolist())
        inclination, distance = position_relative_to_ecliptic(rocket_position, rocket_velocity)
        inclinations.append(inclination)
        distances.append(distance)

    return np.array(trajectory), inclinations, distances

def position_relative_to_ecliptic_coordinate_system(rocket_position, rocket_velocity):
    # Calculate angular momentum vector
    h = np.cross(rocket_position, rocket_velocity)

    # Calculate inclination
    if np.linalg.norm(h) > 0:
        inclination = np.arccos(h[2] / np.linalg.norm(h))
    else:
        inclination = 0  # Or handle this scenario appropriately
    inclination_deg = np.degrees(inclination)

    # Distance from ecliptic plane (z-component)
    distance_from_ecliptic = abs(rocket_position[2])

    return inclination_deg, distance_from_ecliptic


target_planet = "Mars"

def launch_with_curved_trajectory(rocket_position, rocket_velocity):
    trajectory = np.empty((0, 3))  # Initialize as an empty 2D array
    inclinations = []
    distances = []

    max_angle_of_attack = np.radians(90)
    min_angle_of_attack = np.radians(0)
    atmosphere_height = planets["Earth"]["atmosphere_height"]

    print("Initial rocket position:", rocket_position)
    print("Atmosphere height:", atmosphere_height)

    while rocket_position[2] - planets["Earth"]["radius"] < atmosphere_height:
        altitude = rocket_position[2] - planets["Earth"]["radius"]


        # Perform calculations (gravity, drag, thrust, etc.)
        # Example drag and thrust calculations:
        drag_coefficient = get_drag_coefficient(0, 0)  # Replace with actual parameters
        drag = drag_force1(rocket_velocity, altitude, rocket_cross_sectional_area, drag_coefficient, temp, pressure)
        distance = rocket_position[2] + planets["Earth"]["radius"]
        if distance > 0:
            gravity = -planets["Earth"]["mass"] * 6.67430e-11 / min(distance ** 2, 1e20)
        else:
            gravity = 0  # Handle edge case

        thrust = thrust_force_vector(rocket_velocity, altitude, burn_rate) # Replace with actual thrust calculation
        net_force = thrust - drag + np.array([9.81, 9.81, gravity])
        rocket_velocity += net_force * dt / rocket_mass
        rocket_position += rocket_velocity * dt

        # Append position to trajectory
        earth_orbit_radius = 149.6e6  # Earth's orbit in km
        mars_orbit_radius = 227.9e6  # Mars' orbit in km
        trajectory = generate_hohmann_transfer(earth_orbit_radius, mars_orbit_radius)

        # Debugging shapes

        inclination, distance = position_relative_to_ecliptic_coordinate_system(rocket_position, rocket_velocity)
        inclinations.append(inclination)
        distances.append(distance)

    print("Exiting Earth's atmosphere...")
    print(f"Final position: {rocket_position}")
    print(f"Final velocity: {rocket_velocity}")

    return trajectory, inclinations, distances


trajectory, inclinations, distances = launch_with_curved_trajectory(rocket_position, rocket_velocity)

# Visualize the trajectory
er = 6378
G2 = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M2 = 5.972e24  # Mass of the Earth (kg)

t1 = 2*np.pi*er**(3/2)/(np.sqrt(G2*M2))
print(t1)
t2 = np.cbrt((G2*M2*(t1**2))/(4*np.pi**2))

print(t2*1000, "alt")

# Example usage in the main loop:
inclination, distance_from_ecliptic = position_relative_to_ecliptic_coordinate_system(rocket_position, rocket_velocity)
print(f"Inclination: {inclination:.2f} degrees, Distance from ecliptic plane: {distance_from_ecliptic:.2f} km")

launching = launch_with_curved_trajectory(rocket_position, rocket_velocity)
position = np.array(rocket_position, dtype=np.float64)
exit_altitude = 6371000 + 100000  # Define exit altitude (Earth radius + 100 km)

altitudes = []
times = np.array([])  # Start with an empty NumPy array
# Later in the loop, append new times:

total_simulation_time = 5000
for step in range(int(total_simulation_time / time_interval)):
    time = step * time_interval  # Calculate the current time
    times = np.append(times, time)  # Append current time to the array

    # Simulate altitude change (replace this logic with your physics calculations)
    position = update_position1(position, velocity, time_interval)
    _, rocket_velocity, _, _, _ = runge_kutta_4(rocket_position, rocket_velocity, rocket_mass, fuel_mass, temp,
                                                pressure, dt, mach_number, rocket_velocity)
    rocket_velocity = np.array(rocket_velocity, dtype=float)
    print(rocket_velocity)
    new_altitude = altitude + step * dt

    print(new_altitude, "mww")
    print(f"Step {step}: Time {time} seconds, PositionAltitude: {altitude}, Velocity: {rocket_velocity}")

    # Exit condition: Check if altitude exceeds the threshold
    if altitude > exit_altitude:
        print(f"The rocket exits the atmosphere at {time} seconds.")
        break


print("Times array:", times)
print("Length of times array:", len(times))

# Debugging the time array
if times.size == 0:
    print("Error: The 'times' array is empty.")
else:
    if np.min(times) < 0:
        times = np.abs(times)  # Ensure positive values

# Example scaling, adjust as per your data
if np.max(times) > 1e6:  # If time is in microseconds
    times = times / 1e6  # Convert to seconds


def visualize_full_trajectory():
    """Visualize the full rocket trajectory from Earth to Mars."""
    earth_radius = 149.6e6  # Earth's orbit radius (km)
    mars_radius = 227.9e6  # Mars' orbit radius (km)
    orbit_radius = 1.1 * earth_radius  # Slightly larger than Earth's orbit radius

    # Generate trajectory segments
    launch_path = generate_launch_path(earth_radius * 0.01, earth_radius, num_points=100)
    earth_orbit = generate_circular_orbit(earth_radius, num_points=200)
    hohmann_transfer = generate_hohmann_transfer(orbit_radius, mars_radius, num_points = 500)
    # Concatenate trajectories for visualization
    trajectory = np.vstack((launch_path, hohmann_transfer))

    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ecliptic plane
    xx, yy = np.meshgrid(np.linspace(-1.5 * mars_radius, 1.5 * mars_radius, 20),
                         np.linspace(-1.5 * mars_radius, 1.5 * mars_radius, 20))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue', label='Ecliptic Plane')

    # Plot the Earth and Mars orbits
    ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], earth_orbit[:, 2], '--', color='blue', label='Earth Orbit')
    ax.plot(mars_radius * np.cos(np.linspace(0, 2 * np.pi, 200)),
            mars_radius * np.sin(np.linspace(0, 2 * np.pi, 200)),
            np.zeros(200), '--', color='red', label='Mars Orbit')

    # Plot Earth and Mars
    ax.scatter(0, 0, 0, color='blue', label='Earth', s=100)
    ax.scatter(mars_radius, 0, 0, color='red', label='Mars', s=100)

    # Plot the rocket's trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='green', label='Rocket Trajectory', linewidth=2)

    # Add labels and customize
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Rocket Trajectory with Ecliptic Plane and Hohmann Transfer')
    ax.legend()

    plt.show()

# Call the function to visualize
visualize_full_trajectory()


