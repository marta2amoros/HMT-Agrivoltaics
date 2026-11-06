# Download libraries!!
from thermo import Mixture
import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.HumidAirProp import HAPropsSI
import pandas as pd
import pvlib
import functions_module as func
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat
from tmm import coh_tmm, unpolarized_RT


# MODULE PROPERTIES: OPTICAL / HEAT
def Dh(module_width, module_length):
    # calculate the hydraulic diameter of PVmodule
    return 2 * (module_width * module_length) / (module_width + module_length)


def read_spectrum(source):
    """Spectral property from a file (.csv, .xlsx, .txt or .mat; first column wavelength [nm], second column value)
    or from a constant. Values given in % are converted to [-]. Returns (wavelength [nm], value [-])."""
    if isinstance(source, (int, float)):
        return np.array([0.0, 1e6]), np.full(2, float(source))  # constant over the whole spectrum
    path = str(source)
    if path.endswith(".mat"):
        data = loadmat(path)
        arr = next(v for k, v in data.items() if not k.startswith("__"))
        wavelength, value = arr[:, 0], arr[:, 1]
    else:
        if path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
        # wavelength column (named "Wavelength..." or the first one), value in the next column
        names = [str(c).lower() for c in df.columns]
        i = next((n for n, c in enumerate(names) if "wavelength" in c), 0)
        wavelength, value = df.iloc[:, i].values, df.iloc[:, i + 1].values
    value = np.asarray(value, dtype=float)
    if np.nanmax(value) > 1.5:  # given in %
        value = value / 100
    return np.asarray(wavelength, dtype=float), value


def thermal_mass_PV(layers_data, packing_factor):
    # Ensure that the input data has the correct format
    if not all(len(layer) == 4 for layer in layers_data):
        raise ValueError(
            "Each layer data should contain exactly four values: (Cp, density, thickness, conductivity)."
        )

    modified_layers = layers_data.copy()  # Create a copy of layers_data to modify

    Cp_Si, density_Si, thickness_Si, conductivity_Si = layers_data[3]
    Cp_glass, density_glass, _, conductivity_glass = layers_data[0]

    # Linearly average the properties using the packing factor
    Cp_new = Cp_Si * packing_factor + Cp_glass * (1 - packing_factor)
    density_new = density_Si * packing_factor + density_glass * (1 - packing_factor)
    conductivity_new = conductivity_Si * packing_factor + conductivity_glass * (
        1 - packing_factor
    )

    # Update the modified layers list
    modified_layers[3] = (Cp_new, density_new, thickness_Si, conductivity_new)

    # Compute the total thermal mass per unit area
    Cap = sum(
        Cp * density * thickness
        for Cp, density, thickness, conductivity in modified_layers
    )

    return Cap


def match_spectral_resolution(
    wl_n, n_vals, wl_k, k_vals, wl_common=None
):  # Interpolate n and k values to a common wavelength grid.
    if wl_common is None:
        # Create common wavelength grid from the overlapping range
        wl_min = max(np.min(wl_n), np.min(wl_k))
        wl_max = min(np.max(wl_n), np.max(wl_k))
        wl_max = min(wl_max, 100)  # limit to thermal radiation range
        wl_interp = np.linspace(wl_min, wl_max, 250)
    else:
        wl_interp = wl_common

    # Interpolate both n and k to this common grid
    n_interp_func = interp1d(
        wl_n, n_vals, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    k_interp_func = interp1d(
        wl_k, k_vals, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    n_interp = n_interp_func(wl_interp)
    k_interp = k_interp_func(wl_interp)

    return wl_interp, n_interp, k_interp


def resample_to_common_grid(wl_target, wl_source, values):
    interp_func = interp1d(
        wl_source, values, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return interp_func(wl_target)


def alpha_cell_calc(
    wavelength_irradiance,
    G_total_spectral,
    E_g,
    spectral_response="Optical/Mono-cSi_response.xlsx",
    cell_reflectance="Optical/PVlaminate.mat",
):
    # spectral_response, cell_reflectance: file path or constant (see read_spectrum)
    wavelength_PV, SR_PV = read_spectrum(
        spectral_response
    )  # fraction of incident light converted to electricity --> 1-SR = fraction of light reflected/heat

    # Extend SR from 300 nm to 100000 nm (0.3 µm to 100 µm)
    wavelength_PV_extended = np.linspace(300, 100000, 10000)
    SR_ext = np.interp(
        wavelength_PV_extended, wavelength_PV, SR_PV
    )  # Extend wavelength and interpolate SR
    SR_ext[wavelength_PV_extended > 1100] = 0  # No electrical conversion above bandgap

    # Interpolate SR to actual spectral wavelengths
    SR = np.interp(wavelength_irradiance, wavelength_PV_extended, SR_ext)
    SR[wavelength_irradiance > 1100] = 0

    valid_mask = (
        (wavelength_irradiance > 0) & ~np.isnan(SR) & ~np.isnan(G_total_spectral)
    )
    wavelength_irradiance = wavelength_irradiance[valid_mask]
    SR = SR[valid_mask]
    G_total_spectral = G_total_spectral[valid_mask]

    # Cell reflectance:
    # Load reflectance data
    wavelength_R_data, R_cell_data = read_spectrum(cell_reflectance)  # [nm], [-]
    R_cell_spectral = np.interp(wavelength_irradiance, wavelength_R_data, R_cell_data)
    # Compute photon energy
    # E_g = 1.12 [eV] Bandgap energy of the silicon cell
    h = 6.62607015e-34  # [J·s] Planck's constant
    c = 2.99792458e8  # [m/s] Speed of light
    q = 1.602176634e-19  # [C] Elementary charge
    E_ph = (
        h * c / (q * 10 ** (-9) * wavelength_irradiance)
    )  # [eV] Energy of the photons in the spectrum

    absorption_cell_spectral = (1 - SR * (E_g / E_ph)) * (
        1 - R_cell_spectral
    )  # [-] Absorption of the cell, 1-SR = fraction of light reflected/heat SR*(E_g/E_ph)~0.2

    q_cell_thermal_lambda = (
        absorption_cell_spectral * G_total_spectral
    )  # spectral thermal absorption (W/m²/nm)
    energy_to_thermal = np.trapezoid(
        q_cell_thermal_lambda, wavelength_irradiance
    )  # total energy to heat (W/m²)
    alpha_ct = energy_to_thermal / np.trapezoid(
        G_total_spectral, wavelength_irradiance
    )  # effective alpha_ct for the cell (BEFORE absorption_cell = 0.83)

    print(f"Energy to heat on cell [W/m²]: {energy_to_thermal:.2f}")
    # print("Effective absorption coefficient (alpha_ct):", alpha_ct)

    return (
        absorption_cell_spectral,
        alpha_ct,
        SR,
        wavelength_irradiance,
        R_cell_spectral,
    )


def alpha_glass_spectral_calc_dynamic(
    n_glass_spectral, k_glass_spectral, total_glass_thickness, wl_um, AOI_front
):
    """
    Parameters:
    - n_glass_spectral: (Nλ,) array of refractive indices
    - k_glass_spectral: (Nλ,) array of extinction coefficients
    - total_glass_thickness: scalar in meters (e.g., 0.0032)
    - wl_um: (Nλ,) array of wavelengths in microns
    - AOI_front_series_deg: (8760,) array of angles of incidence in degrees

    Returns:
    - alpha_glass_2D: (8760, Nλ) spectral absorptance
    - T_glass_2D: (8760, Nλ) spectral transmittance
    - R_glass_2D: (8760, Nλ) spectral reflectance
    """
    AOI_rad = np.radians(np.array(AOI_front))[
        :, np.newaxis
    ]  # AOI in radians, shape (8760, 1)
    wl_m = wl_um * 1e-6  # Convert µm to meters
    n1 = 1.0  # air

    N_time = len(AOI_front)
    N_lambda = len(wl_um)

    # Broadcast n and k to shape (1, Nλ)
    n2 = np.array(n_glass_spectral)[np.newaxis, :]
    k2 = np.array(k_glass_spectral)[np.newaxis, :]
    n_complex = n2 + 1j * k2

    # Snell’s Law: theta_t  [62. EQ 4.11]
    sin_theta_t = (n1 / n2) * np.sin(AOI_rad)
    sin_theta_t = np.clip(
        sin_theta_t, -1, 1
    )  #  (Snell's law consequence at which a critical angle doesn't allow light to exit the glass)
    theta_t = np.arcsin(sin_theta_t)

    # Fresnel reflectance: rs, rp   [62. EQ 4.12/13]
    rs = (n1 * np.cos(AOI_rad) - n_complex * np.cos(theta_t)) / (
        n1 * np.cos(AOI_rad) + n_complex * np.cos(theta_t)
    )
    rp = (n1 * np.cos(theta_t) - n_complex * np.cos(AOI_rad)) / (
        n1 * np.cos(theta_t) + n_complex * np.cos(AOI_rad)
    )

    # GLASS REFLECTANCE [62. EQ 4.19]
    R_glass_2D = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)

    # Spectral transmittance: T_glass Lambert law [62. eq4.25]  +  abs.coeff(k) [62. eq4.26]
    wl_m_2D = wl_m[np.newaxis, :]  # (1, Nλ)
    T_glass_2D = (1 - R_glass_2D) * np.exp(
        -4 * np.pi * k2 * total_glass_thickness / wl_m_2D
    )

    # Total Internal Reflection (glass → air): apply critical angle cutoff [62. eq 4.19]
    theta_crit = np.arcsin(np.minimum(n1 / n2, 1.0))  # (1, Nλ)
    TIR_mask = AOI_rad > theta_crit  # shape (8760, Nλ)
    T_glass_2D[TIR_mask] = 0
    R_glass_2D[TIR_mask] = 1

    # Scale to given glass:
    R_glass_2D_copy = R_glass_2D
    T_glass_2D_copy = T_glass_2D
    T_glass_2D = R_glass_2D_copy
    R_glass_2D = T_glass_2D_copy

    # Spectral absorptance: [BOOK B]
    alpha_glass_2D = np.maximum(0, 1 - R_glass_2D - T_glass_2D)

    return alpha_glass_2D, T_glass_2D, R_glass_2D


def alpha_ARCglass_spectral_calc_dynamic(
    wl_um, AOI_front_deg, d_ARC_um, n_ARC, n_glass_spectral, k_glass
):
    """
    wl_um          : (Nλ,) wavelengths in microns
    AOI_front_deg  : (Nt,) angles of incidence in degrees
    d_ARC_um       : scalar, thickness of ARC layer (e.g., 0.1 µm = 100 nm)
    n_ARC_array    : (Nλ,) SiO₂ refractive index
    n_glass_array  : (Nλ,) soda-lime real refractive index
    k_glass_array  : (Nλ,) soda-lime extinction coefficient
    """
    Nt, Nλ = len(AOI_front_deg), len(wl_um)
    R_2D = np.zeros((Nt, Nλ))
    T_2D = np.zeros((Nt, Nλ))

    for i, angle in enumerate(AOI_front_deg):
        AOI_front_rad = np.radians(angle)  # Convert to radians
        for j, wl in enumerate(wl_um):
            n_air = 1.0
            n_ARC_j = n_ARC[j]  # don't overwrite the array
            n_glass_complex = n_glass_spectral[j] + 1j * k_glass[j]

            n_list = [n_air, n_ARC_j, n_glass_complex]
            d_list = [np.inf, d_ARC_um, np.inf]

            Rs = coh_tmm("s", n_list, d_list, AOI_front_rad, wl)
            Rp = coh_tmm("p", n_list, d_list, AOI_front_rad, wl)

            R_2D[i, j] = 0.5 * (Rs["R"] + Rp["R"])
            T_2D[i, j] = 0.5 * (Rs["T"] + Rp["T"])

    alpha_2D = np.maximum(0, 1 - R_2D - T_2D)
    return alpha_2D, T_2D, R_2D


def absorption_avg(absorption_glass, absorption_cell, packing_factor):
    absorption = absorption_cell * packing_factor + absorption_glass * (
        1 - packing_factor
    )
    return absorption


def emissivity_T_calc(wl_um, alpha_spectral, T):
    wavelength_m = wl_um * 1e-6  # Convert wavelength from microns to meters
    h = 6.62607015e-34  # Planck [J.s]
    c = 2.99792458e8  # speed of light [m/s]
    kB = 1.380649e-23  # Boltzmann [J/K]
    B_lambda = (
        (2 * h * c**2)
        / (wavelength_m**5)
        / (np.exp(h * c / (wavelength_m * kB * T)) - 1)
    )  # Planck's weightedlaw B(λ,T)  [33. pg737, eq.12.23]
    # B = np.mean(B_lambda) # average Planck's law for the whole spectrum
    numerator = np.trapezoid(alpha_spectral * B_lambda, wavelength_m)
    denominator = np.trapezoid(B_lambda, wavelength_m)
    emissivity = (
        numerator / denominator
    )  # Emissivity of a real surface [33. ch12. eq12.36]

    return emissivity


def emissivity(
    alpha_glass_spectral,
    wl_glass_um,
    absorption_cell_spectral,
    wl_cell_nm,
    packing_factor,
    Tm,
):
    # Emissivity
    wl_cell_um = wl_cell_nm * 1e-3  # Convert wavelength from nm to µm
    emissivity_glass = emissivity_T_calc(wl_glass_um, alpha_glass_spectral, Tm)
    emissivity_cell = emissivity_T_calc(wl_cell_um, absorption_cell_spectral, Tm)
    emissivity = (
        packing_factor * emissivity_cell + (1 - packing_factor) * emissivity_glass
    )

    return emissivity, emissivity_glass, emissivity_cell


# AIR PROPERTIES
def calc_air_properties(temperature: float, RH: float, pressure: float):
    """
    Parameters:
    - temperature (float) [K]
    - pressure (float) [Pa]. Default is 101325 Pa (1 atm).

    Library: thermo and CoolProp (I checked values with: basic heat and mass transfer A.F Mills and CFM Coimbra)

    Returns: AIR dynamic viscosity (Pa·s), density (kg/m³), kinematic viscosity (m²/s), Specific heat at constant pressure (J/kg·K), thermal conductivity (W/m·K), and Thermal diffusivity [m²/s], Dew Point [K].
    """
    # Air composition: 78% Nitrogen, 21% Oxygen, 1% Argon
    air = Mixture(
        ["nitrogen", "oxygen", "argon"],
        ws=[0.78, 0.21, 0.01],
        T=temperature,
        P=pressure,
    )
    dyn_visc = air.mu  # Dynamic viscosity, Pa·s
    density = air.rho  # Density, kg/m³
    kin_visc = dyn_visc / density  # Kinematic viscosity, m²/s
    Cp = air.Cp  # Specific heat at constant pressure (J/kg·K)
    fluid = "Air"
    thermal_conductivity = CP.PropsSI(
        "CONDUCTIVITY", "T", temperature, "P", pressure, fluid
    )  # thermal conductivity of air at the specified T,P (W/m·K)
    alpha = thermal_conductivity / (
        density * Cp
    )  # Thermal diffusivity [m²/s] (conduct thermal energy relative to its ability to store thermal energy) [https://ostad.nit.ac.ir/payaidea/ospic/file8487.pdf]
    T_DP = HAPropsSI("D", "T", temperature, "RH", RH, "P", pressure)

    # Return results as a dictionary
    return dyn_visc, density, kin_visc, Cp, thermal_conductivity, alpha, T_DP


def wind_speed_calc(
    wind_speed_measured,
    wind_speed_direction,
    module_height,
    wind_height,
    field_orientation,
):
    wind_speed_measured_day = np.mean(
        wind_speed_measured.reshape(-1, 24), axis=1
    )  # mean measured wind velocity of the day
    wind_speed_orientation_correction = np.zeros_like(
        wind_speed_measured
    )  # wind speed correction for the orientation of the field rows [m/s]
    for i in range(
        len(wind_speed_measured)
    ):  # Apply orientation correction using cosine projection
        if np.isnan(wind_speed_direction[i]):
            wind_speed_orientation_correction[i] = wind_speed_measured[i]
        elif wind_speed_direction[i] == 0:
            wind_speed_orientation_correction[i] = wind_speed_measured[i]
        else:
            theta_diff = np.radians(wind_speed_direction[i] - field_orientation)
            wind_speed_orientation_correction[i] = wind_speed_measured[i] * np.abs(
                np.cos(theta_diff)
            )

    wind_speed_2m = wind_speed_orientation_correction * (
        4.87 / np.log(67.8 * wind_height - 5.42)
    )  # Wind speed at 2m above ground surface: log profile from [26,FAO] and ASCE
    wind_speed_2m_day = np.mean(
        wind_speed_2m.reshape(-1, 24), axis=1
    )  # mean wind velocity of the day at 2m above ground surface
    wind_speed_module = (
        wind_speed_orientation_correction * (module_height / wind_height) ** 0.2
    )  # Using power law from Fuentes approach
    wind_speed_module_day = np.mean(
        wind_speed_module.reshape(-1, 24), axis=1
    )  # mean modules' wind velocity of the day

    return (
        wind_speed_module,
        wind_speed_module_day,
        wind_speed_2m,
        wind_speed_2m_day,
        wind_speed_orientation_correction,
        wind_speed_measured_day,
    )


# Mass Balance: RH


def RH_bot_calc(m_air, Tamb_bot, ET_kgs, pressure_kPa, RH_in):
    m_air = np.asarray(m_air)
    Tamb_bot = np.asarray(Tamb_bot)
    ET_kgs = np.asarray(ET_kgs)
    pressure_kPa = np.asarray(pressure_kPa)
    RH_in = np.asarray(RH_in)

    e_sat = np.where(
        Tamb_bot > 0,
        0.611 * 10 ** (7.45 * Tamb_bot / (237.3 + Tamb_bot)),
        0.611 * 10 ** (9.5 * Tamb_bot / (265.5 + Tamb_bot)),
    )  # saturation vapor pressure [kPa]
    e_a = RH_in * e_sat

    w_in = 0.622 * e_a / (pressure_kPa - e_a)  # Mixing ratio in [kgH20/kgDRYair] (1)
    m_w = ET_kgs  # Water mass [kg]
    w_out = (m_air * w_in + m_w) / m_air  # Mixing ratio out [kgH20/kgDRYair] (2)

    e_a_out = (pressure_kPa * w_out) / (0.622 + w_out)  # Actual vapour pressure [kPa]
    RH_out = e_a_out / e_sat  # Relative humidity out [kPa] (3)
    RH_out = np.clip(RH_out, 0, 1)
    return RH_out, w_out


# CONVECTION
def calc_Plug_flow_model(
    wind_speed, thermal_conductivity, diffusivity, x, kinematic_viscosity
):  # book pg 403
    hconv_x = (thermal_conductivity / np.pi**0.5) * (
        wind_speed / (diffusivity * x)
    ) ** (1 / 2)
    Nu_x = hconv_x * x / thermal_conductivity
    # Nu_x = 0.564 * Re_x**0.5 * Pr**0.5  # Both methods work
    Re_x = x * wind_speed / kinematic_viscosity
    Pr = kinematic_viscosity / diffusivity
    Pe_x = Re_x * Pr
    if Pe_x < 10**3:
        return "Bad method, Pe too low"
    else:
        return hconv_x, Nu_x, Pe_x


def calc_BL_top(
    wind_speed,
    thermal_conductivity,
    diffusivity,
    kinematic_viscosity,
    tmod,
    tamb,
    module_length,
    module_width,
    tilt_angle,
):  # All references published in Table 1 of the manuscript
    "FORCED CONVECTION average along the PV module"
    L = Dh(module_width, module_length)
    Re_L = L * wind_speed / kinematic_viscosity
    Pr = kinematic_viscosity / diffusivity
    Re_tr = 5e05  # BookF[33]
    x_tr = Re_tr * kinematic_viscosity / wind_speed
    # print('Pr:', Pr, 'Re_L:', Re_L, 'Re_tr:', Re_tr, 'x_tr:', x_tr, 'L', L, 'wind_speed:', wind_speed, 'tilt_angle:', tilt_angle)

    if x_tr > L:  # Laminar Flow
        if Pr > 0.5:
            Nu_forced = 0.664 * Re_L**0.5 * Pr ** (1 / 3)
            h_forced = Nu_forced * thermal_conductivity / L
            pass
        else:
            print("Laminar Flow, Error model: Pr < 0.5")

    if x_tr < L:  # Turbulent Flow
        if 0.6 < Pr < 60:
            if Re_L >= 1e08:
                Nu_forced = 0.0296 * Re_L ** (4 / 5) * Pr ** (1 / 3)  # [33]
                h_forced = Nu_forced * thermal_conductivity / L

    if x_tr < L:  # Transient Flow (Laminar + turbulent)
        if 0.6 < Pr < 60:
            if Re_L < 1e08:
                Nu_forced = 0.664 * Re_L**0.5 * Pr ** (
                    1 / 3
                ) + 0.036 * Re_L**0.8 * Pr**0.33 * (
                    1 - (Re_tr / Re_L) ** 0.8
                )  # laminar + turbulent component Rearranged [33]

                h_forced = Nu_forced * thermal_conductivity / L
            else:
                print("Laminar + turbulent flow, Error model: RE_L>3e07")
        else:
            print("Laminar + turbulent flow, Error model: Pr < 0.7 or Pr > 400")

    # flow along the rows is parallel to the module plane, so inclined-plate correlations are not applied
    tilt_angle_perp = 0
    # Account for tilt for particular empirical studies: 
    
    if 0 < tilt_angle_perp <= 20 and 5.2e4 < Re_L < 1.71e5:  # [40] error max.dev 5%
        Nu_forced = 0.325 * Re_L**0.6255 * (1 + np.sin(np.radians(tilt_angle))) ** 0.5
        h_forced = Nu_forced * thermal_conductivity / L
    if (
        tilt_angle_perp > 20
    ):  
        if tilt_angle_perp <= 45:
            p1 = -0.26 * np.radians(tilt_angle_perp) + 3.65
            p2 = 2.18 * np.radians(tilt_angle_perp) + 13.07
        elif tilt_angle_perp > 45:
            p1 = 0.85 * np.radians(tilt_angle_perp) + 2.71
            p2 = 0.47 * np.radians(tilt_angle_perp) ** 3.9 + 14.39
        h_forced = p1 * wind_speed + p2
        if 1 >= wind_speed or wind_speed >= 8:
            h_forced = p2  # Assume no wind dependance
            # print('warning: wind speed over limits')
        Nu_forced = h_forced * L / thermal_conductivity

    "  FREE CONVECTION average along the PV module  "
    tave = (tmod + tamb) / 2  # film temperature for overall convective coefficient
    g = 9.81  # gravity [m/s2]
    beta = 1 / tave  # volumatric air expansion [1/K]
    Gr = (
        g
        * np.sin(np.radians(tilt_angle))
        * np.abs(tmod - tamb)
        * beta
        * L**3
        / kinematic_viscosity**2
    )


    Ra = Gr * Pr
    if tilt_angle < 30:
        if np.abs(Gr / Re_L**2) < 0.1:
            h_free = 0  # h_free <<< h_forced --> hconv = h_forced  
        else:  # paper [39]
            if Gr == 0:
                Nu_free = 0
            if Gr < 0:  # tmod < tamb --> the heat goes to the module
                Nu_free = 0.405 * (np.abs(Gr) * np.cos(np.radians(tilt_angle))) ** (
                    1 / 4
                )
                h_free = -Nu_free * thermal_conductivity / L
            else:  # tmod > tamb --> the heat goes out of the module
                Nu_free = 0.405 * (Gr * np.cos(np.radians(tilt_angle))) ** (1 / 4)
                h_free = Nu_free * thermal_conductivity / L

    else:  # ONLY FOR TILT < 60º wrt vertical
        if np.abs(Ra) <= 1e09:  # laminar 
            if Gr == 0:
                Nu_free = 0
                h_free = 0
            else:  # tmod > tamb --> the heat goes out of the module
                Nu_free = 0.68 + 0.670 * Ra ** (1 / 4) / (
                    1 + (0.492 / Pr) ** (9 / 16)
                ) ** (
                    4 / 9
                )  # [33 eq 9.27]
                h_free = Nu_free * thermal_conductivity / L

        else:  # turbulent
            if Gr == 0:
                Nu_free = 0
                h_free = 0
            else:
                Nu_free = (
                    0.825
                    + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)
                ) ** 2  
                h_free = Nu_free * thermal_conductivity / L

    # Top free horizontal convection:
    if tilt_angle == 0 and tmod > tamb:  # [33] fig 9.6
        if Ra < 1e07:
            Nu_free = 0.54 * Ra**0.25  # [33] 9.3
        else:
            Nu_free = 0.15 * Ra ** (1 / 3)  # [33] 9.31
        h_free = Nu_free * thermal_conductivity / L

    if tilt_angle == 0 and tmod < tamb:  # [33] fig 9.6
        Nu_free = 0.27 * Ra**0.25  # [33] 9.32
        h_free = Nu_free * thermal_conductivity / L

    # Convection coeff
    if np.isnan(h_free):
        h_free = 0

    if tmod - tamb < 0:  # tmod < tamb --> the heat goes to the module
        if h_forced**3 - np.abs(h_free) ** 3 < 0:
            hconv_top = -((np.abs(h_forced**3 - np.abs(h_free) ** 3)) ** (1 / 3))
        else:
            hconv_top = (h_forced**3 - np.abs(h_free) ** 3) ** (1 / 3)  # W/m2·K

    else:  # tmod > tamb --> the heat goes out of the module
        hconv_top = (h_forced**3 + np.abs(h_free) ** 3) ** (1 / 3)  # W/m2·K

    return Nu_forced, hconv_top, h_forced, h_free, Gr


def calc_BL_bot(
    wind_speed,
    thermal_conductivity,
    diffusivity,
    kinematic_viscosity,
    tmod,
    tamb,
    module_length,
    module_width,
    tilt_angle,
): 
    "FORCED CONVECTION average along the PV module"
    L = Dh(module_width, module_length)
    Re_L = L * wind_speed / kinematic_viscosity
    Pr = kinematic_viscosity / diffusivity
    Re_tr = 5e05  # BookF[33]
    x_tr = Re_tr * kinematic_viscosity / wind_speed

    if x_tr > L:  # Laminar Flow
        if Pr > 0.5:
            Nu_forced = 0.664 * Re_L**0.5 * Pr ** (1 / 3)
            h_forced = Nu_forced * thermal_conductivity / L
            # print('Laminar flow')
        else:
            print("Laminar Flow, Error model: Pr < 0.5")

    if x_tr < L:  # Turbulent Flow
        if 0.6 < Pr < 60:
            if Re_L >= 1e08:
                Nu_forced = 0.0296 * Re_L ** (4 / 5) * Pr ** (1 / 3)  # [33]
                h_forced = Nu_forced * thermal_conductivity / L

    if x_tr < L:  # Transient Flow (Laminar + turbulent)
        # (and 1.5*L < x_tr:)
        if 0.6 < Pr < 60:
            if Re_L < 1e08:
                Nu_forced = 0.664 * Re_L**0.5 * Pr ** (
                    1 / 3
                ) + 0.036 * Re_L**0.8 * Pr**0.33 * (
                    1 - (Re_tr / Re_L) ** 0.8
                )  # laminar + turbulent component Rearranged [33]
                # A = 0.037*Re_tr**(4/5) - 0.664*Re_tr**(1/2)
                # Nu_forced = (0.037*Re_L**(4/5) - A)*Pr**(1/3) # laminar + turbulent component [33]
                h_forced = Nu_forced * thermal_conductivity / L
            else:
                print("Laminar + turbulent flow, Error model: RE_L>3e07")
        else:
            print("Laminar + turbulent flow, Error model: Pr < 0.7 or Pr > 400")


    tilt_angle_perp = 0  # see calc_BL_top
    if 0 < tilt_angle_perp <= 20 and 5.2e4 < Re_L < 1.71e5:  
        Nu_forced = 0.325 * Re_L**0.6255 * (1 + np.sin(np.radians(tilt_angle))) ** 0.5
        h_forced = Nu_forced * thermal_conductivity / L

    if (
        tilt_angle_perp > 20
    ): 
        if tilt_angle_perp <= 45:
            p1 = -0.26 * np.radians(tilt_angle_perp) + 3.65
            p2 = 2.18 * np.radians(tilt_angle_perp) + 13.07
        elif tilt_angle_perp > 45:
            p1 = 0.85 * np.radians(tilt_angle_perp) + 2.71
            p2 = 0.47 * np.radians(tilt_angle_perp) ** 3.9 + 14.39
        h_forced = p1 * wind_speed + p2
        if 1 >= wind_speed or wind_speed >= 8:
            h_forced = p2  # Assume no wind dependence
        Nu_forced = h_forced * L / thermal_conductivity

    "  FREE CONVECTION average along the PV module  "
    tave = (tmod + tamb) / 2  # film temperature for overall convective coefficient
    g = 9.81  # gravity [m/s2]
    beta = 1 / tave  # volumatric air expansion [1/K]
    Gr = (
        g
        * np.sin(np.radians(tilt_angle))
        * np.abs(tmod - tamb)
        * beta
        * L**3
        / kinematic_viscosity**2
    )

    Ra = Gr * Pr
    if tilt_angle < 30:
        if np.abs(Gr / Re_L**2) < 0.1:
            h_free = 0  # h_free <<< h_forced --> hconv = h_forced 
        else:  # paper [39]
            if Gr == 0:
                Nu_free = 0
            if Gr < 0:  # tmod < tamb --> the heat goes to the module
                Nu_free = 0.405 * (np.abs(Gr) * np.cos(np.radians(tilt_angle))) ** (
                    1 / 4
                )
                h_free = -Nu_free * thermal_conductivity / L
            else:  # tmod > tamb --> the heat goes out of the module
                Nu_free = 0.405 * (Gr * np.cos(np.radians(tilt_angle))) ** (1 / 4)
                h_free = Nu_free * thermal_conductivity / L

    else:  # ONLY FOR TILT < 60º wrt vertical
        if np.abs(Ra) <= 1e09:  # laminar 
            if Gr == 0:
                Nu_free = 0
                h_free = 0
            else:  # tmod > tamb --> the heat goes out of the module
                Nu_free = 0.68 + 0.670 * Ra ** (1 / 4) / (
                    1 + (0.492 / Pr) ** (9 / 16)
                ) ** (
                    4 / 9
                )  
                h_free = Nu_free * thermal_conductivity / L

        else:  # turbulent
            if Gr == 0:
                Nu_free = 0
                h_free = 0
            else:
                Nu_free = (
                    0.825
                    + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)
                ) ** 2  
                h_free = Nu_free * thermal_conductivity / L

    # Bottom free horizontal convection:
    if tilt_angle == 0 and tmod < tamb:  # [33] fig 9.6
        if Ra < 1e07:
            Nu_free = 0.54 * Ra**0.25  # [33] 9.3
        else:
            Nu_free = 0.15 * Ra ** (1 / 3)  # [33] 9.31
        h_free = Nu_free * thermal_conductivity / L
    if tilt_angle == 0 and tmod > tamb:  # [33] fig 9.6
        Nu_free = 0.27 * Ra**0.25  # [33] 9.32
        h_free = Nu_free * thermal_conductivity / L

    # Convection coeff
    if np.isnan(h_free):
        h_free = 0

    if tmod - tamb < 0:  # tmod < tamb --> the heat goes to the module
        if h_forced**3 - np.abs(h_free) ** 3 < 0:
            hconv_bot = -((np.abs(h_forced**3 - np.abs(h_free) ** 3)) ** (1 / 3))
        else:
            hconv_bot = (h_forced**3 - np.abs(h_free) ** 3) ** (1 / 3)  # W/m2·K

    else:  # tmod > tamb --> the heat goes out of the module
        hconv_bot = (h_forced**3 + np.abs(h_free) ** 3) ** (1 / 3)  # W/m2·K

    return Nu_forced, hconv_bot, h_forced, h_free, Gr


# SKY TEMPERATURE
def calc_temperature_sky(Tair_K, dew_point, daytime):
    T_DP = dew_point - 273.15  # ºC

    if daytime == "day":
        e_sky = 0.727 + 0.0060 * T_DP  # DAYTIME 
    elif daytime == "night":
        e_sky = 0.741 + 0.0062 * T_DP  # NIGHTIME 

    T_sky_K = Tair_K * e_sky ** (1 / 4)  
    T_sky_clouds_K = (
        0.6 * T_sky_K + 0.4 * Tair_K
    )  # cloud coverage avg EU 1991-2020 copernicus.eu
    return T_sky_clouds_K


# GROUND TEMPERATURE
def Tground_calc(Rn, Gn, L_ET, pressure, Cp, Tamb, RH):  # Gn input, ENERGY BALANCE
    # Model Parameters
    dt = 3600  # Time step (s) - 1 hour
    density_soil = 1700  # Soil density (kg/m3)                                                           
    Cp_soil = 2000  # Specific soil heat capacity (J/kg·K)
    Cv_s = (
        density_soil * Cp_soil
    )  # ~e6, Volumetric soil heat capacity (J/m3·K), typical for dry soil
    timesteps = 8760  # Simulation Time: 1-year (8760 hours)

    # Bowen Ratio parameters:
    gamma = (
        Cp * 10 ** (-6) * pressure * 10 ** (-3) / (0.622 * 2.45)
    )  # psychrometric constant [kPa °C-1]  ~0.066kPa/ºC
    e_sat = np.where(
        Tamb > 0,
        0.611 * 10 ** (7.45 * Tamb / (237.3 + Tamb)),
        0.611 * 10 ** (9.5 * Tamb / (265.5 + Tamb)),
    )  # saturation vapor pressure at air temperature [kPa] using Magnus formula with Tamb [ºC]
    e_a = e_sat * RH  # average hourly actual vapour pressure [kPa]
    VPD = e_sat - e_a

    # Initialize Sensible Heat Flux (H) and Bowen Ratio (B)
    B = np.full(timesteps, 0.2)  # Initial guess for Bowen Ratio
    H = B * L_ET  # Sensible heat flux (W/m²)
    H.index = pd.Index(np.arange(1, timesteps + 1), name="Time")
    L_ET.index = pd.Index(np.arange(1, timesteps + 1), name="Time")

    # Initialize surface ground temperature
    T_surface = pd.Series(
        np.zeros(timesteps), index=pd.Index(np.arange(timesteps), name="Time")
    )
    T_surface.iloc[0] = Tamb[0]  # Initial surface temperature (°C)

    # Time integration using Euler's method with Bowen Ratio Convergence Loop
    for t in range(1, timesteps):
        T_old = float(T_surface.iloc[t - 1])  # Ensure scalar
        tolerance = 0.001  # Convergence threshold
        max_iterations = 5  # Prevent infinite looping
        iteration = 0

        while iteration < max_iterations:
            # Extract scalars to avoid Pandas Series errors
            VPD_safe = max(
                float(VPD.iloc[t]), 0.0001
            )  # Prevent division by zero
            L_ET_t = float(L_ET.iloc[t])
            gamma_t = float(gamma[t])

            B_new = gamma_t * (T_old - Tamb[t]) / VPD_safe
            B_new = np.clip(B_new, 0.01, 5)  # Bowen ratio limited to 0.01-5, as in Tground_Gn_calc
            # B_new = min(B_new, 5)  # Constrain, Stability check

            H_new = float(B_new * L_ET_t)  # Compute new Sensible Heat Flux

            # Compute new surface temperature change
            dT_surface_dt = (Rn[t] - H_new - L_ET_t - Gn[t]) / Cv_s
            T_new = float(T_surface.iloc[t - 1]) + dT_surface_dt * dt  # Ensure scalar

            # Ensure `T_new` is a scalar before comparison
            if abs(T_new - T_old) < tolerance:
                break  # Stop iteration if convergence is achieved

            T_old = T_new  # Update for next iteration
            iteration += 1

        # Store final values for this timestep
        T_surface.iloc[t] = T_new
        B[t] = B_new
        H[t] = H_new

    return T_surface, H, B


def Tground_ForceRestore(L_ET, Tamb):
    # Model Parameters with Convection
    dz = 0.05  # Spatial step (m)
    dt = 3600  # Time step (s) - 1 hour
    # Soil parameters
    depth = 10  # Soil depth (m)
    # Moisture influence on thermal properties:
    thermal_diffusivity, Cv_s, lambda_soil = func.thermal_soil_properties(
        theta=0.18, soil_type="sandy loam"
    )
    timesteps = 8760  # 1-year simulation (8760 hours)

    # Force-Restore Analytical Solution Parameters
    T_mean = np.mean(Tamb)
    A = 5  # Amplitude of daily wave (°C)
    P = 24 * 3600  # Diurnal period (seconds)
    omega = 2 * np.pi / P
    d = np.sqrt(2 * thermal_diffusivity / omega)  # Damping depth [m]

    depths = np.arange(dz, depth, dz)
    T_surface = pd.Series(
        np.zeros(timesteps), index=pd.Index(np.arange(timesteps), name="Time")
    )
    T_subsurface = np.zeros((len(depths), timesteps))
    G_subsurface = np.zeros((len(depths), timesteps))
    Gn = np.zeros(timesteps)
    B = np.full(timesteps, 0.2)
    H = B * L_ET  # Not actively used
    H.index = pd.Index(np.arange(1, timesteps + 1), name="Time")
    L_ET.index = pd.Index(np.arange(1, timesteps + 1), name="Time")

    for t in range(1, timesteps):
        # Analytical surface temperature
        T_surface[t] = T_mean + A * np.sin(omega * t * dt)

        # Analytical subsurface temperature 
        for i, z in enumerate(depths):
            T_subsurface[i, t] = T_mean + A * np.exp(-z / d) * np.sin(
                omega * t * dt - z / d
            )

            # Ground heat flux 
            G_subsurface[i, t] = (
                lambda_soil
                / d
                * A
                * np.exp(-z / d)
                * (np.sin(omega * t * dt - z / d) + np.cos(omega * t * dt - z / d))
            )

        Gn[t] = G_subsurface[0, t]

    return T_surface, T_subsurface, G_subsurface, Gn, H, B


def Tground_empirical_calc(Tamb):  # EMPIRICAL wrt Tamb
    # Tground: Compute ground temperature
    ext_LAI = 3.03 * 2.46  # 1-sided leaf area index * extinction coefficient
    tground_array = np.zeros(len(Tamb))
    daily_tamb_values = []  # Initialize empty list to store daily temperatures
    Tair_K = Tamb + 273.15  # Convert to Kelvin
    # Initial values
    tground = Tair_K.iloc[0]  # Set initial ground temp to first air temp value
    tini_day = Tair_K.iloc[0]
    avg_tamb_day = Tair_K.iloc[0]

    for i in range(len(Tamb)):
        countday = i - 6  # Set reference time for daily reset
        daily_tamb_values.append(Tair_K.iloc[i])  # Collect the temperature

        if countday % 24 == 0 and i > 0:
            avg_tamb_day = np.mean(daily_tamb_values)  # Compute daily average temp
            daily_tamb_values = [
                Tair_K.iloc[i]
            ]  # Reset with the first temp of the new day
            tini_day = Tair_K[i]

        # Update ground temperature
        if tini_day > tground:
            tground = tground + (avg_tamb_day - tground) * 0.25 * np.exp(-ext_LAI)
        else:
            tground = tground + (avg_tamb_day - tground) * 0.25

        tground_array[i] = tground  # Store result

    return tground_array


def Tground_Gn_calc(
    Rn, L_ET, pressure, Cp, Tamb, RH, theta, soil_type
):  # NUMERICAL METHODS
    """
    Solves soil surface and subsurface temperature using 1D finite difference
    with energy balance at the surface and optional seasonal bottom BC.

    Includes moisture effects on λ and Cv using Johansen's method [88].

    Parameters:
        - Rn, L_ET, pressure, Cp, Tamb, RH: environmental/precalculated inputs
        - theta: volumetric moisture content [m³/m³]
        - soil_type: 'Sandy loam', 'Clay', or 'Silt'

    Returns:
        - T_surface, T_subsurface, G_subsurface, Gn, H, B, Cv_s, lambda_soil, thermal_diffusivity
    """

    # Model Parameters with Convection
    dz = 0.06  # Spatial step (m)
    dt = 3600  # Time step (s) - 1 hour
    # Soil parameters
    depth = 10  # Soil depth (m)

    # Moisture influence on thermal properties:
    thermal_diffusivity, Cv_s, lambda_soil = func.thermal_soil_properties(
        theta, soil_type
    )
    print(
        f"thermal_diffusivity = {thermal_diffusivity:.2e}, Cv_s = {Cv_s:.2f}, lambda_soil = {lambda_soil:.2f}"
    )
    stability_number = thermal_diffusivity * dt / dz**2
    # print('Stability check:', stability_number, '<0.5')
    if stability_number > 0.5:
        print("Scheme may be unstable, CFL =", round(stability_number, 3))

    timesteps = len(Tamb)  # number of hours of the simulation (8760 for one year)
    nz = int(depth / dz)  # Number of depth layers

    # Atmospheric variables
    gamma = (
        Cp * 10 ** (-6) * pressure * 10 ** (-3) / (0.622 * 2.45)
    )  # Psychrometric constant [kPa °C-1]
    e_sat = np.where(
        Tamb > 0,
        0.611 * 10 ** (7.45 * Tamb / (237.3 + Tamb)),
        0.611 * 10 ** (9.5 * Tamb / (265.5 + Tamb)),
    )
    e_a = e_sat * RH  # Actual vapour pressure [kPa]
    VPD = e_sat - e_a

    B = np.full(timesteps, 1.0)  # Initial Bowen Ratio (float: an int array truncated B)
    Hcb = B * L_ET  # Sensible heat flux (W/m²)
    Hcb.index = pd.Index(np.arange(1, timesteps + 1), name="Time")
    L_ET.index = pd.Index(np.arange(1, timesteps + 1), name="Time")

    T_surface = pd.Series(
        np.zeros(timesteps), index=pd.Index(np.arange(timesteps), name="Time")
    )
    T_surface.iloc[0] = Tamb[0]

    Gn = np.zeros(timesteps)  # Ground heat flux at surface
    G_ET = np.zeros(
        timesteps
    )  # heat flux (heat utilized in heating the soil) for ET calculation

    depths = np.arange(dz, depth, dz)
    T_subsurface = np.zeros((len(depths), timesteps))
    T_subsurface[:, 0] = np.mean(Tamb)
    G_subsurface = np.zeros((len(depths), timesteps))

    for t in range(1, timesteps):
        T_old = float(T_surface.iloc[t - 1])
        tolerance = 0.005
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            VPD_safe = max(float(VPD[t]), 0.0001)
            L_ET_t = float(L_ET.iloc[t])
            gamma_t = float(gamma[t])

            B_new = gamma_t * (T_old - Tamb[t]) / VPD_safe
            B_new = np.clip(B_new, 0.01, 5)
            Hcb_new = float(B_new * L_ET_t)

            # Explicit method:
            T_subsurface[0, t] = T_subsurface[
                0, t - 1
            ] + thermal_diffusivity * dt / dz**2 * (
                T_surface.iloc[t - 1] - T_subsurface[0, t - 1]
            )
            # Conduction Fourier equation: G(z,t) = -λ (∂T/∂z)
            Gn[t] = -lambda_soil * (T_subsurface[0, t] - T_surface.iloc[t - 1]) / dz

            # T_surface boundary conditions based on surface EB
            dT_surface_dt = (float(Rn.iloc[t]) - Hcb_new - L_ET_t - Gn[t]) / Cv_s
            T_new = float(T_surface.iloc[t - 1]) + dT_surface_dt * dt

            relaxation_factor = 0.2
            T_new = relaxation_factor * T_new + (1 - relaxation_factor) * T_old

            if abs(T_new - T_old) < tolerance:
                break

            T_old = T_new
            iteration += 1

        T_surface.iloc[t] = T_new
        G_ET[t] = float(Rn.iloc[t]) - Hcb_new - L_ET_t
        B[t] = B_new
        Hcb[t] = Hcb_new

        # Compute subsurface temperatures and heat flux based on new T_surface
        for i, z in enumerate(depths):
            if i == 0:
                # TOP boundary condition
                G_subsurface[i, t] = (
                    -lambda_soil * (T_surface.iloc[t] - T_subsurface[i, t - 1]) / z
                )
                continue

            if i < len(depths) - 1:
                # Explicit method:
                T_subsurface[i, t] = T_subsurface[
                    i, t - 1
                ] + thermal_diffusivity * dt / dz**2 * (
                    T_subsurface[i - 1, t - 1]
                    - 2 * T_subsurface[i, t - 1]
                    + T_subsurface[i + 1, t - 1]
                )
            else:
                # Bottom layer Boundary Condition (Stabilized temperature at Tamb,mean or + damping*sin() around annual mean -optional-)
                P = 365 * 24
                A = 0  # Amplitude of seasonal fluctuation at 15m (°C)   ~0-5ºC for annual soil fluctuations, A>0 for sinusoidal damping
                tpeak = 210 * 24  # hot peak
                phi = 2 * np.pi * tpeak / 8760  # phase shift
                omega = 2 * np.pi / P
                d = np.sqrt(
                    2 * thermal_diffusivity / omega
                )  # damping depth: thermal diffusivity for phase lag and depth attenuation
                z_bottom = depths[i]
                T_mean = np.mean(Tamb)

                T_subsurface[i, t] = T_mean + A * np.exp(-z_bottom / d) * np.sin(
                    omega * t - z_bottom / d + phi
                )  # [83], also follows [87] behaviour

            # Compute G(z,t) = -λ (∂T/∂z)
            G_subsurface[i, t] = (
                -lambda_soil * (T_subsurface[i, t] - T_subsurface[i - 1, t]) / dz
            )

    return {
        "T_surface": T_surface,
        "T_subsurface": T_subsurface,
        "G_subsurface": G_subsurface,
        "Gn": Gn,
        "G_ET": G_ET,
        "Hcb": Hcb,
        "B": B,
        "Cv": Cv_s,
        "lambda": lambda_soil,
        "alpha": thermal_diffusivity,
    }


def thermal_soil_properties(theta, soil_type):
    porosity = 0.5
    Sr = np.clip(theta / porosity, 0.01, 1)
    if soil_type.lower() in ["sandy loam", "sand", "loamy sand"]:  # Kersten number
        Ke = np.log10(Sr) + 1
    elif soil_type.lower() in ["clay", "silt", "silty clay", "loam"]:
        Ke = np.sqrt(Sr)
    else:
        raise ValueError("Unknown soil type")

    Cp_soil = 2000  # J/kgK  Dry soil
    density_soil = 1700  # kg/m3  Dry soil
    density_water = 1000  # kg/m3
    Cp_water = 4180  # J/kg·K
    # Thermal conductivity
    lambda_dry = 1.0  # W/m·K 
    lambda_wet = 2.0  # W/m·K 
    lambda_soil = (
        lambda_wet - lambda_dry
    ) * Ke + lambda_dry  # W/m·K 

    # Volumetric Capacity
    Cv_s = (
        1 - theta
    ) * density_soil * Cp_soil + theta * density_water * Cp_water  # [J/m³·K]
    thermal_diffusivity = lambda_soil / Cv_s  # [m2/s]

    return thermal_diffusivity, Cv_s, lambda_soil


# RADIATION


def AOIfront_calc(tilt_angle, module_azimuth, lat, lon, times):
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    SunZen_laimburg = solpos["apparent_zenith"]
    SunAz_laimburg = solpos["azimuth"]
    SunAz = SunAz_laimburg.reset_index(drop=True)  # Reset index to be integer-based
    SunZen = SunZen_laimburg.reset_index(drop=True)  # Reset index to be integer-based
    # print(solpos)
    AOI_front = pvlib.irradiance.aoi(tilt_angle, module_azimuth, SunZen, SunAz)
    AOI_front[AOI_front > 90] = 90
    return AOI_front, solpos


def radiation(
    Tair_K,
    DNI,
    DHI,
    GHI,
    albedo,
    tilt_angle,
    module_azimuth,
    surface,
    modules_density,
    packing_factor,
    tau_glass,
    lat,
    lon,
    times,
):
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    SunZen_calc = solpos["apparent_zenith"]
    SunAz_calc = solpos["azimuth"]
    SunAz = SunAz_calc.reset_index(drop=True)  # Reset index to be integer-based
    SunZen = SunZen_calc.reset_index(drop=True)  # Reset index to be integer-based
    # print(solpos)

    # Extraterrestrial radiation, DNI_extra: Sandia Lab [https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/irradiance-insolation/extraterrestrial-radiation/]
    hours_array = np.arange(1, len(Tair_K) + 1)
    Esc = 1367  # Solar constant in W/m^2
    b = 2 * np.pi * hours_array / 8760  # rad
    Rav_R_squared = (
        1.00011
        + 0.034221 * np.cos(b)
        + 0.00128 * np.sin(b)
        + 0.000719 * np.cos(2 * b)
        + 0.000077 * np.sin(2 * b)
    )  # Calculate (Rav/R)^2
    DNI_extra = (
        Esc * Rav_R_squared
    )  

    if surface == "module":
        # Front side
        AOI_front = pvlib.irradiance.aoi(tilt_angle, module_azimuth, SunZen, SunAz)
        AOI_front[AOI_front > 90] = 90
        SVF_front = 0.5 + np.cos(np.radians(tilt_angle)) / 2

        G_dir_front = DNI * np.cos(np.radians(AOI_front))
        G_diff_front = pvlib.irradiance.haydavies(
            tilt_angle, module_azimuth, DHI, DNI, DNI_extra, SunZen, SunAz
        )  # Improved :) before: G_diff_front = DHI * SVF_front
        G_refl_front = albedo * GHI * (1 - SVF_front)

        G_front = G_dir_front + G_diff_front + G_refl_front

        # Back side
        altitude_module_back = 90 - (90 + tilt_angle)  # 90-(180-(90-tilt_angle))
        azimuth_module_back = module_azimuth
        altitude_sun = 90 - SunZen

        temp = -np.cos(np.radians(SunZen)) * np.cos(np.radians(tilt_angle)) + np.sin(
            np.radians(tilt_angle)
        ) * np.sin(np.radians(SunZen)) * np.cos(
            np.radians(SunAz - module_azimuth + 180)
        )

        temp = np.clip(temp, -1, 1)
        AOI_back = np.degrees(np.arccos(temp))
        AOI_back[AOI_back > 90] = 90
        SVF_back = 1 - SVF_front  

        G_dir_back = DNI * np.cos(np.radians(AOI_back))
        G_diff_back = (
            DHI * SVF_back
        )  
        G_refl_back = (
            albedo * GHI * (1 - SVF_back) * 0.7
        ) 

        G_back = G_dir_back + G_diff_back + G_refl_back

        G_dir = G_dir_front + G_dir_back
        G_diff = G_diff_front + G_diff_back
        G_refl = G_refl_front + G_refl_back

        # Total irradiance
        G = G_front + G_back  # W/m2
        annual_G_back = np.sum(G_back) / 1000  # sum (G * 1h) =  kWh/m2
        annual_G_front = np.sum(G_front) / 1000  # sum (G * 1h) =  kWh/m2
        annual_irradiation = np.sum(G) / 1000  # sum (G * 1h) =  kWh/m2
        G_total = 0
        poa_global = G  # Irradiance [W/m2]

    if surface == "ground":
        tilt_angle = 0
        AOI = pvlib.irradiance.aoi(tilt_angle, module_azimuth, SunZen, SunAz)
        AOI[AOI > 90] = 90
        AOI_front = AOI
        SVF = 0.5 + np.cos(np.radians(tilt_angle)) / 2
        G_dir = DNI * np.cos(np.radians(AOI))
        G_diff = pvlib.irradiance.haydavies(
            tilt_angle, module_azimuth, DHI, DNI, DNI_extra, SunZen, SunAz
        )  
        G_refl = albedo * GHI * (1 - SVF)
        G_total = G_dir + G_diff + G_refl  # W/m2 Total irradianace on a bear ground
        poa_global = G_total  # Irradiance [W/m2]
        annual_irradiation = np.sum(G_total) / 1000  # sum (G * 1h) =  kWh/m2

    if surface == "grass":
        tilt_angle = 0
        AOI = pvlib.irradiance.aoi(tilt_angle, module_azimuth, SunZen, SunAz)
        AOI[AOI > 90] = 90
        AOI_front = AOI
        SVF = 0.5 + np.cos(np.radians(tilt_angle)) / 2

        G_dir = DNI * np.cos(np.radians(AOI))
        G_diff = pvlib.irradiance.haydavies(
            tilt_angle, module_azimuth, DHI, DNI, DNI_extra, SunZen, SunAz
        ) 
        G_refl = albedo * GHI * (1 - SVF)
        G_total = G_dir + G_diff + G_refl  # W/m2 Total irradianace on a bear ground
        G_grass = G_total * (
            (1 - modules_density) + modules_density * tau_glass * (1 - packing_factor)
        )
        # Total irradiance
        poa_global = G = G_grass  # Irradiance [W/m2]
        annual_irradiation = np.sum(G) / 1000  # sum (G * 1h) =  kWh/m2
        G_total = 0

    return poa_global, AOI_front, annual_irradiation, G_total, G_dir, G_diff, G_refl


def calc_albedo(
    crop_density,
    spectral_radiation_crop,
    spectral_radiation_grass,
    crop_reflectance="Optical/spectral_reflectance_apple.csv",
    grass_reflectance="Optical/spectral_reflectance_grass.xlsx",
):
    # crop_reflectance, grass_reflectance: file path or constant (see read_spectrum)
    # CROP REFLECTANCE (apple leaves by default), last value extended to 3000 nm
    wavelength_apple, R_apple = read_spectrum(crop_reflectance)
    if wavelength_apple[-1] < 3000:
        wavelength_ext = np.arange(wavelength_apple[-1] + 1, 3001, 1)
        R_ext = np.full_like(wavelength_ext, R_apple[-1], dtype=float)
        wavelength_apple = np.concatenate([wavelength_apple, wavelength_ext])
        R_apple = np.concatenate([R_apple, R_ext])

    # GRASS REFLECTANCE
    wavelength_grass, R_grass = read_spectrum(grass_reflectance)

    # SPECTRAL IRRADIANCE SCALING
    wavelength_irradiance = spectral_radiation_grass.index
    spectral_radiation_total = (
        spectral_radiation_crop * crop_density
        + spectral_radiation_grass * (1 - crop_density)
    )

    # Interpolate the spectral response and reflectance data to a common wavelength grid ---
    R_grass_interp = np.interp(wavelength_irradiance, wavelength_grass, R_grass)
    R_apple_interp = np.interp(wavelength_irradiance, wavelength_apple, R_apple)
    # Compute effective albedo
    numerator_grass = np.trapezoid(
        R_grass_interp * spectral_radiation_grass, wavelength_irradiance
    )
    numerator_apple = np.trapezoid(
        R_apple_interp * spectral_radiation_crop, wavelength_irradiance
    )
    denominator = np.trapezoid(spectral_radiation_total, wavelength_irradiance)
    albedo_grass = numerator_grass / denominator
    albedo_apple = numerator_apple / denominator
    albedo = crop_density * albedo_apple + (1 - crop_density) * albedo_grass
    albedo_spectral = (
        crop_density * R_apple_interp * spectral_radiation_crop
        + (1 - crop_density) * R_grass_interp * spectral_radiation_grass
    ) / spectral_radiation_total

    return albedo, albedo_spectral


def Radiation_extract_spectral_data(
    component_prefix, df_300_1200, df_1200_2800
):  # Extract spectral data from the raytracing radiation results
    # From 300-1200 nm (N1 to N9)
    cols_1 = [f"{component_prefix}_N{i}" for i in range(1, 10)]
    data_1 = df_300_1200[cols_1].mean()

    # From 1200-2800 nm (N10 to N25)
    cols_2 = [f"{component_prefix}_N{i}" for i in range(10, 26)]
    data_2 = df_1200_2800[cols_2].mean()

    # Combine and label by central wavelength of each band
    wavelengths = [
        350 + 100 * (i - 1) for i in range(1, 26)
    ]  # center of N1 = 350, ..., N25 = 2750
    spectral_values = pd.concat([data_1, data_2]).values

    return pd.Series(data=spectral_values, index=wavelengths, name=component_prefix)


def Radiation_extract_spectral_data_excl(
    component_prefix, df_300_1200_excl, df_1200_2800_excl
):  # Extract spectral data from the raytracing radiation results
    # From 300-1200 nm (N1 to N9)
    cols_1 = [f"{component_prefix}_N{i}" for i in range(1, 10)]
    data_1 = df_300_1200_excl[cols_1].mean()

    # From 1200-2800 nm (N10 to N25)
    cols_2 = [f"{component_prefix}_N{i}" for i in range(10, 26)]
    data_2 = df_1200_2800_excl[cols_2].mean()

    # Combine and label by central wavelength of each band
    wavelengths = [
        350 + 100 * (i - 1) for i in range(1, 26)
    ]  # center of N1 = 350, ..., N25 = 2750
    spectral_values = pd.concat([data_1, data_2]).values

    return pd.Series(data=spectral_values, index=wavelengths, name=component_prefix)


def Radiation_extract_time_series(component_prefix, df_300_1200, df_1200_2800):
    cols_300_1200 = [
        f"{component_prefix}_N{i}"
        for i in range(1, 10)
        if f"{component_prefix}_N{i}" in df_300_1200.columns
    ]
    cols_1200_2800 = [
        f"{component_prefix}_N{i}"
        for i in range(10, 26)
        if f"{component_prefix}_N{i}" in df_1200_2800.columns
    ]

    # Use Date as index
    df_300 = df_300_1200[["Date (MM/DD/YYYY)"] + cols_300_1200].copy()
    df_300.set_index(pd.to_datetime(df_300["Date (MM/DD/YYYY)"]), inplace=True)
    df_300.drop(columns=["Date (MM/DD/YYYY)"], inplace=True)

    df_1200 = df_1200_2800[["Date (MM/DD/YYYY)"] + cols_1200_2800].copy()
    df_1200.set_index(pd.to_datetime(df_1200["Date (MM/DD/YYYY)"]), inplace=True)
    df_1200.drop(columns=["Date (MM/DD/YYYY)"], inplace=True)

    # Combine both (horizontal concatenation)
    full_df = pd.concat([df_300, df_1200], axis=1)

    # Integrate over 100 nm per band
    return full_df.sum(axis=1)  # [W/m²] integrating over band


def Radiation_extract_time_series_excl(
    component_prefix, df_300_1200_excl, df_1200_2800_excl
):
    cols_300_1200 = [
        f"{component_prefix}_N{i}"
        for i in range(1, 10)
        if f"{component_prefix}_N{i}" in df_300_1200_excl.columns
    ]
    cols_1200_2800 = [
        f"{component_prefix}_N{i}"
        for i in range(10, 26)
        if f"{component_prefix}_N{i}" in df_1200_2800_excl.columns
    ]

    # Use Date as index
    df_300 = df_300_1200_excl[["Date (MM/DD/YYYY)"] + cols_300_1200].copy()
    df_300.set_index(pd.to_datetime(df_300["Date (MM/DD/YYYY)"]), inplace=True)
    df_300.drop(columns=["Date (MM/DD/YYYY)"], inplace=True)

    df_1200 = df_1200_2800_excl[["Date (MM/DD/YYYY)"] + cols_1200_2800].copy()
    df_1200.set_index(pd.to_datetime(df_1200["Date (MM/DD/YYYY)"]), inplace=True)
    df_1200.drop(columns=["Date (MM/DD/YYYY)"], inplace=True)

    # Combine both (horizontal concatenation)
    full_df = pd.concat([df_300, df_1200], axis=1)

    # Integrate over 100 nm per band
    return full_df.sum(axis=1)  # [W/m²] integrating over band


# MODULE TEMPERATURE SOLVER
def Tm_model(
    emissivity,
    tilt_angle,
    module_width,
    module_length,
    dew_point,
    num_modules,
    mc,
    poa_global,
    Tair_K,
    wind_speed,
    Tground_C,
    RH,
    pressure,
    times,
    ET,
    H,
    Kcb,
    Ke,
    density,
    kinematic_viscosity,
    thermal_conductivity,
    diffusivity,
    module_height,
    crop_height,
):
    """
    Compute PV module temperature dynamics over time for an array of modules.

    Parameters
    ----------
    emissivity : float
        Module longwave emissivity [-].
    tilt_angle : float
        PV module tilt angle [deg].
    module_width : float
        Module width [m].
    module_length : float
        Module length [m].
    dew_point : array-like
        Dew point temperature time series [K].
    num_modules : int
        Number of PV modules considered (in flow direction).
    mc : float
        Areal heat capacity of the module [J/(m²·K)].
    poa_global : array-like
        Plane-of-array irradiance [W/m²].
    Tair_K : array-like
        Ambient air temperature [K].
    wind_speed : array-like
        Wind speed at module height [m/s].
    Tground_C : array-like
        Ground temperature [°C].
    RH : array-like
        Relative humidity [%].
    pressure : array-like
        Atmospheric pressure [Pa].
    times : array-like
        Time index aligned with inputs.
    ET : array-like
        Crop evapotranspiration rate [mm/day], hourly values as returned by ET_func.Ke_ET_calc (dual crop coefficient).
    H : array-like
        Sensible heat flux released by the soil-crop system [W/m²] (Bowen ratio x latent heat flux).
    Kcb, Ke : array-like
        Basal crop and soil evaporation coefficients [-], used to place the heat release in the control volume.
    density, kinematic_viscosity, thermal_conductivity, diffusivity : array-like
        Ambient air properties [kg/m³], [m²/s], [W/(m·K)], [m²/s].
    module_height, crop_height : float
        Height of the modules and of the crop [m].

    All time series (dew_point to diffusivity) must cover the same hours as `times`: element i is hour i of the run.

    Returns
    -------
    results_df : pd.DataFrame
        Module surface temperature per module [°C] with time index.
    Gr : np.ndarray
        Grashof number (top) time series.
    hconv_top : np.ndarray
        Total top convection coefficients [W/(m²·K)] per module and time.
    tground_array : np.ndarray
        Ground temperature [K] per time.
    h_forced_top : np.ndarray
        Forced convection (top) coefficient [W/(m²·K)] per time.
    h_free_top : np.ndarray
        Free convection (top) coefficient [W/(m²·K)] per module and time.
    hground_array : np.ndarray
        Linearized radiation coefficient to ground [W/(m²·K)] per time.
    tsky_array : np.ndarray
        Sky temperature [K] per time.
    hsky_array : np.ndarray
        Linearized radiation coefficient to sky [W/(m²·K)] per time.
    hconv_bot_array : np.ndarray
        Bottom convection coefficients [W/(m²·K)] per module and time.
    tamb_bot_array : np.ndarray
        Converged bottom air temperatures [K] per module and time.
    Tout_bot_array : np.ndarray
        Outlet temperatures below the module [K] per module and time.
    mair_bot_array : np.ndarray
        Bottom control-volume air mass flow [kg/s] per module and time.
    q_vent_array : np.ndarray
        Ventilation sensible heat flow [W] per module and time.
    RH_out_array : np.ndarray
        Outlet RH below modules [%] per module and time.
    """
    # Setup
    ET = pd.Series(np.asarray(ET, dtype=float))  # positional access, element i = hour i of the run
    H = pd.Series(np.asarray(H, dtype=float))
    Kcb = np.asarray(Kcb, dtype=float)
    Ke = np.asarray(Ke, dtype=float)
    tilt_angle_rad = np.radians(tilt_angle)
    boltz = 5.669e-8  # Stefan–Boltzmann constant [W/m²·K⁴]
    xlen = func.Dh(
        module_width, module_length
    )  # hydraulic length
    cap = mc  # [J/(m²·K)] specific heat of the module
    pressure_kPa = pressure / 1000.0  # [kPa] atmospheric pressure
    Tground_K = Tground_C + 273.15  # [K] Ground temperature

    n = len(wind_speed)
    Gr = np.zeros(n)
    h_forced_top = np.zeros(n)

    # Per-time arrays
    tground_array = np.zeros_like(poa_global)
    hground_array = np.zeros_like(poa_global)
    tsky_array = np.zeros_like(poa_global)
    hsky_array = np.zeros_like(poa_global)

    # Per-module, per-time arrays
    tmod_array = np.zeros((num_modules, n))
    hconv_top = np.zeros((num_modules, n))
    h_free_top = np.zeros((num_modules, n))

    hconv_bot_array = np.zeros((num_modules, n))
    tamb_bot_array = np.zeros((num_modules, n))
    RH_out_array = np.zeros((num_modules, n))
    Tout_bot_array = np.zeros((num_modules, n))
    mair_bot_array = np.zeros((num_modules, n))
    q_vent_array = np.zeros((num_modules, n))

    # Time stepping control
    timedelta_hours = np.ones(len(poa_global))  # Δt = 1 h 
    Tair_K = np.array(Tair_K)
    daily_tamb_values = []

    # Initial conditions 
    sun0 = 0.0
    tmod0 = Tground_K[0]
    tground = Tground_K[0]
    tini_day = Tair_K[0]
    avg_tamb_day = Tair_K[0]

    # Time loop 
    for i, (tamb, sun, windmod, dtime) in enumerate(
        zip(Tair_K, poa_global, wind_speed, timedelta_hours)
    ):
        print(f"Hour {i}")
        day = i // 24 + 1
        countday = i - 6
        daily_tamb_values.append(tamb)

        daytime = "day" if sun > 0 else "night"
        if countday % 24 == 0 and i > 0:
            avg_tamb_day = np.mean(daily_tamb_values)
            daily_tamb_values = [tamb]
            tini_day = Tair_K[i]
            tmod0 = tini_day

        # Initialize per-time values
        tmod = np.full(num_modules, tmod0)

        # Ground temperature
        tground = Tground_K[i]
        tground_array[i] = tground

        # Sky temperature
        tsky = func.calc_temperature_sky(float(Tair_K[i]), float(dew_point[i]), daytime)

        # Initial bottom-air guesses
        tamb_bot = np.full(num_modules, tamb)
        RH_out = np.full(num_modules, RH[i])
        Tin = tamb
        windmod = max(windmod, 0.05)

        # Control volume geometry
        CV_area_ET = module_length * module_width * np.cos(tilt_angle_rad)
        Module_area = module_width * module_length

        # ET (latent heat) terms
        H_ET = (
            H.iloc[i]
        ) * CV_area_ET  # [W] = ET heat flux [W/m2] * Covered ground surface area of CV [m2]
        T_rate = Kcb[i] / (
            Kcb[i] + Ke[i]
        )  # [0-1] fraction of transpiration over evaporation: H = ET*B = ET0(Ke+Kcb)B = ET0*B*Ke + ET0*B*Kcb = H_E + H_T -> T_rate = H_T/H = Kcb/(Kcb+Ke) -> same for ET instead of H
        CV_height = (
            module_height - (crop_height / 4) * T_rate
        )  # [m] Height of the control volume (CV): h_module - center_of_gravity_[E(z=0),T(z=h/4)_fluxes]
        CV_area_vent = (
            CV_height * module_width * np.cos(tilt_angle_rad)
        )  # [m2] Section of air influx

        RH_in = RH_out[0]

        for module in range(num_modules):
            for j in range(10):
                # Top BL convection (forced + free)
                (
                    Nu_forced_i,
                    hconv_top[module, i],
                    h_forced_top[i],
                    h_free_top[module, i],
                    Gr[i],
                ) = func.calc_BL_top(
                    windmod,
                    thermal_conductivity[i],
                    diffusivity[i],
                    kinematic_viscosity[i],
                    tmod[module],
                    tamb,
                    module_length,
                    module_width,
                    tilt_angle,
                )

                # Bottom MB/EB iteration controls
                tol = 0.05
                iteration = 0
                max_iterations = 25

                ET_kgs = CV_area_ET * ET.iloc[i] / (24 * 3600.0)  # [kg/s], ET rate in mm/day = kg/(m2 day)
                m_air = max(density[i] * windmod * (CV_area_vent), 1e-6)
                tamb_bot_guess = tamb_bot[module] + 2 * tol
                tamb_bot_prev = tamb_bot[module]

                while (
                    abs(tamb_bot_guess - tamb_bot_prev) > tol
                    and iteration < max_iterations
                ):
                    iteration += 1

                    # Mass balance for RH below module
                    RH_out[module], w_out = func.RH_bot_calc(
                        m_air, tamb_bot_prev - 273.15, ET_kgs, pressure_kPa[i], RH_in
                    )

                    # Update air properties at bottom
                    air_prop = func.calc_air_properties(
                        tamb_bot_prev, RH_in, pressure[i]
                    )
                    (
                        dynamic_viscosity_bot,
                        density_bot,
                        kinematic_viscosity_bot,
                        Cp_bot,
                        thermal_conductivity_bot,
                        diffusivity_bot,
                        dew_point_bot,
                    ) = air_prop

                    # Bottom BL convection
                    Nu_forced_bot, hconv_bot, h_forced_bot, h_free_bot, Gr_bot = (
                        func.calc_BL_bot(
                            windmod,
                            thermal_conductivity_bot,
                            diffusivity_bot,
                            kinematic_viscosity_bot,
                            tmod[module],
                            tamb_bot_prev,
                            module_length,
                            module_width,
                            tilt_angle,
                        )
                    )

                    # Bottom convective heat
                    qconv_bot = hconv_bot * (tmod[module] - tamb_bot_prev) * Module_area
                    m_air = max(density_bot * windmod * CV_area_vent, 1e-6)

                    # Outlet and new guess
                    Tout = Tin + (H_ET + qconv_bot) / (m_air * Cp_bot)
                    tamb_bot_guess = (Tin + Tout) / 2

                # Store converged bottom air temperature
                tamb_bot[module] = tamb_bot_prev

                # Linearized longwave exchange (radiation losses)
                hsky = (
                    emissivity
                    * boltz
                    * (tmod[module] ** 2 + tsky**2)
                    * (tmod[module] + tsky)
                )
                hground = (
                    emissivity
                    * boltz
                    * (tmod[module] ** 2 + tground**2)
                    * (tmod[module] + tground)
                )

                # Transient ODE step solver for module temperature
                L = -((hconv_bot + hconv_top[module, i] + hsky + hground) / cap) * (
                    dtime * 3600.0
                )
                ex = np.exp(L) if L > -10 else 0.0
                delta_S = sun - sun0

                tmod[module] = tmod0 * ex + (
                    (1 - ex)
                    * (
                        hconv_bot * tamb_bot[module]
                        + hconv_top[module, i] * tamb
                        + hsky * tsky
                        + hground * tground
                        + sun0
                        + delta_S / L
                    )
                    + delta_S
                ) / (hconv_bot + hconv_top[module, i] + hsky + hground)

                tmod0 = tmod[module]
                sun0 = sun

            # Ventilation heat (negative if warming air)
            q_vent_array[module, i] = m_air * Cp_bot * (Tin - Tout)

            print(
                f"Hour {i}, Module {module}, wind:{windmod:.3f} ,Tair: {tamb:.3f}, "
                f"Tin: {Tin:.3f}, Tbot = {tamb_bot[module]:.3f}, Tout: {Tout:.3f}, "
                f"q_vent: {q_vent_array[module,i]:.3f}, H_ET: {H_ET:.3f}, q_conv,bot: {qconv_bot:.3f}"
            )

            # Pass outlet to next module inlet
            if module < num_modules - 1:
                tamb_bot[module + 1] = Tout
                Tin = Tout
                RH_in = RH_out[module]

            # Store per-module results
            hconv_bot_array[module, i] = hconv_bot
            tamb_bot_array[module, i] = tamb_bot[module]
            Tout_bot_array[module, i] = Tout
            mair_bot_array[module, i] = m_air
            tmod_array[module, i] = tmod[module]
            RH_out_array[module, i] = RH_out[module]

        # Store per-time results
        hground_array[i] = hground
        tsky_array[i] = tsky
        hsky_array[i] = hsky

    #  Return 
    results_df = pd.DataFrame(
        (tmod_array - 273.15).T,
        index=times,
        columns=[f"Module_{m+1}" for m in range(num_modules)],
    )

    return (
        results_df,
        Gr,
        hconv_top,
        tground_array,
        h_forced_top,
        h_free_top,
        hground_array,
        tsky_array,
        hsky_array,
        hconv_bot_array,
        tamb_bot_array,
        Tout_bot_array,
        mair_bot_array,
        q_vent_array,
        RH_out_array,
    )
