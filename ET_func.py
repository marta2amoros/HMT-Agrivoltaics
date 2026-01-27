import pyfao56 as fao
import numpy as np
import pandas as pd
import functions_module as func
import func_check as checks
import os
import pyfao56.custom as custom
from datetime import datetime
import math

# PYFAO56

def extract_results(file, result_name):
    with open(file, "r") as file:
        lines = file.readlines()
    # Identify the header row and column positions
    header_line = None
    for i, line in enumerate(lines):
        if "Year-DOY" in line and result_name in line:
            header_line = i
            break
    # Extract the column names and determine the index of result = "ETcadj"
    if header_line is not None:
        headers = lines[header_line].split()
        result_index = headers.index(result_name)
        # Extract the ETcadj column data from the subsequent lines
        result_values = []
        for line in lines[header_line + 1:]:
            columns = line.split()
            if len(columns) > result_index:  # Ensure there is data in the expected column
                result_values.append(columns[result_index]) # array with numbers as strings
        result_df = pd.DataFrame(result_values, columns=[result_name])   # Convert to a DataFrame
        result = result_df[result_name].astype(float) # Convert the ETcadj DataFrame to a normal array (list of floats)
    else:
        print("Could not find the header row. Please check the file structure.")
        
    return result

# REFERENCE EVAPOTRANSPIRATION (ET0) CALCULATION

def ET0_calc(wind_speed_2m, wind_speed_2m_day, Tair, dew_point, RH, pressure, poa_ground, density, Cp):
    # INPUT DATA FROM WEATHER FILE AND AIR PROPERTIES:
    wind_speed_2m # average hourly wind speed [m s-1].
    # wind_speed_2m_day = np.mean(wind_speed_2m.values.reshape(-1, 24), axis=1) # mean wind velocity of the day
    Tamb = Tair # mean hourly air temperature [°C],
    Tmax_day = np.max(Tamb.reshape(-1, 24), axis=1) # max Tº of the day
    Tmin_day = np.min(Tamb.reshape(-1, 24), axis=1) # min Tº of the day
    Tmean_day = np.mean(Tamb.reshape(-1, 24), axis=1) # min Tº of the day
    dew_point_day = np.mean(dew_point.reshape(-1, 24), axis=1) - 273.15 # mean dew point Tº of the day ºC
    RHmax_day = np.max(RH.reshape(-1, 24), axis=1) # max RH of the day
    RHmin_day = np.min(RH.reshape(-1, 24), axis=1) # min RH of the day
    RH_day = np.mean(RH.reshape(-1, 24), axis=1) # mean RH of the day
    density_air = density # kg/m3
    Cp_air = Cp/10**6  # specific heat at constant pressure [MJ kg-1 °C-1] = [J kg-1 K-1]*10^-6  ~1.013*10-3 
    Pressure = pressure/1000 # kPa

    Rn = poa_ground * 0.0036 # net radiation at the grass surface [MJ m-2 hour-1] = poa_ground[W/m2] * [1W=1J/s] * [3600s/1h] * [10^-6 MJ/J] 
    Rn_day = np.sum(Rn.values.reshape(-1, 24), axis=1) # [MJ m-2 day-1]  .values = from pandas to array type, .reshape = Each row represents a day -> Sum along the hours (columns) for each day
    LAI = 24*0.12 # Assumption for clipped grass eq.45 FAO [26]
    G = 0.4*np.exp(-0.5*LAI)*Rn # soil heat flux density [MJ m-2 hour-1], pg 251 [26]
    G_day = 0.4*np.exp(-0.5*LAI)*Rn_day
    lat_vap = 2.45 # latent heat vaporization [MJ/kg]

    e_sat = np.where(Tamb > 0, 0.611 * 10 ** (7.45 * Tamb / (237.3 + Tamb)),
        0.611 * 10 ** (9.5 * Tamb / (265.5 + Tamb)))  # saturation vapor pressure at air temperature [kPa] using Magnus formula [58].
    e_sat_FAO = 0.6108*np.exp(17.27*Tamb/(Tamb + 237.3))
    e_a = e_sat*RH # average hourly actual vapour pressure [kPa]

    delta = 4098*e_sat/(Tamb + 237.3)**2    # saturation slope vapour pressure curve at Thr [kPa °C-1]
    gamma = Cp_air*Pressure/(0.622*2.45)    # psychrometric constant [kPa °C-1]

    gamma_day = np.mean(gamma.reshape(-1, 24), axis=1) 
    delta_day = np.mean(delta.reshape(-1, 24), axis=1) 
    e_sat_day = np.mean(e_sat.reshape(-1, 24), axis=1) 
    e_a_day = np.mean(e_a.reshape(-1, 24), axis=1) 

    # REF CROP RESISTANCES
    h_grass = 0.12                                                                  # Crop height assumption for grass [m]
    zm = 2                                                                          # height of wind [m], we scaled windvel to 2m
    zh = zm                                                                         # height of humidity measurements [m]
    d = 2/3 * h_grass                                                               # zero plane displacement height [m]
    zom = 0.123*h_grass                                                             # roughness length governing momentum transfer [m]
    zoh = 0.1*zom                                                                   # roughness length governing transfer of heat and vapour [m]
    k = 0.41                                                                        # von Karman's constant, 0.41 [-]
    r_a = (np.log((zm-d)/zom)*np.log((zh-d)/zoh)/(k**2*wind_speed_2m)) *(1/3600)    # Aerodynamic resistance [s/m * 1h/3600s] = hr/m
    r_a_FAO = (208/wind_speed_2m)*(1/3600) # Approx

    rl = 100                                # s/m, stomatal resistance well-watered conditions
    LAIactive = 0.5*LAI                     # ActiveLAI = 0.5*LAI [m2leaf/m2soil]
    r_s = rl/LAIactive *(1/3600)            # Bulk surface resistance [s/m * 1h/3600s] = hr/m
    r_s_FAO = 70*(1/3600)                   # Approx h/m


    ET0_FAO_day = (0.408*delta_day*(Rn_day-G_day) + gamma_day*(900/(Tmean_day + 273))*wind_speed_2m_day*(e_sat_day - e_a_day))/(delta_day + gamma_day*(1 + 0.34*wind_speed_2m_day)) #pg 74 FAO[26] Using all the assumptions :(
    ET0_FAO_h = 24*((0.408)*delta*(Rn-G) + gamma*(37/(Tamb + 273))*wind_speed_2m*(e_sat - e_a))/(delta + gamma*(1 + 0.34*wind_speed_2m))  # mm/day

    
    ET0 = (24)*(1/lat_vap)*(delta*(Rn-G) + (density_air*Cp_air)*(e_sat - e_a)/r_a)/(delta + gamma*(1 + r_s/r_a)) # [mm/day] == [1day = 24hr]*[kg/m2*hr] * [1 kg/m2 = 1mm_water_depth],  pg 19
    q_ET0_Wm2 = ET0*(lat_vap*10**6)/(24*3600)  # W/m2 == [mm/day] * [MJ/kg] * [10^6 J/ 1MJ] * [1day / 24*3600 s] 

    # Calculate the mean for each day
    times = pd.date_range('2023-01-01 00:30:00', freq='h', periods=8760, tz='Etc/GMT-1')
    ET0.index = times
    ET0_day = ET0.resample('D').mean()

    return ET0, ET0_day, ET0_FAO_day, ET0_FAO_h, q_ET0_Wm2

# SINGLE CROP COEFFICIENT MODEL

def Kc_calc(windvel_day, RHmin_day, h, Lini, Ldev, Lmid, Llate, Start_growth, Kc_nul, Kc_ini, Kc_mid, Kc_end):
    # LIMITS for RHmin, wind_speed_2m
    RHmin_d = np.clip(RHmin_day, 20, 80)
    windvel_d = np.clip(windvel_day, 1, 6)
    Ltotal = Lini + Ldev + Lmid + Llate  # Total active growth period

    # Create an array for Kcb values over the year (365 days)
    Kc_table = np.full(365, Kc_nul)  # Default to dormancy value

    # Assign values based on growth stages
    day_index = Start_growth  

    # Adjust Lini so that Kcb_ini is at the middle of the period
    Kcb_initial_values = np.linspace(Kc_nul, 2 * Kc_ini - Kc_nul, Lini)  # Ensures average is Kcb_ini
    Kc_table[day_index:day_index + Lini] = Kcb_initial_values
    last_Kcb_Lini = Kcb_initial_values[-1]  # Last value of Lini
    day_index += Lini

    # Development stage (linear increase from last_Kcb_Lini to Kcb_mid)
    dev_start = day_index
    for i in range(Ldev):
        Kc_table[day_index + i] = last_Kcb_Lini + (i / Ldev) * (Kc_mid - last_Kcb_Lini)  # FAO [eq 66] linear interpolation
    day_index += Ldev

    # Mid-season stage (constant value)
    mid_start = day_index
    Kc_table[mid_start:mid_start + Lmid] = Kc_mid
    day_index += Lmid

    # Late-season stage (linear decrease)
    late_start = day_index
    for i in range(Llate):
        Kc_table[late_start + i] = Kc_mid - (i / Llate) * (Kc_mid - Kc_end)
    day_index += Llate

    # Gradual transition back to dormancy (Kcb_end to Kcb_nul over Llate)
    for i in range(Llate):
        Kc_table[day_index + i] = Kc_end - (i / Llate) * (Kc_end - Kc_nul)
    
    # Apply FAO [70] adjustments only during development, mid, and late stages
    Kc_fluct = Kc_table.copy()
    Kc_fluct[mid_start:day_index] += ((0.04 * (windvel_d[mid_start:day_index] - 2)) - 
                                       (0.004 * (RHmin_d[mid_start:day_index] - 45)) * ((h / 3) ** 0.3))
 
    return Kc_table, Kc_fluct

# DUAL CROP COEFFICIENT MODEL

def Kcb_calc(windvel_day, RHmin_day, h, Lini, Ldev, Lmid, Llate, Start_growth, Kcb_nul, Kcb_ini, Kcb_mid, Kcb_end):
    """
    Compute the Kcb table and adjusted Kcb values for an apple tree over a year.

    This function calculates the basal crop coefficient (Kcb) values over the year based on predefined growth stages and then adjusts them using wind speed and relative humidity following FAO equation [70, FAO 26].

    Parameters:
    - windvel_day: NumPy array of daily wind speed values (m/s) for the year.
    - RHmin_day: NumPy array of daily minimum relative humidity (%) for the year.
    - h: Crop height (m).
    - Lini: Length of the initial growth stage (days).
    - Ldev: Length of the development stage (days).
    - Lmid: Length of the mid-season stage (days).
    - Llate: Length of the late-season stage (days).
    - Start_growth: Day of the year when growth starts (March 1st = 59).
    - Kcb_nul: Kcb value during dormancy (before and after the growing season).
    - Kcb_ini: Kcb value at the start of growth.
    - Kcb_mid: Peak Kcb value during mid-season.
    - Kcb_end: Kcb value at the end of the growing season before transition to dormancy.

    Returns:
    - Kcb_table: NumPy array of unadjusted Kcb values for each day of the year (365 days).
    - Kcb: NumPy array of adjusted Kcb values incorporating wind speed and relative humidity.

    Notes:
    - The transition period between dormancy and the initial growth stage, as well as between 
      the end of the growing season and dormancy, lasts 4 weeks as referenced in paper [75] and [26].
    - The function implements FAO equation [66] for linear interpolation during the development 
      and late-season stages and FAO equation [70] for environmental adjustments.
    """
    # LIMITS for RHmin, wind_speed_2m
    RHmin_d = np.clip(RHmin_day, 20, 80)
    windvel_d = np.clip(windvel_day, 1, 6)

    Ltotal = Lini + Ldev + Lmid + Llate  # Total active growth period

    # Create an array for Kcb values over the year (365 days)
    Kcb_table = np.full(365, Kcb_nul)  # Default to dormancy value

    # Assign values based on growth stages
    day_index = Start_growth  

    # Adjust Lini so that Kcb_ini is at the middle of the period
    Kcb_initial_values = np.linspace(Kcb_nul, 2 * Kcb_ini - Kcb_nul, Lini)  # Ensures average is Kcb_ini
    Kcb_table[day_index:day_index + Lini] = Kcb_initial_values
    last_Kcb_Lini = Kcb_initial_values[-1]  # Last value of Lini
    day_index += Lini

    # Development stage (linear increase from last_Kcb_Lini to Kcb_mid)
    dev_start = day_index
    for i in range(Ldev):
        Kcb_table[day_index + i] = last_Kcb_Lini + (i / Ldev) * (Kcb_mid - last_Kcb_Lini)  # FAO [eq 66] linear interpolation
    day_index += Ldev

    # Mid-season stage (constant value)
    mid_start = day_index
    Kcb_table[mid_start:mid_start + Lmid] = Kcb_mid
    day_index += Lmid

    # Late-season stage (linear decrease)
    late_start = day_index
    for i in range(Llate):
        Kcb_table[late_start + i] = Kcb_mid - (i / Llate) * (Kcb_mid - Kcb_end)
    day_index += Llate

    # Gradual transition back to dormancy (Kcb_end to Kcb_nul over Llate)
    for i in range(Llate):
        Kcb_table[day_index + i] = Kcb_end - (i / Llate) * (Kcb_end - Kcb_nul)

    # Apply FAO [70] adjustments only during development, mid, and late stages
    Kcb_fluct = Kcb_table.copy()
    Kcb_fluct[mid_start:day_index] += ((0.04 * (windvel_d[mid_start:day_index] - 2)) - 
                                       (0.004 * (RHmin_d[mid_start:day_index] - 45)) * ((h / 3) ** 0.3))
    
    # Adjust Kcb with constant Kcb_mid and Kcb_end values
    Kcb = Kcb_table.copy()
    Kmid_adj = np.mean(Kcb_fluct[mid_start:mid_start + Lmid])  # Average Kcb_mid value
    Kend_adj = np.mean(Kcb_fluct[late_start:late_start + Llate])  # Average Kcb_end value
    
    Kcb[Start_growth+Lini:mid_start] = np.linspace(Kcb[Start_growth+Lini], Kmid_adj, Ldev)
    Kcb[mid_start:late_start] = Kmid_adj 
    Kcb[late_start:late_start + Llate] = np.linspace(Kmid_adj, Kend_adj, Llate)
    Kcb[late_start + Llate : late_start + 2*Llate] = np.linspace(Kend_adj, Kcb_nul, Llate)

    return Kcb_table, Kcb_fluct, Kcb

def Ke_ET_calc(ET0, ET0_day, Kcb, windvel_day, RHmin_day, net_irr, precipitation_day, Kc_min, fw, TEW, REW, h):
    ET0_day = np.array(ET0_day)    # pandas to numpy array
    num_days = len(ET0_day)
    Kr = np.zeros(num_days)        # Soil evaporation reduction coefficient
    Ke = np.zeros(num_days)        # Soil evaporation coefficient
    ETc_dual_day = np.zeros(num_days)  # Dual crop evapotranspiration
    De_day = np.zeros(num_days)    # Cumulative depletion (evaporation profile)
    q_ETc_dual_day_Wm2 = np.zeros(num_days)

    q_ETc_dual_h_Wm2 = np.zeros(num_days*24)
    ETc_dual_h = np.zeros(num_days*24)
    Kcb_h = np.zeros(num_days*24)        # Soil evaporation reduction coefficient
    Ke_h = np.zeros(num_days*24)        # Soil evaporation coefficient

    windvel_day = np.clip(windvel_day, 1, 6)  # Limit wind speed to [1, 6] m/s
    RHmin_day = np.clip(RHmin_day, 20, 80)  # Limit RHmin to [20, 80] %

    # Compute non-iterative parameters:
    Kc_max = np.maximum(1.2 + ((0.04 * (windvel_day - 2)) - (0.004 * (RHmin_day - 45))) * ((h / 3) ** 0.3), Kcb + 0.05)  # Max crop coefficient, FAO56, Eq. 72  
    fc = ((Kcb-Kc_min)/(Kc_max-Kc_min))**(1+0.5*h) # Compute canopy cover fraction, FAO56, eq 76
    few = np.minimum(1-fc, fw) # FAO56, eq 75
    DPe_day = 0  # Deep percolation (assumed 0)
    De_day[0] = 0

    # Initial depletion condition: Set to TEW or 0 after heavy rain/irrigation
    if np.max(precipitation_day[:3] + net_irr[:3]) > REW:  # Check heavy rainfall/irrigation in recent 3 days
        De_day[0] = 0
    else:
        De_day[0] = TEW   # De_prev = 14 # !!!!! Water balance TEW > De_prev > REW...

    for i in range(num_days):
        # Compute Kr coefficient (limited to [0,1])
        Kr[i] = min((TEW - De_day[i - 1]) / (TEW - REW), 1) # FAO56, Eq. 74
        Ke[i] = np.minimum(Kr[i]*(Kc_max[i]-Kcb[i]), few[i]*Kc_max[i]) # FAO56, eq 73
        # Compute daily cumulative depletion (De)
        De_day[i] = De_day[i - 1] - precipitation_day[i] - net_irr[i] / fw + (ET0_day[i] * Ke[i]) / few[i] + DPe_day # FAO56, eq 77
        # Ensure De does not go below 0 or above TEW
        De_day[i] = max(0, min(TEW, De_day[i]))
        # Compute dual crop evapotranspiration:  hourly values (assuming constant throughout the day)
        Kcb_h[i * 24:(i + 1) * 24] = np.repeat(Kcb[i], 24) 
        Ke_h[i * 24:(i + 1) * 24] = np.repeat(Ke[i], 24)
        
        ETc_dual_day[i] = ET0_day[i] * (Kcb[i] + Ke[i]) # mm/day
        lat_vap = 2.45  # latent heat vaporization [MJ/kg]
        q_ETc_dual_day_Wm2[i] = ETc_dual_day[i]*(lat_vap*10**6)/(24*3600)  # W/m2 == [mm/day] * [MJ/kg] * [10^6 J/ 1MJ] * [1day / 24*3600 s] 

    
    ETc_dual_h = ET0 * (Kcb_h + Ke_h) # mm/day
    q_ETc_dual_h_Wm2 = ETc_dual_h*(lat_vap*10**6)/(24*3600)  # W/m2 == [mm/day] * [MJ/kg] * [10^6 J/ 1MJ] * [1day / 24*3600 s] 
    return Ke, Ke_h, Kcb_h, ETc_dual_h, ETc_dual_day, Kr, fc, De_day, q_ETc_dual_day_Wm2, q_ETc_dual_h_Wm2

def Ke_linear_calc(Ke, Lini, Ldev, Lmid, Llate, Start_growth):
    """
    Linearly interpolate Ke values across different growth stages 
    while ensuring the average Ke at each stage remains at the midpoint.

    Parameters:
    - Ke: Array of fluctuating Ke values for the year (365 days).
    - Lini: Initial growth stage length (days).
    - Ldev: Development stage length (days).
    - Lmid: Mid-season stage length (days).
    - Llate: Late-season stage length (days).
    - Start_growth: Day of the year when growth starts (default: March 1st = 59).

    Returns:
    - Ke_interpolated: NumPy array of interpolated Ke values for each day of the year (365 days).
    """

    # Compute average Ke values at each stage
    Ke_nul1 = np.mean(Ke[:Start_growth])  
    Ke_ini = np.mean(Ke[Start_growth:Start_growth+Lini])
    Ke_mid = np.mean(Ke[Start_growth+Lini:Start_growth+Lini+Lmid])  
    Ke_end = np.mean(Ke[Start_growth+Lini+Lmid:Start_growth+Lini+Lmid+Llate])  
    Ke_nul2 = np.mean(Ke[Start_growth+Lini+Lmid+Llate:]) 

    # Create an array for interpolated Ke values
    Ke_interpolated = np.zeros(365)

    # Assign values based on growth stages
    day_index = 0

    # Before growth stage: Linear interpolation with Ke_nul1 centered at the midpoint
    Ke_nul1_values = np.linspace(Ke_nul1, Ke_ini, Start_growth) 
    Ke_interpolated[day_index:day_index + Start_growth] = Ke_nul1_values
    last_Ke_nul1 = Ke_nul1_values[-1]  
    day_index += Start_growth

    # Initial stage: Linear interpolation with Ke_ini centered at the midpoint
    Ke_initial_values = np.linspace(last_Ke_nul1, Ke_mid, Lini)
    Ke_interpolated[day_index:day_index + Lini] = Ke_initial_values
    last_Ke_Lini = Ke_initial_values[-1]  
    day_index += Lini

    # Development stage: Linearly increase towards Ke_mid
    Ke_dev_values = np.linspace(last_Ke_Lini, Ke_mid, Ldev)
    Ke_interpolated[day_index:day_index + Ldev] = Ke_dev_values
    last_Ke_Ldev = Ke_dev_values[-1]
    day_index += Ldev

    # Mid-season stage: **Now linearly interpolated** instead of constant
    Ke_mid_values = np.linspace(last_Ke_Ldev, Ke_end, Lmid)
    Ke_interpolated[day_index:day_index + Lmid] = Ke_mid_values
    last_Ke_Mid = Ke_mid_values[-1]
    day_index += Lmid

    # Late-season stage: Linearly decrease from Ke_mid to Ke_end
    Ke_late_values = np.linspace(last_Ke_Mid, Ke_end, Llate)
    Ke_interpolated[day_index:day_index + Llate] = Ke_late_values
    last_Ke_Llate = Ke_late_values[-1]
    day_index += Llate

    # Gradual transition back to dormancy (Ke_end to Ke_nul2)
    Ke_end_to_dormancy = np.linspace(last_Ke_Llate, Ke_nul2, 365 - day_index)
    Ke_interpolated[day_index:] = Ke_end_to_dormancy

    return Ke_interpolated