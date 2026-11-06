from thermo import Mixture
import numpy as np
from scipy.stats import pearsonr
import CoolProp.CoolProp as CP
from CoolProp.HumidAirProp import HAPropsSI
import pandas as pd
import pvlib
import matplotlib.pyplot as plt


def fuentes_hconv(tave, windmod, tinoct, temp_delta, xlen, tilt, check_reynold):
    # Calculate the convective coefficient as in Fuentes 1987 -- a mixture of
    # free, laminar, and turbulent convection.
    densair = 0.003484 * 101325.0 / tave  # density
    visair = 0.24237e-6 * tave**0.76 / densair  # kinematic viscosity
    condair = 2.1695e-4 * tave**0.84  # thermal conductivity
    reynold = windmod * xlen / visair
    # the boundary between laminar and turbulent is modeled as an abrupt change at Re = 1.2e5:
    if check_reynold and reynold > 1.2e5:
        # turbulent convection
        hforce = 0.0282 / reynold**0.2 * densair * windmod * 1007 / 0.71**0.4
    else:
        # laminar convection
        hforce = 0.8600 / reynold**0.5 * densair * windmod * 1007 / 0.71**0.67
    
    # free convection via Grashof number
    # NB: Fuentes hardwires sind(tilt) as 0.5 for tilt=30
    grashof = 9.8 / tave * temp_delta * xlen**3 / visair**2 * np.sin(np.radians(tilt))
    # product of Nusselt number and (k/l)
    hfree = 0.21 * (grashof * 0.71)**0.32 * condair / xlen
    
    # combine free and forced components
    hconv = (hfree**3 + hforce**3)**(1/3)
    return hconv, hfree, hforce, densair, visair, condair, reynold, grashof

def fuentes(poa_global, Tair, wind_speed, noct_installed, module_height,
            wind_height, emissivity, absorption,
            surface_tilt, module_width, module_length):

    boltz = 5.669e-8
    emiss = emissivity
    absorp = absorption
    xlen = pvlib.temperature._hydraulic_diameter(module_width, module_length)
    # cap0 has units of [J / (m^2 K)], equal to mass per unit area times
    # specific heat of the module.
    cap0 = 11000
    tinoct = noct_installed + 273.15

    # convective coefficient of top surface of module at NOCT
    windmod = 1.0
    tave = (tinoct + 293.15) / 2
    [hconv, hfree, hforce, densair, visair, condair, reynold, grashof] = fuentes_hconv(tave, windmod, tinoct, tinoct - 293.15, xlen, surface_tilt, False)

    # determine the ground temperature ratio and the ratio of the total convection to the top side convection
    hground = emiss * boltz * (tinoct**2 + 293.15**2) * (tinoct + 293.15)
    backrat = (absorp * 800.0 - emiss * boltz * (tinoct**4 - 282.21**4) - hconv * (tinoct - 293.15)) / ((hground + hconv) * (tinoct - 293.15))
    tground = (tinoct**4 - backrat * (tinoct**4 - 293.15**4))**0.25
    tground = np.clip(tground, 293.15, tinoct)

    tgrat = (tground - 293.15) / (tinoct - 293.15)
    convrat = (absorp * 800 - emiss * boltz * (
        2 * tinoct**4 - 282.21**4 - tground**4)) / (hconv * (tinoct - 293.15))

    # adjust the capacitance (thermal mass) of the module based on the INOCT.
    cap = cap0
    if tinoct > 321.15:
        cap = cap * (1 + (tinoct - 321.15) / 12)

    # iterate through timeseries inputs
    sun0 = 0

    # n.b. the way Fuentes calculates the first timedelta makes it seem like
    # the value doesn't matter -- rather than recreate it here, just assume
    # it's the same as the second timedelta:
    timedelta_seconds = poa_global.index.to_series().diff().dt.total_seconds()
    timedelta_hours = timedelta_seconds / 3600
    timedelta_hours.iloc[0] = timedelta_hours.iloc[1]

    tamb_array = Tair + 273.15
    sun_array = poa_global * absorp

    # Two of the calculations are easily vectorized, so precalculate them:
    # sky temperature -- Equation 24
    tsky_array = 0.68 * (0.0552 * tamb_array**1.5) + 0.32 * tamb_array  
    # wind speed at module height -- Equation 22
    windmod_array = wind_speed * (module_height/wind_height)**0.2 + 1e-4

    tmod0 = 293.15
    tmod_array = np.zeros_like(poa_global)
    tground_array = np.zeros_like(poa_global)
    hconv_array = np.zeros_like(poa_global)
    densair_array = np.zeros_like(poa_global)
    visair_array = np.zeros_like(poa_global)
    condair_array = np.zeros_like(poa_global)
    reynold_array = np.zeros_like(poa_global)
    hforce_array = np.zeros_like(poa_global)
    grashof_array = np.zeros_like(poa_global)
    hfree_array = np.zeros_like(poa_global)
    hground_array = np.zeros_like(poa_global)
    hsky_array = np.zeros_like(poa_global)

    iterator = zip(tamb_array, sun_array, windmod_array, tsky_array,
                   timedelta_hours)
    for i, (tamb, sun, windmod, tsky, dtime) in enumerate(iterator):
        # solve the heat transfer equation, iterating because the heat loss
        # terms depend on tmod. NB Fuentes doesn't show that 10 iterations is
        # sufficient for convergence.
        tmod = tmod0
        for j in range(10):
            # overall convective coefficient
            tave = (tmod + tamb) / 2
            [hconv, hfree, hforce, densair, visair, condair, reynold, grashof] = fuentes_hconv(tave, windmod, tinoct, abs(tmod-tamb), xlen, surface_tilt, True)
            hconv = convrat * hconv
            # sky radiation coefficient (Equation 3)
            hsky = emiss * boltz * (tmod**2 + tsky**2) * (tmod + tsky)
            # ground radiation coeffieicient (Equation 4)
            tground = tamb + tgrat * (tmod - tamb)
            hground = emiss * boltz * (tmod**2 + tground**2) * (tmod + tground)
            # thermal lag -- Equation 8
            eigen = - (hconv + hsky + hground) / cap * dtime * 3600
            # not sure why this check is done, maybe as a speed optimization?
            if eigen > -10:
                ex = np.exp(eigen)
            else:
                ex = 0
            # Equation 7 -- note that `sun` and `sun0` already account for
            # absorption (alpha)
            tmod = tmod0 * ex + ((1 - ex) * (hconv * tamb + hsky * tsky + hground * tground + sun0 + (sun - sun0) / eigen) + sun - sun0 ) / (hconv + hsky + hground)
        
        tmod_array[i] = tmod
        tground_array[i] = tground
        hconv_array[i] = hconv
        densair_array[i] = densair
        visair_array[i] = visair
        condair_array[i] = condair
        reynold_array[i] = reynold
        hforce_array[i] = hforce
        grashof_array[i] = grashof
        hfree_array[i] = hfree
        hground_array[i] = hground
        hsky_array[i] = hsky
        tmod0 = tmod
        sun0 = sun
    print('ConvRat ct: hconv = convrat * hconv_top ->', convrat)

    return (
        pd.Series(tmod_array - 273.15, index=poa_global.index, name='tmod'),
        pd.Series(tground_array - 273.15, index=poa_global.index, name='tground'),
        pd.Series(hconv_array, index=poa_global.index, name='hconv'),
        pd.Series(densair_array, index=poa_global.index, name='densair'),
        pd.Series(visair_array, index=poa_global.index, name='visair'),
        pd.Series(condair_array, index=poa_global.index, name='condair'),
        pd.Series(reynold_array, index=poa_global.index, name='reynold'),
        pd.Series(hforce_array, index=poa_global.index, name='hforce'),
        pd.Series(grashof_array, index=poa_global.index, name='grashof'),
        pd.Series(hfree_array, index=poa_global.index, name='hfree'),
        pd.Series(hground_array, index=poa_global.index, name='hground'),
        pd.Series(tsky_array, name='tsky'),
        pd.Series(hsky_array, index=poa_global.index, name='hsky')        
    )

def numberOfNans(data):
    count = 0
    for i in data:
        if np.isnan(i):
            count += 1
    return count 

def compute_statistics(Tm_model, Tm_reference):
    """Compute error statistics between the model and reference data, ignoring NaNs."""
    Tm_model = np.array(Tm_model)
    Tm_reference = np.array(Tm_reference)
    
    # Mask to exclude pairs with NaN or infinite values
    valid_mask = (~np.isnan(Tm_model)) & (~np.isnan(Tm_reference)) & \
                 (~np.isinf(Tm_model)) & (~np.isinf(Tm_reference))
    
    model_valid = Tm_model[valid_mask]
    reference_valid = Tm_reference[valid_mask]
    
    errors = model_valid - reference_valid
    
    mae = np.mean(np.abs(errors))  # Mean Absolute Error
    rmse = np.sqrt(np.mean(errors ** 2))  # Root Mean Square Error
    mbe = np.mean(errors)  # Mean Bias Error
    msre = np.mean((errors / reference_valid) ** 2)  # Mean Squared Relative Error
    std_dev = np.std(errors)  # Standard deviation of errors
    
    # Pearson correlation only if enough data points remain
    if len(model_valid) > 1:
        r_value, _ = pearsonr(model_valid, reference_valid)
    else:
        r_value = np.nan
    
    max_error = np.max(np.abs(errors)) if errors.size > 0 else np.nan
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MBE": mbe,
        "MSRE": msre,
        "Std Dev": std_dev,
        "Correlation (R)": r_value,
        "max.abs error": max_error
    }


####################################

# ET0 Reference FAO and ASCE (Standardization of Reference Evapotranspiration of the Environmental and Water Resources Institute of ASCE) calculation of ET0: 
# differs from this work: assumes ctPatm(elevation), ideal gas for density, esat, radiation(empirical model, through linear regressions)
def ascehourly(rfcrp,z,lat,lon,lzn,doy,sct,israd,tavg,vapr,
               tdew,rhum,tmin, wndsp,poa_global,wndht,tl,csreq,fcdpt):
    """Compute hourly ASCE Standardized Reference Evapotranspiration

    Parameters
    ----------
    rfcrp : str
        'S' for the short reference crop (0.12-m grass)
        'T' for the tall reference crop (0.50-m alfalfa)
    z : float
        Weather site elevation above mean sea level (m)
    lat : float
        Latitude of the weather site (decimal degrees)
    lon : float
        Longitude of the weather site (decimal degrees)
    lzn : float
        Longitude of the center of the local time zone (decimal degrees)
    doy : float
        Day number of the year between 1 and 366
    sct : float
        Standard clock time at the midpoint of the period (h)
        For the period between 1400-1500 hours, sct = 14.5 h
    israd : float
        Incoming solar radiation (MJ m^-2 h^-1)
    tavg : float
        Average air temperature (deg C)
    vapr : float, optional (but recommended)
        Average vapor pressure (kPa) (default = NaN)
    tdew : float, optional (but recommended)
        Average dew point temperature (deg C) (default = NaN)
    rhum : float, optional
        Average relative humidity (%) (default = NaN)
    tmin : float, optional
        Daily minimum air temperature (deg C) (default = NaN)
    wndsp : float, optional  (but recommended)
        Average wind speed (m s^-1) (default = NaN)
    wndht : float, optional (but recommended)
        Height of wind measurement above the ground (m) (default = 2.0)
    tl : float, optional
        Length of the calculation period (h) (default = 1.0)
    csreq : str, optional
        'S' for the simple Eq. 47 for clear sky solar radiation
        'D' for the complex method in Appendix D (default = 'D')
    fcdpt : float, optional
        Cloudiness value (fcd) from previous timestep (default = 1.0)

    Returns
    -------
    etsz : float
        Hourly standardized reference evapotranspiration for the
        short or tall reference crop (mm)
    fcd : Cloudiness value for possible use in next timestep
    """

    #patm (float) : Mean atmospheric pressure at weather station (kPa)
    #ASCE (2005) Eq. 34
    patm = 101.3*((293.0-0.0065*z)/293.0)**5.26

    #psycon (float) : Psychrometric constant (kPa (deg C)^-1)
    #ASCE (2005) Eq. 35
    psycon = 0.000665*patm

    #Udelta (float) : Slope of the saturation vapor pressure
    #temperature curve (kPa (deg C)^-1)
    #ASCE (2005) Eq. 36
    Udelta = 2503.0*np.exp(17.27*tavg/(tavg+237.3))
    Udelta = Udelta/((tavg+237.3)**2.0)

    #es (float) : Saturation vapor pressure (kPa)
    #ASCE (2005) Eq. 37
    es = 0.6108*np.exp((17.27*tavg)/(tavg+237.3))

    #ea (float): Actual vapor pressure (kPa) ASCE (2005) Table 4
    if not np.isnan(vapr):
        #ASCE (2005) Table 4
        ea = vapr
    elif not np.isnan(tdew):
        #ASCE (2005) Eq. 38
        ea = 0.6108*np.exp((17.27*tdew)/(tdew+237.3))
    elif not np.isnan(rhum):
        #ASCE (2005) Eq. 41
        ea = es*rhum/100.
    else:
        #ASCE (2005) Appendix E
        tdew = tmin - 2.0
        ea = 0.6108*np.exp((17.27*tdew)/(tdew+237.3))

    #rns (float) : Net shortwave radiation (MJ m^-2 h^-1)
    #ASCE (2005) Eq. 43
    albedo = 0.23
    rns = (1.0-albedo)*israd

    #ra (float) : Extraterrestrial radiation (MJ m^-2 h^-1)
    #ASCE (2005) Eqs. 48-58
    dr = 1.0+0.033*np.cos(2.0*np.pi/365.0*doy) #Eq. 50
    ldelta = 0.409*np.sin(2.0*np.pi/365.0*doy-1.39) #Eq. 51
    b = 2.0*np.pi*(doy-81.0)/364.0 #Eq. 58
    sc = 0.1645*np.sin(2.0*b)-0.1255*np.cos(b)-0.025*np.sin(b) #57
    wmid = np.pi/12.0*((sct+0.06667*(lzn-lon)+sc)-12.) #Eq. 55
    w1 = wmid-np.pi*tl/24.0 #Eq. 53
    w2 = wmid+np.pi*tl/24.0 #Eq. 54
    latrad = lat*np.pi/180.0 #Eq. 49
    ws = np.arccos(-1.0*np.tan(latrad)*np.tan(ldelta)) #Eq. 59
    if w1 < -1.0*ws: w1 = -1.0*ws #Eq. 56
    if w2 < -1.0*ws: w2 = -1.0*ws #Eq. 56
    if w1 > ws: w1 = ws #Eq. 56
    if w2 > ws: w2 = ws #Eq. 56
    if w1 > w2: w1 = w2 #Eq. 56
    ra1 = (w2-w1)*np.sin(latrad)*np.sin(ldelta) #Eq. 48
    ra2 = np.cos(latrad)*np.cos(ldelta) #Eq. 48
    ra3 = np.sin(w2)-np.sin(w1) #Eq. 48
    if wmid < -1.0*ws or wmid > ws:
        ra = 0.0
    else:
        ra = 12.0/np.pi*4.92*dr*(ra1+ra2*ra3) #Eq. 48

    #rso (float) : Clear sky solar radiation (MJ m^-2 h^-1)
    #ASCE (2005) Eq. 47 and Appendix D
    beta1 = np.sin(latrad)*np.sin(ldelta)
    beta2 = np.cos(latrad)*np.cos(ldelta)*np.cos(wmid)
    beta = np.arcsin(beta1+beta2) #Eq. 62 or D.6
    if csreq == 'S':
        rso = (0.75+2e-5*z)*ra #Eq. 47
    else:
        if beta < 0.3:
            rso = 0.0
        else:
            pwat = 0.14*ea*patm+2.1 #Eq. D.3
            kt = 1.0
            kb1 = -0.00146*patm/(kt*np.sin(beta))
            kb2 = 0.075*(pwat/np.sin(beta))**0.4
            kb = 0.98*np.exp(kb1-kb2) #Eq. D.2
            if kb >= 0.15:
                kd = 0.35-0.36*kb #Eq. D.4
            else:
                kd = 0.18+0.82*kb #Eq. D.4
            rso = (kb + kd)*ra #Eq. D.1

    #rnl (float) : Net longwave radiation (MJ m^-2 h^-1)
    #ASCE (2005) Eqs. 44, 45, and 62
    if beta < 0.3 or rso <= 0.0: #nighttime
        fcd = fcdpt
    else: #daytime
        ratio = sorted([0.3,israd/rso,1.0])[1] #Eq. 45
        fcd = sorted([0.05,1.35*ratio-0.35,1.0])[1] #Eq. 45
    tk4 = (tavg+273.16)**4.0 #Eq. 44
    rnl = 2.042e-10*fcd*(0.34-0.14*np.sqrt(ea))*tk4 #Eq. 44

    #rn (float) : Net radiation (MJ m^-2 h^-1)
    #ASCE (2005) Eq. 42
    rn = rns-rnl

    #g (float) : Soil heat flux (MJ m^-2 h^-1)
    #ASCE (2005) Eqs. 65 and 66
    #Aerodynamic roughness and surface resistance constants
    #ASCE (2005) Table 1
    if rfcrp == 'S': #Short reference crop (0.12-m grass)
        Cn = 37.0 #K mm s^3 Mg^-1 h^-1
        if rn < 0.0: #nighttime
            g = 0.5*rn
            Cd = 0.96 #!s m^-1
        else: #daytime
            g = 0.1*rn
            Cd = 0.24 #!s m^-1
    elif rfcrp == 'T': #Tall reference crop (0.50-m alfalfa)
        Cn = 66.0 #K mm s^3 Mg^-1 h^-1
        if rn < 0.0: #nighttime
            g = 0.2*rn
            Cd = 1.7 #!s m^-1
        else: #daytime
            g = 0.04*rn
            Cd = 0.25 #!s m^-1

    #u2 (float) : Wind profile relationship (m s^-1)
    #ASCE (2005) Eq. 67 and Appendix E
    if np.isnan(wndsp): wndsp = 2.0
    u2 = wndsp * (4.87/np.log(67.8*wndht-5.42))
    #etsz (float) : Standardized daily reference crop ET (mm h^-1)
    #ASCE (2005) Eq. 1
    etsz = 0.408*Udelta*(rn-g)+psycon*(Cn/(tavg+273.0))*u2*(es-ea)
    etsz = etsz/(Udelta+psycon*(1.0+Cd*u2))

    rad = poa_global*3600/(10**6) #W/m2 --> MJ/m^2/h
    ET0_rad = 0.408*Udelta*(rad-g)+psycon*(Cn/(tavg+273.0))*u2*(es-ea)
    ET0_rad = ET0_rad/(Udelta+psycon*(1.0+Cd*u2))

    return (etsz, rn, ET0_rad)


def plot_irradiance_data(
    spectral_radiation_df,
    spectral_radiation_df_excl,
    irradiance_w_m2,
    irradiance_w_m2_excl,
    spectral_radiation_module_E,
    spectral_radiation_module_W,
    spectral_radiation_ReferenceGrass,
    spectral_radiation_crop,
    spectral_radiation_grass,
    spectral_radiation_ground,
    poa_module_E,
    poa_module_W,
    poa_ReferenceGrass,
    poa_crop,
    poa_grass,
    poa_ground,
):
    """Check plots of the ray-tracing irradiance inputs: spectral and hourly irradiance, per data column and per surface."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # , dpi=600)
    axs = axs.flatten()  # make it a simple list for easier indexing

    ####################### 1 - Spectral Irradiance Data Columns #######################
    axs[0].set_title("Spectral Irradiance Data Columns", fontsize=16)
    axs[0].plot(spectral_radiation_df["PVfE"], label="PVfE", color="pink")
    axs[0].plot(spectral_radiation_df["PVfW"], label="PVfW", color="lightblue")
    axs[0].plot(spectral_radiation_df_excl["GC"], label="GC_RefGrass", color="lightgreen")
    axs[0].plot(spectral_radiation_df["PVbE"], label="PVbE", color="#C71585")
    axs[0].plot(spectral_radiation_df["PVbW"], label="PVbW", color="#5F9EA0")
    axs[0].plot(spectral_radiation_df["GOw"], label="GOw", color="orange")
    axs[0].plot(spectral_radiation_df["GOe"], label="GOe", color="#FFD700")
    axs[0].plot(spectral_radiation_df["GC"], label="GC", color="green")
    axs[0].plot(spectral_radiation_df["VLf"], label="VLf", color="r")
    axs[0].plot(spectral_radiation_df["VLb"], label="VLb", color="darkred")
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.5)
    axs[0].set_xlabel("Wavelength (nm)", fontsize=14)
    axs[0].set_ylabel("Spectral Irradiance [W/m²·nm]", fontsize=14)
    axs[0].legend(fontsize=12)

    ####################### 2 - Spectral Irradiance Field #################################
    axs[1].set_title("Spectral Irradiance on the Field by Surface", fontsize=16)
    axs[1].plot(spectral_radiation_module_E, label="East Module", color="#C71585")
    axs[1].plot(spectral_radiation_module_W, label="West Module", color="#5F9EA0")
    axs[1].plot(
        spectral_radiation_ReferenceGrass, label="Reference Grass", color="lightgreen"
    )
    axs[1].plot(spectral_radiation_crop, label="Canopy", color="green")
    axs[1].plot(spectral_radiation_grass, label="Grass Ground", color="b")
    axs[1].plot(spectral_radiation_ground, label="Averaged Ground", color="red")
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axs[1].set_xlabel("Wavelength (nm)", fontsize=14)
    axs[1].set_ylabel("Spectral Irradiance [W/m²·nm]", fontsize=14)
    axs[1].legend(fontsize=12)

    ####################### 3 - Time Series Irradiance Data Columns #######################
    axs[2].set_title("Time Series Irradiance Data Columns", fontsize=16)
    axs[2].plot(irradiance_w_m2["PVfE"], label="PVfE", color="pink")
    axs[2].plot(irradiance_w_m2["PVfW"], label="PVfW", color="lightblue")
    axs[2].plot(irradiance_w_m2_excl["GC"], label="GC_RefGrass", color="lightgreen")
    axs[2].plot(irradiance_w_m2["PVbE"], label="PVbE", color="#C71585")
    axs[2].plot(irradiance_w_m2["PVbW"], label="PVbW", color="#5F9EA0")
    axs[2].plot(irradiance_w_m2["GOw"], label="GOw", color="orange")
    axs[2].plot(irradiance_w_m2["GOe"], label="GOe", color="#FFD700")
    axs[2].plot(irradiance_w_m2["GC"], label="GC", color="green")
    axs[2].plot(irradiance_w_m2["VLf"], label="VLf", color="r")
    axs[2].plot(irradiance_w_m2["VLb"], label="VLb", color="darkred")
    axs[2].set_xlabel("Date", fontsize=14)
    axs[2].set_ylabel("Irradiance [W/m²]", fontsize=14)
    axs[2].legend(fontsize=12)
    axs[2].grid(True, axis="y", linestyle="--", alpha=0.5)
    axs[2].tick_params(axis="x", rotation=45)

    ####################### 4 - Time Series of Surface Irradiance in the Field #######################
    axs[3].set_title("Time Series of Irradiance on the Field by Surface", fontsize=16)
    axs[3].plot(poa_module_E, label="East Module", linewidth=1.5, color="#C71585")
    axs[3].plot(poa_module_W, label="West Module", linewidth=1.5, color="#5F9EA0")
    axs[3].plot(
        poa_ReferenceGrass, label="Reference Grass", linewidth=1.5, color="lightgreen"
    )
    axs[3].plot(poa_crop, label="Canopy", linewidth=1.5, color="green")
    axs[3].plot(poa_grass, label="Grass Ground", linewidth=1.5, color="b")
    axs[3].plot(poa_ground, label="Averaged Ground", linewidth=1.5, color="red")
    axs[3].set_xlabel("Date", fontsize=14)
    axs[3].set_ylabel("Irradiance [W/m²]", fontsize=14)
    axs[3].legend(fontsize=12)
    axs[3].grid(True, axis="y", linestyle="--", alpha=0.5)
    axs[3].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
