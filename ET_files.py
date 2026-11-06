import pyfao56 as fao
import numpy as np
import pandas as pd
import functions_module as func
import os
import pyfao56.custom as custom
from datetime import datetime
import math

def create_weather_file(daily_data, output_file):
    """   
    Parameters:s
    - daily_data (dict): A dictionary containing weather data for each day.
    - output_file (str): The file path where the weather data should be saved.

    """
    current_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    with open(output_file, 'w') as f:
        f.write("************************************************************************\n")
        f.write("pyfao56: FAO-56 Evapotranspiration in Python\n")
        f.write("Weather Data\n")
        f.write(f"Timestamp: {current_datetime}\n") 
        f.write("************************************************************************\n")
        f.write(f"Comments: DOY {daily_data['DOY'][0]} to {daily_data['DOY'][-1]}\n")
        f.write("************************************************************************\n")
        f.write("           S Reference crop - Short ('S') or Tall ('T')\n")
        f.write(" 361.0000000 Weather station elevation (z) (m)\n")
        f.write("  33.0690000 Weather station latitude (decimal degrees)\n")
        f.write("   3.0000000 Wind speed measurement height (m)\n")
        f.write("\nDaily weather data:\n")
        f.write("Year-DOY   Srad   Tmax   Tmin   Vapr   Tdew  RHmax  RHmin  Wndsp   Rain  ETref   MorP\n")
        
        # Write daily data row by row
        for i in range(len(daily_data["DOY"])):
            vapr_value = "NaN" if daily_data["Vapr"][i] is None else f"{daily_data['Vapr'][i]:6.2f}"
            f.write(f"2023-{daily_data['DOY'][i]:03d} {daily_data['Srad'][i]:6.2f} {daily_data['Tmax'][i]:6.2f} "
                    f"{daily_data['Tmin'][i]:6.2f} {vapr_value:6} {daily_data['Tdew'][i]:6.2f} "
                    f"{daily_data['RHmax'][i]:6.2f} {daily_data['RHmin'][i]:6.2f} {daily_data['Wind'][i]:6.2f} "
                    f"{daily_data['Rain'][i]:6.2f} {daily_data['ETref'][i]:6.2f} {daily_data['MorP'][i]:>6}\n")

    print(f"Manual weather data saved to {output_file}")


def create_parameters_file():
    #Get the module directory
    module_dir = os.getcwd()  # Gets the current working directory
    # Specify the model parameters for Pear crop
    par = fao.Parameters(comment = 'Pear')
    par.Kcbini = 0.6  # table 17, FAO [26]
    par.Kcbmid = 0.95  # table 17, FAO [26]
    par.Kcbend = 0.75  # table 17, FAO [26]
    par.Lini = 20      # table 11, FAO [26]
    par.Ldev = 70      # table 11, FAO [26]
    par.Lmid = 90      # table 11, FAO [26]
    par.Lend = 30      # table 11, FAO [26]
    # par.hini = 4     # initial plant height                                      (random value)   
    par.hmax = 2.4     # table 12, FAO [26]
    par.thetaFC = 0.25      # Vol. Soil Water Content, Field Capacity (cm3/cm3)     
    par.thetaWP = 0.01      # Vol. Soil Water Content, Wilting Point (cm3/cm3)   
    # par.theta0  = 0.1515    # Vol. Soil Water Content, Initial (cm3/cm3)         (Random value)
    # par.Zrini = 1.0        # Rooting Depth Initial (m)                          (Random value)
    par.Zrmax = 2  # 1.0-2.0 table 22, FAO [26] 
    # par.pbase = 0.5  # Depletion fraction, adapt it: p = p_table22(ET~5mm/day) + 0.04*(5 – ETc)     table 22, FAO [26]
    par.Ze = 0.1     # Depth of surface evaporation layer (m)  (Assumption) - Table 19, pg144 of FAO-56 [26]
    par.REW = 6    # Total depth Stage 1 evaporation (mm) - Choosing type of soil: LOAM (Assumption) - Table 19, pg144 of FAO-56 [26]
    par.savefile(os.path.join(module_dir,'Pear.par'))
    print(f"Manual parameters saved in {module_dir}")


def create_irrigation_file():
    module_dir = os.getcwd()  # Gets the current working directory
    #Specify the irrigation schedule for the 100%-100% treatment
    irr100 = fao.Irrigation(comment = 'Pear, 100%-100% ' + 'irrigation treatment')
    # Define the irrigation period (1 March to 26 October)
    start_doy = 60  # Approximate day of year for 1st March
    end_doy = 299   # Approximate day of year for 26th October
    net_irr_values = np.zeros(365)  # Initialize with zeros for full year

    # Open and read the irrigation data file
    file_path = "Irrigation.TXT"  # Use the correct path
    with open(file_path, "r", encoding="latin-1") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 9:  # Ensure valid data row
                try:
                    day_str, month_str = parts[:2]  # Extract date (e.g., "1 Mar")
                    net_irr = float(parts[8].replace(',', '.'))  # Convert to float

                    # Convert date to day of year (DOY)
                    date_str = f"{day_str} {month_str} 2024"  # Example: "1 Mar 2023"
                    doy = pd.to_datetime(date_str, format="%d %b %Y").dayofyear

                    # Assign value only if within irrigation period
                    if start_doy <= doy <= end_doy:
                        net_irr_values[doy - 1] = net_irr  # Adjust for 0-based index
                except (IndexError, ValueError):
                    continue  # Skip invalid lines
    net_irr_array = np.array(net_irr_values)

    "HOMOGENIZE IRRIGATION"
    net_irr_homog = np.zeros_like(net_irr_array)  # Initialize with zeros
    non_zero_indices = np.where(net_irr_array > 0)[0]  # Indices of non-zero values
    if len(non_zero_indices) == 0:
        return net_irr_array  # Return as is if no non-zero values exist
    prev_idx = 0
    for idx in non_zero_indices:
        segment_length = idx - prev_idx + 1  # Number of elements in the segment
        distributed_value = net_irr_array[idx] / segment_length  # Even distribution
        # Assign values to the segment
        net_irr_homog[prev_idx:idx+1] = distributed_value
        prev_idx = idx + 1  # Move to the next segment
    net_irr_array = net_irr_homog

    "SAVE FILE .irr"
    irr_eff = 0.85  # Irrigation efficiency assumption
    for i in range(len(net_irr_array)):  # Loop over available data
        irr100.addevent(2024, i+1, net_irr_array[i], 1, irr_eff)

    irr100.savefile(os.path.join(module_dir,'Pear_100.irr'))
    print(f"Manual parameters saved in {module_dir}")
    return net_irr_array
