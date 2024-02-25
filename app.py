import pandas as pd
import xarray as xr            # to read netcdf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import streamlit as st
import netCDF4

#%% lib
@st.cache_data
def load_netcdf(file_path, selection):
        nc = xr.open_dataset(file_path, engine='netcdf4')  # 
        nc = nc[selection]
        return nc

@st.cache_data
def load_dataarray(file_path):
        dat_ar = xr.open_dataarray(file_path, )
        return dat_ar

@st.cache_data
def load_cities(file_path):
    return pd.read_csv(file_path, sep=';')

@st.cache_data
def load_var_inf(file_path):
    var_inf = pd.read_csv(file_path, sep=';', index_col='name')
    var_inf = var_inf.replace(np.nan, None)
    return var_inf

@st.cache_data
def calculate_yearly_anomalie_nc(_yearly_nc, ref_period):
    yearly_ref_nc = (_yearly_nc.sel(time=slice(f"{ref_period[0]}-01-01", 
                                               f"{ref_period[1]}-12-31"))
                             .mean(dim='time'))
    
    yearly_anom_nc = _yearly_nc - yearly_ref_nc

    return yearly_anom_nc

@st.cache_data
def load_and_process_yearly_nc(file_path, selection, var_information):
        nc = xr.open_dataset(file_path, engine='netcdf4')  # 
        nc = nc[selection]

        rename_var_dict = (var_information.reset_index()
                                          .loc[:,['alias', 'name']]
                                          .query('alias in @selection')
                                          .set_index('alias')
                                          .to_dict()['name'])
        nc = nc.rename(rename_var_dict)

        return nc

#%% Set parameters
st.set_page_config(layout="wide")

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['legend.facecolor'] = 'darkgrey'
plt.rcParams['legend.edgecolor'] = 'white'

# ROOT_DIR = r'/home/mox/script/global_warming_fr_app'
# os.chdir(ROOT_DIR)
P = {
     'cities_list_path': r'./data/cities_france.csv',
     'yearly_path': r'./data/yearly.nc',
     'yearly_ref_path': r'./data/yearly_ref.nc',
     'era5_var_info_path': r'./data/era5L_variable_description.csv',
     'var_info_path': r'./data/variable_description_processed.csv',
     'france_mask_path': r'./data/France_mask.nc',
     'var_sel' : ['t2m',
                  'sf',
                  'stl1',
                  'ssrd',
                  'e',
                  'tp',
                  'swvl1',],
     'ref_period': [1950, 1979],
     }

#%% load data

mask_france = load_dataarray(P['france_mask_path'])
var_inf = load_var_inf(P['var_info_path'])
# yearly_nc = load_netcdf(P['yearly_path'], selection=P['var_sel'])

yearly_nc = load_and_process_yearly_nc(file_path=P['yearly_path'],
                                       selection=P['var_sel'], 
                                       var_information=var_inf)

cities = load_cities(P['cities_list_path'])

#%% data processing

# Use the cached function to calculate yearly_ref_nc
yearly_anom_nc = calculate_yearly_anomalie_nc(yearly_nc, P['ref_period'])

# Extract data variable names from netcdf
var_names = list(yearly_nc.data_vars.keys())

# get major country cities
major_cities = cities[cities['capital'].isin(['primary', 'admin'])]

# Get the list of cities from the 'cities' dataframe
city_list = cities['name'].tolist()


#%% Streamlit

st.sidebar.title("Global Warming France")
st.sidebar.markdown('Display selected variable annual values and anomalies compared to the reference period from 1950 to 1979.')

# side bar - mainly selection
city = st.sidebar.selectbox('Select a city:', city_list)
var = st.sidebar.selectbox('Select a feature:', var_names)

st.sidebar.markdown(f'**Units:** {var_inf.loc[var, "unit"]}')
st.sidebar.markdown(f'**Description:** {var_inf.loc[var, "description_short"]}', )

min_year = yearly_nc.time.min().item()
max_year = yearly_nc.time.max().item()
from_year, to_year = st.sidebar.slider('Select a period:', min_year, max_year, (max_year-5, max_year))

# selection data extraction 
lat_sel, lng_sel = (cities.query('name == @city')
                          .loc[:,['lat', 'lng']]
                          .values[0]
                          )

yearly_df = (yearly_nc.sel(latitude=lat_sel, 
                           longitude=lng_sel, 
                           method='nearest')
                       .to_dataframe()
                       .drop(columns=['latitude', 'longitude']))

# precessing reference and rolling means
yearly_rol5_df = yearly_df.rolling(window=5).mean()
yearly_rol10_df = yearly_df.rolling(window=10).mean()

yearly_df_ref = yearly_df.loc[P['ref_period'][0]:P['ref_period'][1]].mean().T

yearly_anomalie_df = yearly_df - yearly_df_ref
yearly_anom_rol5_df = yearly_anomalie_df.rolling(window=5).mean()
yearly_anom_rol10_df = yearly_anomalie_df.rolling(window=10).mean()


col1, col2 = st.columns([1.33, 1])

with col1:

        #%% value time serie vizualisation
        # plt.figure(figsize=(14, 6))

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()  # Get the current axes

        # Set the background color of the figure and axes to be transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        # ax.axis('off')
        for spine in ax.spines.values():
                spine.set_visible(False)

        # Set labels and title
        plt.xlabel('Year')

        # plt.suptitle(f'{city} ({lat_sel}°, {lng_sel}°)', size=24)
        plt.plot(yearly_df.index, 
                yearly_df[var], 
                color='red', 
                label=var, 
                linewidth=3,
                alpha=0.7)

        plt.grid(color='dimgrey', linestyle=':')

        if var_inf.loc[var, 'min'] != None:
                plt.ylim(var_inf.loc[var, 'min'], 
                         var_inf.loc[var, 'max'])
        

        # Plot the rolling mean as a line plot
        plt.plot(yearly_rol5_df.index, 
                yearly_rol5_df[var], 
                color='white', 
                linestyle='--', 
                marker='o', 
                label='5-Year Rolling Mean', 
                alpha=0.6)


        # Plot the rolling mean as a line plot
        plt.plot(yearly_rol10_df.index, 
                yearly_rol10_df[var], 
                color='white', 
                linestyle='--', 
                marker='+', 
                label='10-Year Rolling Mean', 
                alpha=0.3)

        # Set labels and title
        plt.xlabel('Year')
        plt.ylabel(f'{var} [{var_inf.loc[var, "unit"]}]')
        plt.title(f'Annual "{var}" at {city}', fontsize=14)

        # Set the legend background color to be transparent
        legend = plt.legend(loc='upper left')
        legend.get_frame().set_alpha(0)

        # Use Streamlit's st.pyplot() to display the plot
        st.pyplot(plt)

        # Clear the current figure so it doesn't interfere with the next plot
        plt.clf()

        #%% Anomalie time serie vizualisation
        # Initialize the matplotlib figure
        # plt.figure(figsize=(14, 6))

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()  # Get the current axes

        # Set the background color of the figure and axes to be transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        for spine in ax.spines.values():
                spine.set_visible(False)

        # Plot the 'anomalie' column as a bar plot
        plt.bar(yearly_anomalie_df.index, 
                yearly_anomalie_df[var], 
                color=yearly_anomalie_df[var].apply(lambda x: 'blue' if x < 0 else 'red'),
                alpha=0.6, 
                label='yearly anomalie',
                width=1,
                # Set the y-axis minimum and maximum values
                
                )
        if var_inf.loc[var, 'anom_min'] != None:    
                plt.ylim(var_inf.loc[var, 'anom_min'], 
                         var_inf.loc[var, 'anom_max'])
        
        # Plot the rolling mean as a line plot
        plt.plot(yearly_anom_rol5_df.index, 
                yearly_anom_rol5_df[var], 
                color='white', 
                linestyle='--', 
                marker='o', 
                label='5-Year Rolling Mean', 
                alpha=0.6)


        # Plot the rolling mean as a line plot
        plt.plot(yearly_anom_rol10_df.index, 
                yearly_anom_rol10_df[var], 
                color='white', 
                linestyle='--', 
                marker='+', 
                label='10-Year Rolling Mean', 
                alpha=0.3)

        # Add values above bars
        for index, value in enumerate(yearly_anomalie_df[var]):
                plt.text(index + yearly_anomalie_df.index[0], value, 
                        f'{value:.1f}', 
                        color='white', 
                        ha="center", 
                        va='bottom' if value >= 0 else 'top',
                        fontsize=8)


        # Set labels and title
        plt.xlabel('Year')
        plt.ylabel(f'Anomaly: {var} [{var_inf.loc[var, "unit"]}]')
        plt.title(f'Annual "{var}" anomalies at {city}', fontsize=14)
        plt.grid(color='dimgrey', linestyle=':')

        # Add legend
        #get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        #specify order of items in legend
        order = [2,0,1]
        #add legend to plot
        # Set the legend background color to be transparent
        legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        legend.get_frame().set_alpha(0)
       
        st.pyplot(plt) 

with col2:

        #%% value map
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set the background color of the figure and axes to be transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.axis('off')

        (yearly_nc[var].sel(time=slice(from_year, to_year))
                       .where(mask_france)
                       .mean(dim='time')
                       .plot(ax=ax, 
                                transform=ccrs.PlateCarree(), 
                                x='longitude', 
                                y='latitude',
                                cmap="hot",  # viridis, coolwarm, seismic, hot
                                cbar_kwargs={'shrink': 0.5,
                                             'label' : f'{var} [{var_inf.loc[var, "unit"]}]'},
                                vmin= var_inf.loc[var, 'min'],
                                vmax= var_inf.loc[var, 'max'],
                        ))

        ax.coastlines()

        # Plot cities
        ax.plot('lng', 
                'lat', 
                '+',
                data=major_cities,
                marker='+', 
                color='dimgrey', 
                markersize=8, 
                label='name',
                transform=ccrs.PlateCarree(),
                )

        for i, row in major_cities.iterrows():
                plt.text(row['lng']+0.1, row['lat']+0.1, s=row['name'], 
                        transform=ccrs.PlateCarree(), color='dimgrey',
                        size=8)

        selected_city = cities.query('name == @city')[['name', 'lat', 'lng']]  
        ax.plot('lng', 
                'lat', 
                '+',
                data=selected_city,
                marker='X', 
                color='blue', 
                markersize=9, 
                label='name',
                transform=ccrs.PlateCarree(),
                )

        plt.text(selected_city['lng'], selected_city['lat'], s=selected_city['name'].values[0], 
                        transform=ccrs.PlateCarree(),
                        size=9,
                        weight='bold',
                        color='blue'  # Set the marker color to red
                        )

        plt.title(f'"{var}" yearly average from {from_year} to {to_year}', fontsize=12)
        st.pyplot(plt)

#       %% anomalie map

        map_sel = (yearly_anom_nc[var].sel(time=slice(from_year, to_year))
                                      .where(mask_france)
                                      .mean(dim='time'))

        fig = plt.figure(figsize=(10, 6))
        # Create an axes object with the specified projection
        # The rect parameter [left, bottom, width, height] can be adjusted as needed
        # ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Set the background color of the figure and axes to be transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.axis('off')

        # set the colormap limits based on the data values
        v_min = map_sel.min()
        v_max = map_sel.max()

        if v_min*v_max < 0 : 
                c_map = 'bwr'
                v_abs_max = max(abs(map_sel.min()), abs(map_sel.max()))
                v_min = -v_abs_max
                v_max = v_abs_max
        else: 
                c_map = 'inferno'


        # map
        map_sel.plot(ax=ax, 
                        transform=ccrs.PlateCarree(), 
                        x='longitude', 
                        y='latitude',
                        cmap=c_map,  # viridis, coolwarm, seismic, hot, bwr, plasma
                        cbar_kwargs={'shrink': 0.5, 
                                     'label' : f'Anomaly: {var} [{var_inf.loc[var, "unit"]}]'},
                        vmin= v_min,
                        vmax= v_max,
                        )

        ax.coastlines()

        # Plot cities
        ax.plot('lng', 
                'lat', 
                '+',
                data=major_cities,
                marker='+', 
                color='dimgrey', 
                markersize=8, 
                label='name',
                transform=ccrs.PlateCarree(),
                )

        for i, row in major_cities.iterrows():
                plt.text(row['lng']+0.1, row['lat']+0.1, s=row['name'], 
                        transform=ccrs.PlateCarree(), color='dimgrey',
                        size=8)
        ax.plot('lng', 
                'lat', 
                '+',
                data=selected_city,
                marker='X', 
                color='blue', 
                markersize=9, 
                label='name',
                transform=ccrs.PlateCarree(),
                )

        plt.text(selected_city['lng'], selected_city['lat'], s=selected_city['name'].values[0], 
                        transform=ccrs.PlateCarree(),
                        size=9,
                        weight='bold',
                        color='blue'  # Set the marker color to red
                        )

        plt.title(f'"{var}" anomaly from {from_year} to {to_year}', fontsize=12)
        
        st.pyplot(plt)  

st.markdown("""**Data source:** [ERA5-Land monthly average](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview), a global land-surface dataset at 9 km resolution, available from 1950 to present. 
The data is provided through [Copernicus](https://www.copernicus.eu/en/about-copernicus) by the European Centre for Medium-Range Weather Forecasts [ECMWF](https://www.ecmwf.int/).  
**Note:** Given that this is a grid-based model, the data values for cities are derived from the nearest grid point. While this approach is generally effective, it may result in substantial discrepancies in mountainous regions."""
)