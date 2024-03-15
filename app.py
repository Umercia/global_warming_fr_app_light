import pandas as pd
import xarray as xr            # to read netcdf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import streamlit as st
import netCDF4
import plotly.express as px
import plotly.graph_objects as go

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

P = {
     'cities_list_path': r'./data/cities_france.csv',
     'yearly_path': r'./data/yearly.nc',
     'yearly_ref_path': r'./data/yearly_ref.nc',
     'era5_var_info_path': r'./data/era5L_variable_description.csv',
     'var_info_path': r'./data/variable_description_processed.csv',
     'france_mask_path': r'./data/France_mask.nc',
     'var_sel' : ['t2m',
                #   'sf',
                #   'stl1',
                #   'ssrd',
                #   'e',
                #   'tp',
                #   'swvl1',
                  ],
     'ref_period': [1950, 1979],
     }

#%% load data

mask_france = load_dataarray(P['france_mask_path'])
var_inf = load_var_inf(P['var_info_path'])

yearly_nc = load_and_process_yearly_nc(file_path=P['yearly_path'],
                                       selection=P['var_sel'], 
                                       var_information=var_inf)

cities = load_cities(P['cities_list_path'])

#%% data processing

yearly_anom_nc = calculate_yearly_anomalie_nc(yearly_nc, P['ref_period'])
var_names = list(yearly_nc.data_vars.keys())
major_cities = cities[cities['capital'].isin(['primary', 'admin'])]
city_list = cities['name'].tolist()

# set default values and period
var = '2m temperature'
min_year = yearly_nc.time.min().item()
max_year = yearly_nc.time.max().item()
years = range(min_year, max_year+1)

#%% Streamlit
st.sidebar.title("Effet du réchauffement climatique en France")
st.sidebar.markdown('Variations temporelles et spatiales des anomalies de température en France.')

# side bar - mainly selection
city = st.sidebar.selectbox('Sélectionnez une ville:', city_list, index=city_list.index('Paris'))
selected_city = cities.query('name == @city')[['name', 'lat', 'lng']] 

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
yearly_rol10_df = yearly_df.rolling(window=10).mean()
yearly_df_ref = yearly_df.loc[P['ref_period'][0]:P['ref_period'][1]].mean().T
yearly_anomalie_df = yearly_df - yearly_df_ref
yearly_anom_rol10_df = yearly_anomalie_df.rolling(window=10).mean()

# prevision 2050...
n_years = [5, 10, 20, 30]
table_data = []
for n in n_years:
        warming_rate = (yearly_anom_rol10_df.loc[max_year, var] - yearly_anom_rol10_df.loc[max_year-n, var])/n
        estimated_anomalie = warming_rate * (2050 - max_year) + yearly_anom_rol10_df.loc[max_year, var]
        table_data.append([n, warming_rate, estimated_anomalie])

prevision_2050_df = pd.DataFrame(table_data, columns=['Years', 'Warming Rate', 'prev_anomalie'])
prevision_2050_df[f'ref_temperature'] = yearly_df_ref['2m temperature']
prevision_2050_df[f'prev_temperature'] = prevision_2050_df[f'ref_temperature'] + prevision_2050_df['prev_anomalie']
prevision_2050_df.set_index('Years', inplace=True)

col1, col2 = st.columns([1.4, 1])

with col1:

        #%% Anomalie time serie vizualisation

        st.markdown(f'##### Anomalie de température à {city}')
        
        fig = go.Figure()

        # main yearly bar anomalie
        fig.add_trace(go.Bar(x=yearly_anomalie_df.index, 
                             y=yearly_anomalie_df[var], 
                             marker_color=yearly_anomalie_df[var].apply(lambda x: 'lightcoral' if x > 0 else 'cornflowerblue'),
                             showlegend=False,
                             name='anomalie',
                             width=0.9,  # Adjust the width of the bars
                             customdata=np.array(yearly_df.loc[yearly_anomalie_df.index, [var]].assign(ref=yearly_df_ref[var])),
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C <br>Moyenne: %{customdata[0]:.1f} °C <br>Référence: %{customdata[1]:.1f} °C',  # Display the year, anomaly value, and yearly value in the hover tooltip
                             ),
                         )
        
        # Add a 10-year rolling average line
        fig.add_trace(go.Scatter(x=yearly_anom_rol10_df.index, 
                                 y=yearly_anom_rol10_df[var], 
                                 mode='lines', 
                                 name='Moyenne mobile 10 ans',
                                 line=dict(color='red', 
                                           width=4, 
                                           dash='solid'),
                                 hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                                )
                                
        # Get the most recent point 
        last_point = yearly_anom_rol10_df.iloc[-1]

        # Add a trace for the last point
        fig.add_trace(go.Scatter(x=[last_point.name], 
                                y=[last_point[var]], 
                                mode='markers', 
                                marker=dict(symbol='circle',
                                            color='red', 
                                            size=15,
                                            line_color="black",
                                            line_width=1),
                                showlegend=False,
                                hovertemplate='Anomalie: %{y:.1f} °C'),
                                )

        # Add an annotation for the last point
        fig.add_annotation(x=last_point.name, 
                           y=last_point[var],
                           text='+' + str(round(last_point[var],1)) + '°C',
                           showarrow=False,
                           font=dict(size=20,
                                     color="red",
                                     family="Calibri",),
                           xanchor='left',
                           yanchor='bottom',
                           )

        fig.update_layout(margin=dict(t=0),
                        xaxis_title='',
                        legend=dict(x=0, y=1,),  # moves the legend to the top left corner
                        )

        fig.update_yaxes(range=[-1.5, 3.5],
                         tickformat='+')

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

with col2:

        st.markdown(f'##### Anomalie de température sur les 10 dernieres années')
        st.markdown("&nbsp;")  # blanc area
#       %% anomalie map

        map_sel = (yearly_anom_nc[var].sel(time=slice(max_year-9, max_year))
                                      .where(mask_france)
                                      .mean(dim='time'))

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set the background color of the figure and axes to be transparent
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.axis('off')

        # set the colormap limits based on the data values
        v_min = map_sel.min() + 0.15
        v_max = map_sel.max() - 0.15
        c_map = 'gist_heat' # 'RdGy_r' #'PuOr_r' #'hot' #RdPu'
        # v_abs_max = max(abs(map_sel.min()), abs(map_sel.max()))
        # v_min = -v_abs_max
        # v_max = v_abs_max

        # if v_min*v_max < 0 : 
        #         c_map = 'bwr'
        #         v_abs_max = max(abs(map_sel.min()), abs(map_sel.max()))
        #         v_min = -v_abs_max
        #         v_max = v_abs_max
        # else: 
        #         c_map = 'inferno'

        # map
        map_sel.plot(ax=ax, 
                        transform=ccrs.PlateCarree(), 
                        x='longitude', 
                        y='latitude',
                        cmap=c_map,  # viridis, coolwarm, seismic, hot, bwr, plasma
                        cbar_kwargs={'shrink': 0.5, 
                                     'label' : 'Anomalie de température [°C]'},
                        vmin= v_min,
                        vmax= v_max,
                        )

        ax.coastlines()

        # Plot cities location
        ax.scatter('lng',
                'lat',
                data=major_cities, 
                marker='+',
                color='green',
                s=8,
                label='_nolegend_',  # This will exclude this plot from the legend
                transform=ccrs.PlateCarree(),
                )

        # add citie names
        for i, row in major_cities.iterrows():
                plt.text(row['lng']+0.1, row['lat']+0.1, 
                        s=row['name'], 
                        transform=ccrs.PlateCarree(), 
                        color='green',
                        size=9,
                        # label='_nolegend_',  # This will exclude this plot from the legend
                        )

        ax.plot('lng', 
                'lat', 
                marker='X', 
                data=selected_city,
                color='darkgrey', 
                markersize=13, 
                markeredgewidth=1.5,
                markeredgecolor='black',
                label='_nolegend_',  # This will exclude this plot from the legend
                transform=ccrs.PlateCarree(),
                )

        ax.legend(fontsize='large', frameon=False)
         
        st.pyplot(plt)  

st.markdown(f'''
**{city} 2050 - Prévision:**  
Avec un de taux de "rechauffement" annuel se situant entre {prevision_2050_df.loc[30, 'Warming Rate']:+.3f} et {prevision_2050_df.loc[5, 'Warming Rate']:+.3f} °C/an,
 l'anomalie de temperature en 2050 devrait se situer entre **{prevision_2050_df.loc[30, 'prev_anomalie']:+.1f} et {prevision_2050_df.loc[5, 'prev_anomalie']:+.1f} °C**. Appliqué a la temperature de refence de {yearly_df_ref[var]:.1f} °C, cela donne pour 2050 une temperature annuelle comprise entre **{prevision_2050_df.loc[30, 'prev_temperature']:.1f} et {prevision_2050_df.loc[5, 'prev_temperature']:.1f} °C**.
''')    
   
st.markdown("""
**Source des données :**  
[ERA5-Land monthly average](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview), est un ensemble de données météorolgique de la surface terrestre avec une résolution de 9 km, disponible de 1950 à aujourd'hui.
Les données sont fournies par le Centre européen de prévisions météorologiques à moyen terme ([ECMWF](https://www.ecmwf.int/)) par l'intermédiaire de [Copernicus](https://www.copernicus.eu/en/about-copernicus).
  
**Remarques :**  
- La période de réference utilisé pour le calcul d'anomalie va de 1950 à 1979.  
- Les données extraitent pour les villes sont issues du point de données le plus proche (distance < 4.5 km).
Cette approche est généralement fiable sur les données de température. Néamoins, cette fiablilité baisse fortement dans les zones trés montagneuses.    
""")