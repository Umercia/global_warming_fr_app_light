import pandas as pd
import xarray as xr            # to read netcdf
import numpy as np
import streamlit as st
from local_functions import compute_data_frames, create_time_serie_fig, create_map_fig


#%% cached functions
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
        nc = xr.open_dataset(file_path, engine='netcdf4')
        nc = nc[selection]

        rename_var_dict = (var_information.reset_index()
                                          .loc[:,['alias', 'name']]
                                          .query('alias in @selection')
                                          .set_index('alias')
                                          .to_dict()['name'])
        nc = nc.rename(rename_var_dict)

        return nc

#%% Set parameters

P = {
     'cities_list_path': r'./data/cities_france.csv',
     'yearly_path': r'./data/yearly.nc',
     'yearly_ref_path': r'./data/yearly_ref.nc',
     'era5_var_info_path': r'./data/era5L_variable_description.csv',
     'var_info_path': r'./data/variable_description_processed.csv',
     'france_mask_path': r'./data/France_mask.nc',
     'var_sel' : ['t2m',],       #'sf','stl1','ssrd','e','tp','swvl1',
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
major_cities = cities[cities['capital'].isin(['primary', 'admin'])]
city_list = [' France'] + cities['name'].tolist()

# set default values and period
var = '2m temperature'

min_year = yearly_nc.time.min().item()
max_year = yearly_nc.time.max().item()
years = range(min_year, max_year+1)

# France df
fr_avg_df = (yearly_nc[var].mean(dim=['latitude', 'longitude'])
                           .to_dataframe())

#%% Streamlit
st.markdown('<h2 style="font-size: 2.2rem;">Effet du réchauffement climatique en France</h2>', unsafe_allow_html=True)
st.markdown("Variations temporelles et spatiales de l'anomalie de température (écart à la normalité).")
st.markdown("&nbsp;") 

#%% Anomalie time serie vizualisation

st.markdown(f"##### Répartition temporelle de l'anomalie de température")
city = st.selectbox('', city_list, index=city_list.index(' France'))
selected_city = cities.query('name == @city')[['name', 'lat', 'lng']] 

if city == ' France':
        yearly_df = fr_avg_df
else:
        lat_sel, lng_sel = (cities.query('name == @city')
                                  .loc[:,['lat', 'lng']]
                                  .values[0])

        yearly_df = (yearly_nc.sel(latitude=lat_sel, 
                                   longitude=lng_sel, 
                                   method='nearest')
                              .to_dataframe()
                              .drop(columns=['latitude', 'longitude']))

yearly_df_ref, yearly_anomalie_df, projections, yearly_anom_rol10_df = compute_data_frames(yearly_df, ref_period=P['ref_period'])

fig = create_time_serie_fig(yearly_anomalie_df, 
                            yearly_df, 
                            yearly_df_ref,
                            yearly_anom_rol10_df, 
                            var, 
                            projections, 
                            )

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

st.markdown(f'''
**Projection 2050 pour {city}:**  
Avec un de "taux de rechauffement" annuel se situant entre {projections[1]['warming_rate']:+.3f} et {projections[0]['warming_rate']:+.3f} °C/an,
l'anomalie de temperature en 2050 devrait se situer entre **{projections[1]['anomalie'][-1]:+.1f} et {projections[0]['anomalie'][-1]:+.1f} °C**. Appliqué a la temperature de refence de {yearly_df_ref[var]:.1f} °C, cela donne pour 2050 une temperature annuelle comprise entre **{projections[1]['temperature'][-1]:.1f} et {projections[0]['temperature'][-1]:.1f} °C**.
''')  

st.markdown("&nbsp;")  # blanc area

#%% anomalie map

st.markdown(f"##### Répartition spacial de l'anomalie de température")
selected_years = st.multiselect('', 
                                yearly_anom_nc['time'].values, 
                                default=[2023,]
                                )

map_sel = (yearly_anom_nc[var].sel(time=selected_years)
                              .where(mask_france)
                              .mean(dim='time'))

fig = create_map_fig(map_sel, major_cities, selected_city)
st.pyplot(fig)  

st.markdown(f"""
**Source des données :**  
[ERA5-Land monthly average](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview), est un ensemble de données météorolgique de la surface terrestre avec une résolution de 9 km, disponible de 1950 à aujourd'hui.
Les données sont fournies par le Centre européen de prévisions météorologiques à moyen terme ([ECMWF](https://www.ecmwf.int/)) par l'intermédiaire de [Copernicus](https://www.copernicus.eu/en/about-copernicus).
  
**Remarques :**  
- La période de réference utilisé pour le calcul d'anomalie va de 1950 à 1979 (30 ans).  
- Le calcul des moyennes haute et basse du "taux de rechauffement" se fait respectivement sur la base des {projections[0]['period']} et {projections[1]['period']} dernieres années.
- Les données extraitent pour les villes sont issues du point de données le plus proche (distance < 4.5 km).
Cette approche est généralement fiable sur les données de température. Néamoins, cette fiablilité baisse fortement dans les zones trés montagneuses.    
""")

st.markdown('Contact: [gen1.tweezers809@passinbox.com](mailto:gen1.tweezers809@passinbox.com)')
