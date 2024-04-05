import pandas as pd
import xarray as xr            # to read netcdf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"

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
        nc = xr.open_dataset(file_path, engine='netcdf4')
        nc = nc[selection]

        rename_var_dict = (var_information.reset_index()
                                          .loc[:,['alias', 'name']]
                                          .query('alias in @selection')
                                          .set_index('alias')
                                          .to_dict()['name'])
        nc = nc.rename(rename_var_dict)

        return nc


def compute_data_frames(yearly_df, ref_period, var='2m temperature',
                        look_back_period_short=10, look_back_period_long=30):
    """
    Compute various data frames for global warming analysis.

    Parameters:
    - yearly_df (pandas.DataFrame): DataFrame containing yearly data.
    - ref_period (tuple): Tuple containing the start and end years of the reference period.
    - var (str, optional): Variable to analyze. Defaults to '2m temperature'.
    - look_back_period_short (int, optional): Look back period for short-term analysis. Defaults to 10.
    - look_back_period_long (int, optional): Look back period for long-term analysis. Defaults to 30.

    Returns:
    - list: List containing the following data frames:
        - yearly_df_ref: DataFrame containing the mean values of the reference period.
        - yearly_anomalie_df: DataFrame containing the anomalies of the yearly data.
        - projections: Dictionary containing projections for different look back periods.
        - yearly_anom_rol10_df: DataFrame containing the rolling mean of the yearly anomalies.
    """
    
    yearly_df_ref = yearly_df.loc[ref_period[0]:ref_period[1]].mean().T
    yearly_anomalie_df = yearly_df - yearly_df_ref
    yearly_anom_rol10_df = yearly_anomalie_df.rolling(window=10).mean()

    # compute prevision for 2050
    ## 
    max_year = yearly_df.index.max()
    table_data = []
    for n in [look_back_period_short, look_back_period_long]:
            warming_rate = (yearly_anom_rol10_df.loc[max_year, var] - yearly_anom_rol10_df.loc[max_year-n, var])/n
            estimated_anomalie = warming_rate * (2050 - max_year) + yearly_anom_rol10_df.loc[max_year, var]
            table_data.append([n, warming_rate, estimated_anomalie])

    prevision_2050_df = pd.DataFrame(table_data, columns=['Years', 'Warming Rate', 'prev_anomalie'])
    prevision_2050_df[f'ref_temperature'] = yearly_df_ref[var]
    prevision_2050_df[f'prev_temperature'] = prevision_2050_df[f'ref_temperature'] + prevision_2050_df['prev_anomalie']
    prevision_2050_df.set_index('Years', inplace=True)

    ## data for the projection plot's traces 
    current_anomalie = yearly_anom_rol10_df[var].iloc[-1]
    current_year = yearly_df.index.max()

    projections : list[dict] = []
    for look_back_period in [look_back_period_short, look_back_period_long]:
            warming_rate = prevision_2050_df.loc[look_back_period, 'Warming Rate']
            years_temp = list(range(current_year - look_back_period, 2051))
            anomalies = [current_anomalie + warming_rate * (year - current_year) for year in years_temp]
            temperatures = anomalies + yearly_df_ref[var]
            projections.append({'year': years_temp, 
                                            'warming_rate':warming_rate,
                                            'anomalie': anomalies,
                                            'temperature': temperatures,
                                            'period': look_back_period,
                                            })


    return [yearly_df_ref, 
            yearly_anomalie_df,
            projections, 
            yearly_anom_rol10_df,]


def create_time_serie_fig(yearly_anomalie_df, yearly_anom_rol10_df, var, projections, title='Répartition temporelle de l anomalie de température'):
        """
        Creates a time series figure with various traces and annotations.

        Parameters:
        - yearly_anomalie_df (pandas.DataFrame): DataFrame containing yearly anomaly data.
        - yearly_anom_rol10_df (pandas.DataFrame): DataFrame containing 10-year rolling average data.
        - var (str): Variable name for the anomaly data.
        - projections (list): List of dictionaries containing projection data.

        Returns:
        - fig (plotly.graph_objects.Figure): The created time series figure.
        """
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

        # PROJECTION --------------------

        ## SHORT PERIOD PROJECTION
        fig.add_trace(go.Scatter(x=projections[0]['year'], 
                                        y=projections[0]['anomalie'], 
                                        mode='lines',
                                        line=dict(color='grey', 
                                                width=2, 
                                                dash='dot',), 
                                        name='Projection',
                                        showlegend=True,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )

        # last point
        last_point = [projections[0]['year'][-1], 
                        projections[0]['anomalie'][-1]]
        fig.add_trace(go.Scatter(x=[last_point[0]],  
                                        y=[last_point[1]],
                                        mode='markers', 
                                        marker=dict(symbol='circle',
                                                color='grey', 
                                                size=8,
                                                line_color="black",
                                                line_width=1),
                                        showlegend=False,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )   
                
        fig.add_annotation(x=last_point[0],  
                                y=last_point[1],
                                text='+' + str(round(last_point[1],1)) + '°C',
                                showarrow=False,
                                font=dict(size=15,
                                        color="grey",
                                        family="Calibri",),
                                xanchor='left',
                                yanchor='bottom',
                        )


        ## LONG PERIOD PROJECTION
        fig.add_trace(go.Scatter(x=projections[1]['year'], 
                                        y=projections[1]['anomalie'], 
                                        mode='lines',
                                        line=dict(color='grey', 
                                                width=2, 
                                                dash='dot',), 
                                        name='Projection',
                                        showlegend=False,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )

        # last point
        last_point = [projections[1]['year'][-1], 
                        projections[1]['anomalie'][-1]]
        fig.add_trace(go.Scatter(x=[last_point[0]],  
                                        y=[last_point[1]],
                                        mode='markers', 
                                        marker=dict(symbol='circle',
                                                color='grey', 
                                                size=8,
                                                line_color="black",
                                                line_width=1),
                                        showlegend=False,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )   
                
        fig.add_annotation(x=last_point[0],  
                                y=last_point[1],
                                text='+' + str(round(last_point[1],1)) + '°C',
                                showarrow=False,
                                font=dict(size=15,
                                        color="grey",
                                        family="Calibri",),
                                xanchor='left',
                                yanchor='bottom',
                        )

        fig.add_trace(go.Scatter(x=[projections[1]['year'][-1]],  
                                        y=[projections[1]['anomalie'][-1]],
                                        mode='markers', 
                                        marker=dict(symbol='circle',
                                                color='grey', 
                                                size=8,
                                                line_color="black",
                                                line_width=1),
                                        showlegend=False,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        ) 

        # -----------------------------

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
                                hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
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
                                yanchor='top',
                                )

        fig.update_layout(
                                #margin=dict(t=100),
                                xaxis_title='',
                                legend=dict(x=0, y=1,),  # moves the legend to the top left corner
                                title=title,  # Add the title to the graph
                        )

        fig.update_yaxes(range=[-1.5, 6],
                                tickformat='+',
                                title_text='Anomalie de temperature [°C]')

        fig.update_xaxes(dtick=10)

        return fig
#%% Set parameters
# st.set_page_config(layout="wide")

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
city_list = [' France'] + cities['name'].tolist()

# set default values and period
var = '2m temperature'

min_year = yearly_nc.time.min().item()
max_year = yearly_nc.time.max().item()
years = range(min_year, max_year+1)

# France df
fr_avg_df = (yearly_nc[var].mean(dim=['latitude', 'longitude'])
                                        .to_dataframe())
fr_df_ref, fr_anom_df, projections_fr, fr_anom_rol10_df = compute_data_frames(fr_avg_df, ref_period=P['ref_period'])


#%% Streamlit
st.markdown('<h2 style="font-size: 2.2rem;">Effet du réchauffement climatique en France</h2>', unsafe_allow_html=True)
st.markdown('Variations temporelles et spatiales des anomalies de température en France.')

# Anomalie time serie vizualisation

# city selection
city = st.selectbox('Sélectionnez une ville:', city_list, index=city_list.index(' France'))
# city = 'Arpajon'
selected_city = cities.query('name == @city')[['name', 'lat', 'lng']] 

if city == ' France':
        yearly_df = fr_avg_df
else:
        # selection data extraction 
        lat_sel, lng_sel = (cities.query('name == @city')
                                        .loc[:,['lat', 'lng']]
                                        .values[0]
                                        )

        yearly_df = (yearly_nc.sel(latitude=lat_sel, 
                                longitude=lng_sel, 
                                method='nearest')
                        .to_dataframe()
                        .drop(columns=['latitude', 'longitude'])
                        )

yearly_df_ref, yearly_anomalie_df, projections, yearly_anom_rol10_df = compute_data_frames(yearly_df, ref_period=P['ref_period'])

fig = create_time_serie_fig(yearly_anomalie_df, yearly_anom_rol10_df, var, projections, title=f'Anomalie de température: {city}')

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

st.markdown(f'''
**Projection 2050 pour {city}:**  
Avec un de "taux de rechauffement" annuel se situant entre {projections[1]['warming_rate']:+.3f} et {projections[0]['warming_rate']:+.3f} °C/an,
l'anomalie de temperature en 2050 devrait se situer entre **{projections[1]['anomalie'][-1]:+.1f} et {projections[0]['anomalie'][-1]:+.1f} °C**. Appliqué a la temperature de refence de {yearly_df_ref[var]:.1f} °C, cela donne pour 2050 une temperature annuelle comprise entre **{projections[1]['temperature'][-1]:.1f} et {projections[0]['temperature'][-1]:.1f} °C**.
''')  

st.markdown("&nbsp;")  # blanc area

#%% anomalie map

st.markdown(f'##### Répartition spacial de l anomalie de température')
selected_years = st.multiselect('Selectionnez des années', 
                                yearly_anom_nc['time'].values, 
                                default=[2021, 2022, 2023,]
                                )

map_sel = (yearly_anom_nc[var].sel(time=selected_years)
                              .where(mask_france)
                              .mean(dim='time'))

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

fig.patch.set_facecolor('none') # Set the background color of the figure and axes to be transparent
ax.set_facecolor('none')
ax.axis('off')

# set the colormap limits based on the data values
v_min = 3.25  # map_sel.min() + 0.15
v_max = 0     # map_sel.max() - 0.15
c_map = 'gist_heat' # 'RdGy_r' #'PuOr_r' #'hot' #RdPu'

# map
map_sel.plot(ax=ax, 
                transform=ccrs.PlateCarree(), 
                x='longitude', 
                y='latitude',
                cmap=c_map,
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
        label='_nolegend_',  # Exclude this plot from the legend
        transform=ccrs.PlateCarree(),
        )

# add citie names
for i, row in major_cities.iterrows():
        plt.text(row['lng']+0.1, row['lat']+0.1, 
                s=row['name'], 
                transform=ccrs.PlateCarree(), 
                color='green',
                size=9,
                )

ax.plot('lng', 
        'lat', 
        marker='X', 
        data=selected_city,
        color='darkgrey', 
        markersize=13, 
        markeredgewidth=1.5,
        markeredgecolor='black',
        label='_nolegend_',  # Exclude this plot from the legend
        transform=ccrs.PlateCarree(),
        )

ax.legend(fontsize='large', frameon=False)

st.pyplot(plt)  

st.markdown(f"""
**Source des données :**  
[ERA5-Land monthly average](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview), est un ensemble de données météorolgique de la surface terrestre avec une résolution de 9 km, disponible de 1950 à aujourd'hui.
Les données sont fournies par le Centre européen de prévisions météorologiques à moyen terme ([ECMWF](https://www.ecmwf.int/)) par l'intermédiaire de [Copernicus](https://www.copernicus.eu/en/about-copernicus).
  
**Remarques :**  
- La période de réference utilisé pour le calcul d'anomalie va de 1950 à 1979 (30 ans).  
- Le calcul des moyennes haute et basse du "taux de rechauffement" se fait respectivement sur la base des {projections_fr[0]['period']} et {projections_fr[1]['period']} dernieres années.
- Les données extraitent pour les villes sont issues du point de données le plus proche (distance < 4.5 km).
Cette approche est généralement fiable sur les données de température. Néamoins, cette fiablilité baisse fortement dans les zones trés montagneuses.    
""")

st.markdown('Contact: [gen1.tweezers809@passinbox.com](mailto:gen1.tweezers809@passinbox.com)')


## hack to change st.tab size
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)