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
city_list = cities['name'].tolist()

# set default values and period
var = '2m temperature'
min_year = yearly_nc.time.min().item()
max_year = yearly_nc.time.max().item()
years = range(min_year, max_year+1)

# France df

fr_avg_df = yearly_nc['2m temperature'].mean(dim=['latitude', 'longitude']).to_dataframe()
fr_df_ref = fr_avg_df.loc[P['ref_period'][0]:P['ref_period'][1]].mean().T
fr_anom_df = (yearly_anom_nc['2m temperature'].mean(dim=['latitude', 'longitude'])
                                              .to_dataframe())
fr_anom_rol10_df = fr_anom_df.rolling(window=10).mean()

n_years = [5, 10, 20, 30]
table_data = []
for n in n_years:
        warming_rate = (fr_anom_rol10_df .loc[max_year, var] - fr_anom_rol10_df .loc[max_year-n, var])/n
        estimated_anomalie = warming_rate * (2050 - max_year) + fr_anom_rol10_df .loc[max_year, var]
        table_data.append([n, warming_rate, estimated_anomalie])

fr_prev_2050_df = pd.DataFrame(table_data, columns=['Years', 'Warming Rate', 'prev_anomalie'])
fr_prev_2050_df[f'ref_temperature'] = fr_df_ref['2m temperature']
fr_prev_2050_df[f'prev_temperature'] = fr_prev_2050_df[f'ref_temperature'] + fr_prev_2050_df['prev_anomalie']
fr_prev_2050_df.set_index('Years', inplace=True)

## data for the projection plot's traces 
current_anomalie = fr_anom_rol10_df ['2m temperature'].iloc[-1]
current_year = fr_avg_df.index.max()

period_length_short = n_years[1] 
period_length_long = n_years[3] 
projections_fr : dict[dict] = {}
for period_length in [period_length_short, period_length_long]:
        warming_rate = fr_prev_2050_df.loc[period_length, 'Warming Rate']
        years_temp = list(range(current_year - period_length, 2051))
        anomalies = [current_anomalie + warming_rate * (year - current_year) for year in years_temp]
        temperatures = anomalies + fr_df_ref['2m temperature']
        projections_fr[period_length]= {'year': years_temp, 
                                'warming_rate':warming_rate,
                                'anomalie': anomalies,
                                'temperature': temperatures,
                                }


#%% Streamlit
st.title("Effet du réchauffement climatique en France", )
st.markdown('Variations temporelles et spatiales des anomalies de température en France.')

 
# col1, col2 = st.columns([1.4, 1])
tab1, tab2 = st.tabs(["📈 **Villes**", ":flag-fr: **France**"], )

with tab1:

        #%% Anomalie time serie vizualisation

        # city selection
        city = st.selectbox('Sélectionnez une ville:', city_list, index=city_list.index('Arpajon'))
        # city = 'Arpajon'
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

        # compute reference and rolling means for selected city
        yearly_rol10_df = yearly_df.rolling(window=10).mean()
        yearly_df_ref = yearly_df.loc[P['ref_period'][0]:P['ref_period'][1]].mean().T
        yearly_anomalie_df = yearly_df - yearly_df_ref
        yearly_anom_rol10_df = yearly_anomalie_df.rolling(window=10).mean()

        # compute prevision for 2050
        ## 
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

        ## data for the projection plot's traces 
        current_anomalie = yearly_anom_rol10_df['2m temperature'].iloc[-1]
        current_year = yearly_df.index.max()

        period_length_short = n_years[1] 
        period_length_long = n_years[3] 
        projections : dict[dict] = {}
        for period_length in [period_length_short, period_length_long]:
                warming_rate = prevision_2050_df.loc[period_length, 'Warming Rate']
                years_temp = list(range(current_year - period_length, 2051))
                anomalies = [current_anomalie + warming_rate * (year - current_year) for year in years_temp]
                temperatures = anomalies + yearly_df_ref['2m temperature']
                projections[period_length]= {'year': years_temp, 
                                        'warming_rate':warming_rate,
                                        'anomalie': anomalies,
                                        'temperature': temperatures,
                                        }

        # st.markdown(f'##### Anomalie de température à {city}')
        
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
        fig.add_trace(go.Scatter(x=projections[period_length_short]['year'], 
                                 y=projections[period_length_short]['anomalie'], 
                                 mode='lines',
                                 line=dict(color='grey', 
                                           width=2, 
                                           dash='dot',), 
                                 name='Projection',
                                 showlegend=True,
                                 hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )
        
        # last point
        last_point = [projections[period_length_short]['year'][-1], 
                      projections[period_length_short]['anomalie'][-1]]
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
        fig.add_trace(go.Scatter(x=projections[period_length_long]['year'], 
                                 y=projections[period_length_long]['anomalie'], 
                                 mode='lines',
                                 line=dict(color='grey', 
                                           width=2, 
                                           dash='dot',), 
                                 name='Projection',
                                 showlegend=False,
                                 hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )
        
        # last point
        last_point = [projections[period_length_long]['year'][-1], 
                      projections[period_length_long]['anomalie'][-1]]
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

        fig.add_trace(go.Scatter(x=[projections[period_length_long]['year'][-1]],  
                                 y=[projections[period_length_long]['anomalie'][-1]],
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

        fig.update_layout(margin=dict(t=0),
                          xaxis_title='',
                          legend=dict(x=0, y=1,),  # moves the legend to the top left corner
                        )

        fig.update_yaxes(range=[-1.5, 6],
                         tickformat='+',
                         title_text='Anomalie de temperature [°C]')
        
        fig.update_xaxes(dtick=10)

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

        st.markdown(f'''
        **Projection 2050 pour {city}:**  
        Avec un de "taux de rechauffement" annuel se situant entre {projections[period_length_long]['warming_rate']:+.3f} et {projections[period_length_short]['warming_rate']:+.3f} °C/an,
        l'anomalie de temperature en 2050 devrait se situer entre **{projections[period_length_long]['anomalie'][-1]:+.1f} et {projections[period_length_short]['anomalie'][-1]:+.1f} °C**. Appliqué a la temperature de refence de {yearly_df_ref[var]:.1f} °C, cela donne pour 2050 une temperature annuelle comprise entre **{projections[period_length_long]['temperature'][-1]:.1f} et {projections[period_length_short]['temperature'][-1]:.1f} °C**.
        ''')  

with tab2:

        st.markdown(f'##### Anomalie de température sur les 10 dernieres années')
        st.markdown("&nbsp;")  # blanc area

#%% plot temp
        fig_fr = go.Figure()

        # main yearly bar anomalie
        fig_fr.add_trace(go.Bar(x=fr_anom_df.index, 
                                y=fr_anom_df[var], 
                                marker_color=fr_anom_df[var].apply(lambda x: 'lightcoral' if x > 0 else 'cornflowerblue'),
                                showlegend=False,
                                name='anomalie',
                                width=0.9,  # Adjust the width of the bars
                                customdata=np.array(fr_avg_df.loc[fr_anom_df.index, [var]].assign(ref=fr_df_ref[var])),
                                hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C <br>Moyenne: %{customdata[0]:.1f} °C <br>Référence: %{customdata[1]:.1f} °C',  # Display the year, anomaly value, and yearly value in the hover tooltip
                                ),
                                )

# PROJECTION --------------------

        ## SHORT PERIOD PROJECTION
        fig_fr.add_trace(go.Scatter(x=projections_fr[period_length_short]['year'], 
                                        y=projections_fr[period_length_short]['anomalie'], 
                                        mode='lines',
                                        line=dict(color='grey', 
                                                width=2, 
                                                dash='dot',), 
                                        name='Projection',
                                        showlegend=True,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )

        # last point
        last_point = [projections_fr[period_length_short]['year'][-1], 
                        projections_fr[period_length_short]['anomalie'][-1]]
        fig_fr.add_trace(go.Scatter(x=[last_point[0]],  
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
                
        fig_fr.add_annotation(x=last_point[0],  
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
        fig_fr.add_trace(go.Scatter(x=projections_fr[period_length_long]['year'], 
                                        y=projections_fr[period_length_long]['anomalie'], 
                                        mode='lines',
                                        line=dict(color='grey', 
                                                width=2, 
                                                dash='dot',), 
                                        name='Projection',
                                        showlegend=False,
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                        )

        # last point
        last_point = [projections_fr[period_length_long]['year'][-1], 
                        projections_fr[period_length_long]['anomalie'][-1]]
        fig_fr.add_trace(go.Scatter(x=[last_point[0]],  
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
                
        fig_fr.add_annotation(x=last_point[0],  
                                y=last_point[1],
                                text='+' + str(round(last_point[1],1)) + '°C',
                                showarrow=False,
                                font=dict(size=15,
                                        color="grey",
                                        family="Calibri",),
                                xanchor='left',
                                yanchor='bottom',
                        )

        fig_fr.add_trace(go.Scatter(x=[projections_fr[period_length_long]['year'][-1]],  
                                        y=[projections_fr[period_length_long]['anomalie'][-1]],
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
        fig_fr.add_trace(go.Scatter(x=fr_anom_rol10_df.index, 
                                        y=fr_anom_rol10_df[var], 
                                        mode='lines', 
                                        name='Moyenne mobile 10 ans',
                                        line=dict(color='red', 
                                                width=4, 
                                                dash='solid'),
                                        hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'),
                                )
                                
        # Get the most recent point 
        last_point = fr_anom_rol10_df.iloc[-1]

        # Add a trace for the last point
        fig_fr.add_trace(go.Scatter(x=[last_point.name], 
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
        fig_fr.add_annotation(x=last_point.name, 
                                y=last_point[var],
                                text='+' + str(round(last_point[var],1)) + '°C',
                                showarrow=False,
                                font=dict(size=20,
                                        color="red",
                                        family="Calibri",),
                                xanchor='left',
                                yanchor='top',
                                )

        fig_fr.update_layout(margin=dict(t=0),
                                xaxis_title='',
                                legend=dict(x=0, y=1,),  # moves the legend to the top left corner
                        )

        fig_fr.update_yaxes(range=[-1.5, 6],
                                tickformat='+',
                                title_text='Anomalie de temperature [°C]')

        fig_fr.update_xaxes(dtick=10)

        st.plotly_chart(fig_fr, use_container_width=True, config={'displayModeBar': False}) 



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
        **Projection 2050 pour France:**  
        Avec un de "taux de rechauffement" annuel se situant entre {projections_fr[period_length_long]['warming_rate']:+.3f} et {projections_fr[period_length_short]['warming_rate']:+.3f} °C/an,
        l'anomalie de temperature en 2050 devrait se situer entre **{projections_fr[period_length_long]['anomalie'][-1]:+.1f} et {projections_fr[period_length_short]['anomalie'][-1]:+.1f} °C**. Appliqué a la temperature de refence de {fr_df_ref[var]:.1f} °C, cela donne pour 2050 une temperature annuelle comprise entre **{projections_fr[period_length_long]['temperature'][-1]:.1f} et {projections_fr[period_length_short]['temperature'][-1]:.1f} °C**.
        ''')

  

st.markdown(f"""
**Source des données :**  
[ERA5-Land monthly average](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview), est un ensemble de données météorolgique de la surface terrestre avec une résolution de 9 km, disponible de 1950 à aujourd'hui.
Les données sont fournies par le Centre européen de prévisions météorologiques à moyen terme ([ECMWF](https://www.ecmwf.int/)) par l'intermédiaire de [Copernicus](https://www.copernicus.eu/en/about-copernicus).
  
**Remarques :**  
- La période de réference utilisé pour le calcul d'anomalie va de 1950 à 1979 (30 ans).  
- Le calcul des moyennes haute et basse du "taux de rechauffement" se fait respectivement sur la base des {period_length_short} et {period_length_long} dernieres années.
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