
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['legend.facecolor'] = 'darkgrey'
plt.rcParams['legend.edgecolor'] = 'white'

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


def create_time_serie_fig(yearly_anomalie_df, yearly_df, yearly_df_ref, yearly_anom_rol10_df, var, projections, title='Répartition temporelle de l anomalie de température'):
    """
    Creates a time series figure with various traces and annotations.

    Parameters:
    - yearly_anomalie_df (pandas.DataFrame): DataFrame containing yearly anomaly data.
    - yearly_df (pandas.DataFrame): DataFrame containing yearly data.
    - yearly_df_ref (pandas.DataFrame): DataFrame containing reference yearly data.
    - yearly_anom_rol10_df (pandas.DataFrame): DataFrame containing 10-year rolling average data.
    - var (str): Variable name for the anomaly data.
    - projections (list): List of dictionaries containing projection data.
    - title (str): Title of the figure. Default is 'Répartition temporelle de l anomalie de température'.

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
                         hovertemplate=
                         '''Année: %{x} <br>Anomalie: %{y:.1f} °C <br>Moyenne: %{customdata[0]:.1f} °C <br>Référence: %{customdata[1]:.1f} °C''',
                        ),
                )

    # PROJECTION --------------------

    ## add scenario 1 projection for 2050 
    fig.add_trace(go.Scatter(x=projections[0]['year'], 
                             y=projections[0]['anomalie'], 
                             mode='lines',
                             line=dict(color='grey', 
                                       width=2, 
                                       dash='dot',), 
                             name='Projection',
                             showlegend=True,
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                             ),
                    )

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
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                             ),
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


    # add scenario 2 projection for 2050 
    fig.add_trace(go.Scatter(x=projections[1]['year'], 
                             y=projections[1]['anomalie'], 
                             mode='lines',
                             line=dict(color='grey', 
                                       width=2, 
                                       dash='dot',), 
                             name='Projection',
                             showlegend=False,
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                             ),
                   )

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
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                            ),
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

    # Add a 10-year rolling average line
    fig.add_trace(go.Scatter(x=yearly_anom_rol10_df.index, 
                             y=yearly_anom_rol10_df[var], 
                             mode='lines', 
                             name='Moyenne mobile 10 ans',
                             line=dict(color='red', 
                                       width=4, 
                                       dash='solid'),
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                             ),
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
                             hovertemplate='Année: %{x} <br>Anomalie: %{y:.1f} °C'
                             ),
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

    fig.update_layout(margin=dict(t=30),
                      xaxis_title='',
                      legend=dict(x=0, y=1,),  # moves the legend to the top left corner
                    )

    fig.update_yaxes(range=[-1.5, 6],
                     tickformat='+',
                     title_text='Anomalie de temperature [°C]'
                    )

    fig.update_xaxes(dtick=10)

    return fig


def create_map_fig(map_sel, major_cities, selected_city):
    """
    Creates a map figure using matplotlib and cartopy.

    Parameters:
    map_sel (GeoDataFrame): A GeoDataFrame containing the map data to be plotted.
    major_cities (DataFrame): A DataFrame containing the longitude and latitude of major cities to be plotted.
    selected_city (DataFrame): A DataFrame containing the longitude and latitude of a selected city to be highlighted on the map.

    Returns:
    fig (Figure): A matplotlib Figure object containing the created map.

    The function first creates a figure and axes with a PlateCarree projection. It then plots the map data from map_sel onto the axes.
    The colormap limits are set based on the data values. The major cities are plotted as green '+' markers, and their names are added next to the markers.
    The selected city is highlighted with a 'X' marker. The function finally adds a legend and returns the created figure.
    """

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    fig.patch.set_facecolor('none') # Set the background color of the figure and axes to be transparent
    ax.set_facecolor('none')
    ax.axis('off')

    # set the colormap limits based on the data values
    v_min = 3.25  # map_sel.min() + 0.15
    v_max = 0.5     # map_sel.max() - 0.15
    c_map = 'gist_heat' # 'RdGy_r' #'PuOr_r' #'hot' #RdPu'

    # plot map
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

    # add cities location
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

    # add selected city location
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

    return fig