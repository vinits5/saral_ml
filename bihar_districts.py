import os
import csv
import json
import requests
import pandas as pd
import plotly.express as px

import numpy as np
from pandas import DataFrame
import plotly.io as pio
import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

online_data = True

if online_data:
	filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/india/district/india_district.json'
	resp = requests.get(filename)
	json_data = json.loads(resp.text)
	filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/br.json'
	resp = requests.get(filename)
	json_data_village = json.loads(resp.text)
else:
	file = open('datasets/india/district/india_district.geojson', 'r')
	json_data = json.load(file)
	file = open('datasets/br.json', 'r')
	json_data_village = json.load(file)

states = [dd['properties']['NAME_1'] for dd in json_data['features']]
districts = [dd['properties']['NAME_2'] for dd in json_data['features']]

districts_village = []
villages = []
for dd in json_data_village['features']:
	districts_village.append(dd['properties']['DISTRICT'])
	villages.append(dd['properties']['NAME'])

def crop(json_data_village, select_districts):
	select_districts = [select_districts]		# Temp. sol.
	updated_data = {}

	updated_data['type'] = 'FeatureCollection'
	updated_data['features'] = []

	idxs = []
	for select_district in select_districts:
		idxs_temp = np.where(np.array(districts_village) == select_district)
		idxs.append(idxs_temp[0])
	
	idxs = np.concatenate(idxs)
	colors = []
	chosen_villages = []

	for idx in idxs:
		updated_data['features'].append(json_data_village['features'][idx])
		colors.append(np.random.random())
		chosen_villages.append(villages[idx])

	return updated_data, colors, chosen_villages

idx = np.where(np.array(states) == 'Bihar')
bihar_districts = np.array(districts)[idx[0]]
random = np.array([0 for _ in range(bihar_districts.shape[0])])

data = {'districts': bihar_districts, 'random': random}
df = DataFrame(data)
districts_map = go.Figure(go.Choroplethmapbox(customdata=df,
							geojson=json_data,
							featureidkey='properties.NAME_2',
							locations=df['districts'],
							# text = df_state['color'],
							z=df['random'],
							# hovertemplate = '<b>Village</b>: <b>%{customdata[5]}</b>'+
											# '<br><b>Population</b>: <b>%{customdata[9]}</b><br>'+
											# '<b>District</b>: <b>%{customdata[3]}</b>',
							marker_opacity=0.5,
							# color=self.df_india.variable,
							# colorscale=[[0.0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# colorscale=[[0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# hoverinfo=hover_data,
							))

districts_map.update_layout(mapbox_style="carto-positron",
					mapbox_zoom=7, mapbox_center = {"lat": 25.856, "lon": 85.786}, 
					height=550, width=1200,
					title='Sanitary Pads Machine App', title_font_size=20,
					margin_l=160, margin_r=0, margin_t=40, margin_b=0)

app = Dash(__name__)

app.layout = html.Div([
			html.Div([
				dcc.Graph(id='map', style={'width': '120'}, figure={}),
			], style={'width': '100%', 'float': 'center', 'display': 'inline-block'}),
			html.Div([html.P(""), 
					 ], style={'width': '40%', 'display': 'inline-block'}),
			html.Div([html.Button('Back Button', id='btn-1', n_clicks=0)],
				style={'width': '50%', 'float': 'right'}),
			])

@app.callback([Output(component_id='map', component_property='figure')],
			  [Input("btn-1", "n_clicks"),
			   Input("map", "clickData")])

def update_map(btn1, clickData):
	# import ipdb; ipdb.set_trace()
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'btn-1' in changed_id:
		print("Back Button Clicked")
		districts_map.data = [districts_map.data[0]]

	# print(btn1)
	if clickData is not None:
		chosen_district = clickData['points'][0]['location']
		if chosen_district in districts:
			districts_map.data = [districts_map.data[0]]
			updated_data, colors, chosen_villages = crop(json_data_village, chosen_district)
			df_villages = DataFrame({'villages': chosen_villages, 'colors': colors})

			districts_map.add_trace(
				px.choropleth_mapbox(df_villages,
									geojson=updated_data,
									featureidkey='properties.NAME',
									locations=df_villages['villages'],
									color=df_villages['colors'],
									# colorscale=[[0.0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
									opacity=1).data[0]
			)
			districts_map.update_layout(mapbox_style="carto-positron",
					mapbox_zoom=7, mapbox_center = {"lat": 25.856, "lon": 85.786}, 
					height=550, width=1200,
					title='Sanitary Pads Machine App', title_font_size=20,
					margin_l=160, margin_r=0, margin_t=40, margin_b=0)
		elif chosen_district in villages:
			print("Village is clicked: ", chosen_district)
			raise PreventUpdate

	return [districts_map]

app.run_server()