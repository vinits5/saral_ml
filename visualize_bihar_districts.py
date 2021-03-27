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

import time
start = time.time()
# file = open('datasets/br.json', 'r')
# json_data = json.load(file)

json_filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/br.json'
resp = requests.get(json_filename)
json_data = json.loads(resp.text)

end = time.time()
print("Time to load bihar json: ", end-start)

districts = []

for dd in json_data['features']:
	districts.append(dd['properties']['DISTRICT'])

district_names = list(set(districts))


def crop(json_data, select_districts):
	updated_data = {}

	updated_data['type'] = 'FeatureCollection'
	updated_data['features'] = []

	districts_temp = np.array(districts)
	idxs = []
	for select_district in select_districts:
		idxs_temp = np.where(districts_temp == select_district)
		idxs.append(idxs_temp[0])
	
	idxs = np.concatenate(idxs)
	district_villages = []
	locs = []

	for idx in idxs:
		updated_data['features'].append(json_data['features'][idx])
		district_villages.append(json_data['features'][idx]['properties']['NAME'])
		locs.append(np.mean(json_data['features'][idx]['geometry']['coordinates'][0], 0))

	return updated_data, district_villages, locs

app = Dash(__name__)

app.layout = html.Div([
			# html.H1("Choropleth Maps Sanitary Pads", style={'text-align': 'center'}),
			html.Div([html.P("Select District")], 
					 style={'width': '50%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='select_district',
					options = [{"value": x, "label": x} for x in district_names],
					value = [district_names[0]],
					multi=True,
					style={'width': "90%"}),], 
					style={'width': '50%', 'display': 'inline-block', 'float': 'right'}),
			html.Div([
				dcc.Graph(id='map', style={'width': '120'}, figure={}),
			], style={'width': '100%', 'float': 'center', 'display': 'inline-block'}),
			])

@app.callback([Output(component_id='map', component_property='figure')],
			  [Input("select_district", "value")])

def update_map(select_districts):
	import time
	start = time.time()
	updated_json_data, district_villages, locs = crop(json_data, select_districts)
	end = time.time()
	print("Time for crop: ", end-start)
	locs = np.array(locs)
	lon, lat = np.mean(locs, 0)
	print(lon, lat)
	# import ipdb; ipdb.set_trace()
	
	random = []
	random = [np.random.random() for _ in district_villages]
	
	data = {'villages': district_villages, 'random': random, 'lat': locs[:,1], 'lon': locs[:, 0]}
	df = DataFrame(data)
	
	districts_map = go.Figure(go.Choroplethmapbox(customdata=df,
							geojson=updated_json_data,
							featureidkey='properties.NAME',
							locations=df['villages'],
							z=df['random'],
							hovertemplate = '<b>State</b>: <b>%{customdata[0]}</b>'+
											'<br><b>lat</b>: <b>%{customdata[2]}</b><br>'+
											'<br><b>lon</b>: <b>%{customdata[3]}</b><br>',
							marker_opacity=0.5,
							# color=self.df_india.variable,
							# colorscale=[[0.0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# hoverinfo=hover_data,
							))
	districts_map.update_layout(mapbox_style="carto-positron",
			mapbox_zoom=9, mapbox_center = {"lat": lat, "lon": lon}, 
			height=550, width=1200,
			title='Sanitary Pads Machine App', title_font_size=20,
			margin_l=160, margin_r=0, margin_t=40, margin_b=0)
	return [districts_map]

app.run_server()