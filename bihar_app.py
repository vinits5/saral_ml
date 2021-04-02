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

online_data = True

if online_data:
	data = pd.read_csv('https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/bihar.csv')
	json_filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/br.json'
	resp = requests.get(json_filename)
	json_data = json.loads(resp.text)
else:
	data = pd.read_csv('datasets/bihar.csv')
	file = open('datasets/br.json', 'r')
	json_data = json.load(file)
# district_json_data = json.load(open('datasets/india_district.json', 'r'))
district_json_data = None

districts = []
for dd in json_data['features']:
	districts.append(dd['properties']['DISTRICT'])

district_names = set(data['District Name'].to_numpy())
lats = data['lat'].to_numpy()
lons = data['lon'].to_numpy()
locations = np.concatenate([lats.reshape(-1,1), lons.reshape(-1,1)], axis=-1)
villages = data['Village Name'].to_numpy().reshape(-1)
populations = data['Total Population of Village'].to_numpy().reshape(-1)

machine_choice_dict = {
	'swach-micro': 5700,
	'swach-1.3': 17000,
	'swach-3.2': 30000,
	'swach-4.2': 132000,
	'swach-5.0': 285000,
}

target_population = [x/10 for x in range(1, 11)]

class DataHandler:
	def __init__(self):
		self.select_district = list(district_names)[0]
		self.set_district(self.select_district)
		self.select_village = self.village_names[0]

	def set_district(self, select_district):
		self.data_district = data.loc[data['District Name'] == select_district]
		self.village_names = self.data_district['Village Name'].to_numpy()
		self.select_district = select_district

	@staticmethod
	def crop(json_data, district_json_data, select_districts):
		updated_data = {}

		updated_data['type'] = 'FeatureCollection'
		updated_data['features'] = []

		idxs = []
		for select_district in select_districts:
			idxs_temp = np.where(np.array(districts) == select_district)
			idxs.append(idxs_temp[0])
		
		idxs = np.concatenate(idxs)
		district_villages = []
		locs = []
		population = []

		for idx in idxs:
			updated_data['features'].append(json_data['features'][idx])

		# for select_district in select_districts:
		# 	for dd in district_json_data['features']:
		# 		if dd['properties']['NAME_2'] == select_district:
		# 			updated_data.append(dd)

		return updated_data

	@staticmethod
	def search_custom_data(villages_targeted, districts_targeted, select_village):
		df1 = []
		for district in districts_targeted:
			df1.append(data.loc[data['District Name'] == district])
		df = pd.concat(df1)
		df['color'] = 0.0	

		for village in villages_targeted:
			temp_indices = np.array(df.loc[df['Village Name'] == village].index)
			# print(village, temp_indices)			
			if village == select_village:
				for temp_idx in temp_indices:
					df.at[temp_idx, 'color'] = 0.5
			else:
				for temp_idx in temp_indices:
					df.at[temp_idx, 'color'] = 1.0

		df.rename(columns={'Village Name': 'villages'}, inplace=True)
		return df

	# max: 80.88, 21.75
	# min: 75.38, 17.25
	def find_closest_villages(self, select_village, machine_choice=50000, target_population=0.2):
		data_vil = self.data_district.loc[self.data_district['Village Name'] == select_village]
		lat = float(data_vil.iloc[0]['lat'])
		lon = float(data_vil.iloc[0]['lon'])

		dist = locations - np.array([[lat, lon]])
		dist = dist**2
		dist = np.sqrt(np.sum(dist, -1))
		closest = np.argsort(dist)

		villages_targeted = []
		districts_targeted = []
		highlight = []
		populations_targeted = []
		locations_targeted = []

		population = 0.0
		# import ipdb; ipdb.set_trace()
		for ii in closest:
			if population<machine_choice:
				population += int(data.iloc[ii]['Total Population of Village'])*target_population

				village_temp = str(data.iloc[ii]['Village Name'])
				villages_targeted.append(village_temp)
				districts_targeted.append(data.iloc[ii]['District Name'])
			else:
				break

		# print([data.iloc[closest[x]]['Village Name'] for x in range(30)])
		# print([dist[closest[x]] for x in range(30)])

		districts_targeted = list(set(districts_targeted))
		updated_data = self.crop(json_data, district_json_data, districts_targeted)
		# import ipdb; ipdb.set_trace()
		df = self.search_custom_data(villages_targeted, districts_targeted, select_village)
		return df, updated_data, lat, lon


class Update:
	def __init__(self):
		self.india_corr = "Villages : []"
		self.districts_map = None
		self.dh = DataHandler()

	def __call__(self, select_district, machine_choice, select_village, target_population):
		if self.dh.select_district != select_district:
			self.dh.set_district(select_district)
			return self.update_map(machine_choice_dict[machine_choice], self.dh.village_names[0], target_population)
		try:
			return self.update_map(machine_choice_dict[machine_choice], select_village, target_population)
		except:
			return self.districts_map, [{"value": x, "label": x} for x in self.dh.village_names]

	def update_map(self, machine_choice, select_village, target_population):
		df_state, updated_data, lat, lon = self.dh.find_closest_villages(select_village, machine_choice, target_population)
		
		import time
		start = time.time()
		self.districts_map = go.Figure(go.Choroplethmapbox(customdata=df_state,
							geojson=updated_data,
							featureidkey='properties.NAME',
							locations=df_state.villages,
							text = df_state['color'],
							z=df_state['color'],
							hovertemplate = '<b>Village</b>: <b>%{customdata[5]}</b>'+
											'<br><b>Population</b>: <b>%{customdata[9]}</b><br>'+
											'<b>District</b>: <b>%{customdata[3]}</b>',
							marker_opacity=0.5,
							# color=self.df_india.variable,
							colorscale=[[0.0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# colorscale=[[0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# hoverinfo=hover_data,
							))
		end = time.time()
		print("\nTime taken for map: ", end-start)
		# import ipdb; ipdb.set_trace()

		self.districts_map.update_layout(mapbox_style="carto-positron",
				mapbox_zoom=10, mapbox_center = {"lat": lat, "lon": lon}, 
				height=550, width=1200,
				title='Sanitary Pads Machine App', title_font_size=20,
				margin_l=160, margin_r=0, margin_t=40, margin_b=0)

		return self.districts_map, [{"value": x, "label": x} for x in self.dh.village_names]

uc = Update()
app = Dash(__name__)

app.layout = html.Div([
			# html.H1("Choropleth Maps Sanitary Pads", style={'text-align': 'center'}),
			html.Div([
				dcc.Graph(id='map', style={'width': '120'}, figure={}),
			], style={'width': '100%', 'float': 'center', 'display': 'inline-block'}),

			html.Div([html.P(""), 
					 ], style={'width': '5%', 'display': 'inline-block'}),
			html.Div([html.P("District:"), 
					 ], style={'width': '20%', 'display': 'inline-block'}),
			html.Div([html.P("Machine Choice:"), 
					 ], style={'width': '25%', 'display': 'inline-block'}),
			html.Div([html.P("Percentage Population Penetrated:"), 
					 ], style={'width': '25%', 'display': 'inline-block'}),
			html.Div([html.P("Select Village:"), 
					 ], style={'width': '25%', 'display': 'inline-block', 'float': 'right'}),

			html.Div([html.P(""), 
					 ], style={'width': '5%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='select_district',
					options = [{"value": x, "label": x} for x in district_names],
					value = list(district_names)[0],
					style={'width': "90%"}),], 
					style={'width': '20%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='machine_choice',
					options = [{"value": x, "label": x} for x in list(machine_choice_dict.keys())],
					value = list(machine_choice_dict.keys())[0],
					style={'width': "90%"}),], 
					style={'width': '25%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='target_population',
					options = [{"value": x, "label": x} for x in target_population],
					value = target_population[2],
					style={'width': "90%"}),], 
					style={'width': '25%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='select_village',
					options = [{"value": x, "label": x} for x in uc.dh.village_names],
					value = uc.dh.village_names[0],
					style={'width': "90%"}),], 
					style={'width': '25%', 'display': 'inline-block', 'float': 'right'}),			
			])

@app.callback([Output(component_id='map', component_property='figure'),
			   Output(component_id='select_village', component_property='options'),],
			  [Input("select_district", "value"),
			   Input("machine_choice", "value"),
			   Input(component_id='select_village', component_property='value'),
			   Input("target_population", "value"),
			   Input('map', 'clickData'),])
			   # Input('submit-val', 'n_clicks')])


def update_map(select_district, machine_choice, select_village, target_population, clickData):
	if clickData is not None:
		select_village = clickData['points'][0]['location']
		print(select_village)
	return uc(select_district, machine_choice, select_village, target_population)

app.run_server()
