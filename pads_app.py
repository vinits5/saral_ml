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

# filename = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/mh_villages/Dhule.json"
# file = open(filename)
# resp = requests.get(filename)
# json_data = json.loads(resp.text)
mh_csv = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/MH_District_Data.csv"
csvfile = pd.read_csv(mh_csv)

district_json_files_df = pd.read_csv('https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/village_json_files.csv')
district_names = district_json_files_df['districts'].to_numpy()
district_json_files = district_json_files_df['raw_github_locs'].to_numpy()


machine_choice_dict = {
	'machine1': 10000,
	'machine2': 20000,
	'machine3': 30000,
	'machine4': 40000,
	'machine5': 50000,
}


class Village:
	def __init__(self):
		self.select_district = "Dhule"
		self.set_village(self.select_district)

	def set_village(self, select_district):
		district_idx = np.where(select_district == district_names)[0][0]
		district_json_file = district_json_files[district_idx]

		self.select_district = select_district
		resp = requests.get(district_json_file)
		self.json_data = json.loads(resp.text)
		villages = []
		districts = []
		locs = []

		for data in self.json_data['features']:
			villages.append(data['properties']['VILLNAME'])
			districts.append(data['properties']['DTNAME'])
			loc = np.mean(data['geometry']['coordinates'][0], 0)
			locs.append(loc)

		villages = np.array(villages)
		districts = np.array(districts)
		locs = np.array(locs)

		data_districts = np.array(csvfile['1'])[1:]
		idxs = np.where(data_districts == 'Dhule')
		data_districts = data_districts[idxs[0]]

		data_villages = np.array(csvfile['3'])[idxs[0]]
		total_population = np.array(csvfile['5'][idxs[0]])
		total_population = total_population.astype(np.float)


		village_updated = []
		total_population_updated = []
		locs_updated = []
		for idx, vil in enumerate(villages):
			v_id = np.where(vil==data_villages)
			village_updated.append(vil)
			locs_updated.append(locs[idx])
			if v_id[0].size > 0:
				total_population_updated.append(total_population[v_id[0][0]])
			else:
				total_population_updated.append(0.0)

		self.village_updated = np.array(village_updated)
		self.total_population_updated = np.array(total_population_updated)
		self.locs_updated = np.array(locs_updated)

	# max: 80.88, 21.75
	# min: 75.38, 17.25
	def find_closest_villages(self, lat, lon, machine_choice=50000):
		dist = self.locs_updated - np.array([[lat, lon]])
		dist = dist**2
		dist = np.sum(dist, -1)
		closest = np.argsort(dist)

		villages_covered = []
		# highlight = np.zeros((locs_updated.shape[0]), dtype=np.int64)
		highlight = np.array(['not useful' for _ in range(self.locs_updated.shape[0])])
		population = 0.0
		# import ipdb; ipdb.set_trace()
		for ii in closest:
			if population<machine_choice:
				try:
					population += int(self.total_population_updated[ii])
				except:
					import ipdb; ipdb.set_trace()
				highlight[ii] = 'useful'
				villages_covered.append(self.village_updated[ii])
			else:
				break
		return highlight, villages_covered


class Update:
	def __init__(self):
		self.india_corr = "Villages : []"
		self.districts_map = None
		self.village = Village()

	def __call__(self, select_district, input_lat, input_lon, machine_choice):
		if self.village.select_district != select_district:
			self.village.set_village(select_district)
		
		self.lat_range = "Range of Latitude: {} to {}".format(round(np.min(self.village.locs_updated,0)[0], 3), round(np.max(self.village.locs_updated, 0)[0], 3))
		self.lon_range = "Range of Longitude: {} to {}".format(round(np.min(self.village.locs_updated,0)[1], 3), round(np.max(self.village.locs_updated, 0)[1], 3))
		try:
			lat = float(input_lat)
			lon = float(input_lon)
			return self.update_map(lat, lon, machine_choice_dict[machine_choice])
		except:
			return self.india_corr, self.districts_map, self.lat_range, self.lon_range

	def update_map(self, lat, lon, machine_choice):
		highlight, villages_covered = self.village.find_closest_villages(lat, lon, machine_choice)
		# print(highlight)
		result = ""
		for x in villages_covered:
			result += x +", "
		self.india_corr = "Villages: {}".format(result)

		data = {'villages': self.village.village_updated, 'random': highlight, 'lat': self.village.locs_updated[:,0], 'lon': self.village.locs_updated[:,1]}
		df_state = DataFrame(data)

		self.districts_map = px.choropleth(df_state,
								geojson=self.village.json_data,
								featureidkey='properties.VILLNAME',
								locations='villages',
								color='random',
								# color_continuous_scale='gray',
								hover_data=['lat', 'lon']
								)
		self.districts_map.update_geos(fitbounds='locations',visible=False)
		return self.india_corr, self.districts_map, self.lat_range, self.lon_range

uc = Update()
app = Dash(__name__)

app.layout = html.Div([
			html.H1("Choropleth Maps Sanitary Pads", style={'text-align': 'center'}),
			html.P("Range of Latitude: [] to []", id='lat_range'),
			html.P("Range of Longitude: [] to []", id='lon_range'),
			html.Div([
				html.P("Distirct:"),
				html.P("Latitude:"),
				html.P("Longitude:"),
				html.P("Machine Choice:")
			], style={'width': '48%', 'float': 'left', 'display': 'inline-block'}),
			html.Div([
				dcc.Dropdown(id='select_district',
					options = [{"value": x, "label": x} for x in district_names],
					value = 'Dhule',
					style={'width': "90%"}),
				html.Br(),
				dcc.Input(id="input_lat", type="text", placeholder="latitude", value='76.0'),
				html.Br(),
				dcc.Input(id="input_lon", type="text", placeholder="longitude", value='18.0'),
				html.Br(),
				dcc.Dropdown(id='machine_choice',
					options = [{"value": x, "label": x} for x in list(machine_choice_dict.keys())],
					value = 'machine1',
					style={'width': "90%"}),
			], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
			html.Br(),
			# dbc.Button('Submit', id='submit-val', n_clicks=0),
			html.Div([
				html.P("Villages: []", id='villages'),
				dcc.Graph(id='map', style={'width': '120'}, figure={}),
			], style={'width': '100%', 'float': 'center', 'display': 'inline-block'})
			])

@app.callback([Output(component_id='villages', component_property='children'), 
			   Output(component_id='map', component_property='figure'),
			   Output(component_id='lat_range', component_property='children'),
			   Output(component_id='lon_range', component_property='children'),],
			  [Input("select_district", "value"),
			   Input("input_lat", "value"),
			   Input("input_lon", "value"),
			   Input("machine_choice", "value"),])
			   # Input('submit-val', 'n_clicks')])


def update_map(select_district, input_lat, input_lon, machine_choice):
	# print(select_district)
	return uc(select_district, input_lat, input_lon, machine_choice)

app.run_server()