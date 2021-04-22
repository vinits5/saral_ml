import os
import csv
import json
import requests
import pandas as pd
import plotly.express as px
import argparse

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

def options():
	parser = argparse.ArgumentParser(description='Indian Pads App')
	parser.add_argument('--data_location', 
						type=str, 
						default='local', 
						metavar='N',
						choices=['drive', 'local', 'git'],
						help='Specify path to find the data')
	parser.add_argument('--drive_location',
						type=str,
						default='/content/drive/MyDrive/Job/Saral/india_pads_app',
						help='Specify path from the drive if drive location is choosed.')
	args = parser.parse_args()
	return args

class IndiaMap:
	def __init__(self, create_map=True):
		if args.data_location == 'git':
			filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/india/state/india_state.json'
			resp = requests.get(filename)
			self.json_data = json.loads(resp.text)
		elif args.data_location == 'local':
			filename = 'datasets/india/state/india_state.json'
			file = open(filename, 'r')
			self.json_data = json.load(file)
		elif args.data_location == 'drive':
			filename = os.path.join(args.drive_location, f"datasets/india/state/india_state.json")
			file = open(filename, 'r')
			self.json_data = json.load(file)

		self.featureidkey = 'properties.NAME_1'
		self.states = [dd['properties']['NAME_1'] for dd in self.json_data['features']]
		self.locations = [np.mean(dd['geometry']['coordinates'][0], 0) for dd in self.json_data['features']]

		if create_map:
			self.create_layer()

	def create_dataframe(self, states):
		colors = [0 for _ in states]
		self.df = DataFrame({'states': states, 'colors': colors})
		del colors

	def create_layer(self):
		self.create_dataframe(self.states)
		self.map_india_layer = go.Figure(go.Choroplethmapbox(customdata=self.df,
							geojson=self.json_data,
							featureidkey=self.featureidkey,
							locations=self.df['states'],
							z=self.df['colors'],
							marker_opacity=0.5,
							))

		self.map_india_layer.update_layout(mapbox_style="carto-positron",
							mapbox_zoom=3.4, mapbox_center = {"lat": 20.5937, "lon": 78.9629}, 
							height=550, width=1200,
							title='Sanitary Pads Machine App', title_font_size=20,
							margin_l=160, margin_r=0, margin_t=40, margin_b=0)

	def update_map_layout(self, mapbox_zoom=3.4, lat=20.5937, lon=78.9629):
		self.map_india_layer.update_layout(mapbox_style="carto-positron",
							mapbox_zoom=mapbox_zoom, mapbox_center = {"lat": lat, "lon": lon}, 
							height=550, width=1200,
							title='Sanitary Pads Machine App', title_font_size=20,
							margin_l=160, margin_r=0, margin_t=40, margin_b=0)
		
		# Only show the scale of the last layer.
		for i in range(len(self.map_india_layer.data)):
			if i == len(self.map_india_layer.data)-1:
				self.map_india_layer.data[i].update(showscale=True)
			else:
				self.map_india_layer.data[i].update(showscale=False)

	def add_layer(self, layer):
		self.map_india_layer.add_trace(layer.data[0])

	def remove_layer(self, idx=-1):
		layers_data = []
		for i in range(len(self.map_india_layer.data)):
			if i == idx:
				pass
			else:
				layers_data.append(self.map_india_layer.data[i])
		self.map_india_layer.data = layers_data

	def remove_added_layers(self):
		self.map_india_layer.data = [self.map_india_layer.data[0]]

	def find_no_layers(self):
		return len(self.map_india_layer.data)


class StateMap:
	def __init__(self):
		if args.data_location == 'git':
			filename = 'https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/india/district/india_district.json'
			resp = requests.get(filename)
			self.json_data = json.loads(resp.text)
		elif args.data_location == 'local':
			filename = 'datasets/india/district/india_district.json'
			file = open(filename, 'r')
			self.json_data = json.load(file)
		elif args.data_location == 'drive':
			filename = os.path.join(args.drive_location, f"datasets/india/district/india_district.json")
			file = open(filename, 'r')
			self.json_data = json.load(file)

		self.featureidkey = 'properties.NAME_2'
		self.states = [dd['properties']['NAME_1'] for dd in self.json_data['features']]
		self.districts = [dd['properties']['NAME_2'] for dd in self.json_data['features']]
		self.locations = [np.mean(dd['geometry']['coordinates'][0], 0) for dd in self.json_data['features']]

	def create_dataframe(self, state):
		state_idxs = np.where(np.array(self.states) == state)
		colors = [0 for _ in state_idxs[0]]
		districts = np.array(self.districts)[state_idxs[0]]
		self.df = DataFrame({'districts': districts, 'colors': colors})
		del districts, colors, state_idxs

	def create_state_json_file(self, state):
		state_idxs = np.where(np.array(self.states) == state)
		print("State Map -> Districts", set(np.array(self.districts)[state_idxs[0]]))
		self.state_json_data = {}
		self.state_json_data['type'] = 'FeatureCollection'
		self.state_json_data['features'] = []
		for ii in state_idxs[0]:
			self.state_json_data['features'].append(self.json_data['features'][ii])
		del state_idxs

	def create_center(self, state):
		import ipdb; ipdb.set_trace()
		state_idxs = np.where(np.array(self.states) == state)
		locations = np.array(self.locations)[state_idxs[0]]
		print(locations)
		self.center = np.mean(locations, 0)
		print(self.center)
		del state_idxs

	def create_layer(self, state):
		self.create_dataframe(state)
		self.create_state_json_file(state)
		# self.create_center(state)
		return go.Figure(go.Choroplethmapbox(customdata=self.df,
							geojson=self.state_json_data,
							featureidkey=self.featureidkey,
							locations=self.df['districts'],
							z=self.df['colors'],
							marker_opacity=0.5,
							))


class Selections:
	def __init__(self):
		self.past_selected_state = None
		self.selected_state = None
		self.selected_district = None
		self.selected_village = None
		self.state_changed = False
		self.district_map = None

	def which_state(self):
		return self.selected_state

	def which_district(self):
		return self.selected_district

	def which_village(self):
		return self.selected_village

	def set_state(self, state):
		self.selected_state = state

	def set_district(self, district):
		self.selected_district = district

	def set_village(self, village):
		self.selected_village = village

	def find_district_map(self):
		remove_state_map = False
		if self.which_state() != self.past_selected_state:
		# If the new state is different than the old selected state.
			remove_state_map = True
			self.district_map = DistrictMap(self.which_state())			# Create new district map.
			self.past_selected_state = self.selected_state 				# Update the past seletected state.
		return self.district_map, remove_state_map

	def find_pads_map(self):
		return PadsMap(self.district_map)


class DistrictMap:
	def __init__(self, state):
		self.state = state
		if args.data_location == 'git':
			filename = f"https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/india/villages/{state}.json"
			resp = requests.get(filename)
			self.json_data = json.loads(resp.text)
		elif args.data_location == 'local':
			filename = f"datasets/india/villages/{state}.json"
			file = open(filename, 'r')
			self.json_data = json.load(file)
		elif args.data_location == 'drive':
			filename = os.path.join(args.drive_location, f"datasets/india/villages/{state}.json")
			file = open(filename, 'r')
			self.json_data = json.load(file)

		self.featureidkey = 'properties.NAME'
		self.states = [dd['properties']['STATE'] for dd in self.json_data['features']]
		self.districts = [dd['properties']['DISTRICT'] for dd in self.json_data['features']]
		print("District Map -> Districts", set(self.districts))
		# self.districts = [dd['properties']['SUB_DIST'] for dd in self.json_data['features']]
		# import pdb; pdb.set_trace()
		self.villages = [dd['properties']['NAME'] for dd in self.json_data['features']]
		self.locations = np.array([np.mean(dd['geometry']['coordinates'][0], 0) for dd in self.json_data['features']])

	def create_dataframe(self, district):
		if district == 'Nashik': district = 'Nasik'
		if district == 'Bid': district = 'Beed'
		district_idxs = np.where(np.array(self.districts) == district)
		colors = [0 for _ in district_idxs[0]]
		villages = np.array(self.villages)[district_idxs[0]]
		locations = self.locations[district_idxs[0]]
		self.df = DataFrame({'villages': villages, 
							 'colors': colors,
							 'lat': locations[:,1],
							 'lon': locations[:,0]})
		del villages, colors, district_idxs, locations

	def create_district_json_file(self, district):
		if district == 'Nashik': district = 'Nasik'
		if district == 'Bid': district = 'Beed'
		district_idxs = np.where(np.array(self.districts) == district)
		self.district_json_data = {}
		self.district_json_data['type'] = 'FeatureCollection'
		self.district_json_data['features'] = []
		for ii in district_idxs[0]:
			self.district_json_data['features'].append(self.json_data['features'][ii])
		del district_idxs

	def create_center(self, district):
		if district == 'Nashik': district = 'Nasik'
		if district == 'Bid': district = 'Beed'
		state_idxs = np.where(np.array(self.districts) == district)
		locations = np.array(self.locations)[state_idxs[0]]
		self.center = np.mean(locations, 0)
		del state_idxs

	def create_layer(self, district):
		self.create_dataframe(district)
		self.create_district_json_file(district)
		self.create_center(district)

		return go.Figure(go.Choroplethmapbox(customdata=self.df,
							geojson=self.district_json_data,
							featureidkey=self.featureidkey,
							locations=self.df['villages'],
							z=self.df['colors'],
							marker_opacity=0.5,
							))


class PadsMap:
	def __init__(self, district_map):
		self.district_map = district_map
		self.featureidkey = 'properties.NAME'
		self.data = data_file.loc[data_file['State Name'] == self.district_map.state] 

	def find_village_population(self, state, district, village):
		data = self.data.loc[self.data['Village Name'] == village]
		data = data.loc[data['District Name'] == district]
		if len(data) == 1: return int(data['Total Population of Village'].to_numpy())
		else:
			try:
				return int(data.iloc[0]['Total Population of Village'])
			except:
				return 0

	def create_dataframe(self, location, target_population, machine_choice):
		# if district == 'Nashik': district = 'Nasik'
		# dist = self.locs_updated - np.array([[lat, lon]])
		dist = self.district_map.locations - location
		dist = dist**2
		dist = np.sum(dist, -1)
		closest = np.argsort(dist)
		total_population = 0
		self.highlighted_index, populations = [], []

		for ii in closest:
			population = self.find_village_population(self.district_map.states[ii], self.district_map.districts[ii], self.district_map.villages[ii])
			self.highlighted_index.append(ii)
			populations.append(population)
			total_population += population*target_population

			if total_population > machine_choice_dict[machine_choice]:
				break

		villages = np.array(self.district_map.villages)[self.highlighted_index]
		colors = [0.5 for _ in villages]
		colors[0] = 1

		self.df = DataFrame({'villages': villages,
							 'colors': colors,
							 'populations': populations,
							 'distance': dist[self.highlighted_index]})
		del villages, colors

	def create_json_file(self):
		self.highlighted_json_data = {}
		self.highlighted_json_data['type'] = 'FeatureCollection'
		self.highlighted_json_data['features'] = []
		for ii in self.highlighted_index:
			self.highlighted_json_data['features'].append(self.district_map.json_data['features'][ii])

	def create_layer(self, lat, lon, target_population, machine_choice):
		location = np.array([[lon, lat]])
		self.create_dataframe(location, target_population, machine_choice)
		self.create_json_file()
		return go.Figure(go.Choroplethmapbox(customdata=self.df,
							geojson=self.highlighted_json_data,
							featureidkey=self.featureidkey,
							locations=self.df['villages'],
							z=self.df['colors'],
							hovertemplate = '<b>Village</b>: <b>%{customdata[0]}</b>'+
											'<br><b>Population</b>: <b>%{customdata[2]}</b><br>'+
											'<br><b>Distance</b>: <b>%{customdata[3]}</b><br>',
							marker_opacity=1,
							))


online_data = False

args = options()

if args.data_location == 'git':
	data_file = pd.read_csv('https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/village_compiled_dataset.csv')
elif args.data_location == 'drive':
	data_file = pd.read_csv(os.path.join(args.drive_location, 'datasets/village_compiled_dataset.csv'))
elif args.data_location == 'local':
	data_file = pd.read_csv('datasets/village_compiled_dataset.csv')

machine_choice_dict = {
	'swach-micro': 5700,
	'swach-1.3': 17000,
	'swach-3.2': 30000,
	'swach-4.2': 132000,
	'swach-5.0': 285000,
}

target_population = [x/10 for x in range(1, 11)]

india_map = IndiaMap(create_map=True)
state_map = StateMap()
selection = Selections()



app = Dash(__name__)

app.layout = html.Div([
			html.Div([
				dcc.Graph(id='map', style={'width': '120'}, figure={}),
			], style={'width': '100%', 'float': 'center', 'display': 'inline-block'}),
			html.Div([html.P(""), 
					 ], style={'width': '5%', 'display': 'inline-block'}),
			html.Div([html.Button('Go to State Level', id='btn-1', n_clicks=0)],
				style={'width': '20%', 'display': 'inline-block'}),
			html.Div([html.Button('Go to India Level', id='btn-2', n_clicks=0)],
				style={'width': '25%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='machine_choice',
					options = [{"value": x, "label": x} for x in list(machine_choice_dict.keys())],
					value = list(machine_choice_dict.keys())[0],
					style={'width': "90%"}),], 
					style={'width': '25%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='target_population',
					options = [{"value": x, "label": x} for x in target_population],
					value = target_population[2],
					style={'width': "90%"}),], 
					style={'width': '25%', 'display': 'inline-block', 'float': 'right'}),
			])

@app.callback([Output(component_id='map', component_property='figure')],
			  [Input("btn-1", "n_clicks"),
			   Input("btn-2", "n_clicks"),
			   Input("map", "clickData"),
			   Input("machine_choice", "value"),
			   Input("target_population", "value"),])


def update_map(btn1, btn2, clickData, machine_choice, target_population):
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	print(clickData)

	if 'btn-1' in changed_id:
		if india_map.find_no_layers() == 1 or india_map.find_no_layers() == 2:
			raise PreventUpdate
		else:
			india_map.remove_layer(2)		# Remove old district map
			india_map.update_map_layout()

	elif 'btn-2' in changed_id:
		if india_map.find_no_layers() == 1:
			raise PreventUpdate
		else:
			india_map.remove_added_layers()
			india_map.update_map_layout()

	elif clickData is not None:
		if clickData['points'][0]['curveNumber'] == 1:
			selected_district = clickData['points'][0]['location']
			selection.set_district(selected_district)
			if selection.which_state() in ['Maharashtra', 'Bihar']:
				district_map, remove_state_map = selection.find_district_map()
				# if remove_state_map: india_map.remove_layer(1)	# remove state map.
				print("Selected District", selected_district)

				india_map.remove_layer(3)
				india_map.remove_layer(2)		# Remove old district map
				india_map.add_layer(district_map.create_layer(selected_district))
				center = district_map.center
				india_map.update_map_layout(mapbox_zoom=7, lat=center[1], lon=center[0])
				del district_map
			else:
				raise PreventUpdate
		elif clickData['points'][0]['curveNumber'] == 0:
			selected_state = clickData['points'][0]['location']
			selection.set_state(selected_state)
			print("State clicked: ", selected_state)
			
			india_map.remove_layer(1)						# In order to remove the existing state.
			india_map.add_layer(state_map.create_layer(selected_state))

			india_map.update_map_layout()
		elif clickData['points'][0]['curveNumber'] == 2:
			print(selection.which_state(), selection.which_district())
			lat = clickData['points'][0]['customdata'][2]
			lon = clickData['points'][0]['customdata'][3]
			selected_village = clickData['points'][0]['location']
			print('Village clicked: ', selected_village)
			selection.set_village(selected_village)

			pads_map = selection.find_pads_map()
			india_map.remove_layer(3)
			layer = pads_map.create_layer(lat, lon, target_population, machine_choice)
			india_map.add_layer(layer)
			center = pads_map.district_map.center
			india_map.update_map_layout(mapbox_zoom=7, lat=center[1], lon=center[0])
		else:
			raise PreventUpdate

	return [india_map.map_india_layer]

app.run_server()