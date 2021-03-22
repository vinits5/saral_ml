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
		self.select_village = self.village_updated[0]

	def set_village(self, select_district):
		district_idx = np.where(select_district == district_names)[0][0]
		district_json_file = district_json_files[district_idx]

		self.select_district = select_district
		if select_district == 'Nasik': select_district = "Nashik"
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
		idxs = np.where(data_districts == select_district)
		data_districts = data_districts[idxs[0]]

		data_villages = np.array(csvfile['3'])[idxs[0]]
		total_population = np.array(csvfile['5'][idxs[0]])
		total_population = total_population.astype(np.float)


		village_updated = []
		total_population_updated = []
		locs_updated = []
		# import ipdb; ipdb.set_trace()
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
	def find_closest_villages(self, select_village, machine_choice=50000):
		idx = np.where(self.village_updated==select_village)
		if idx[0].size > 0:
			lat = self.locs_updated[idx[0]][0][0]
			lon = self.locs_updated[idx[0]][0][1]

		dist = self.locs_updated - np.array([[lat, lon]])
		dist = dist**2
		dist = np.sum(dist, -1)
		closest = np.argsort(dist)

		villages_covered = []
		# highlight = np.zeros((locs_updated.shape[0]), dtype=np.int64)
		# highlight = np.array(['not useful' for _ in range(self.locs_updated.shape[0])])
		highlight = np.array([0.0 for _ in range(self.locs_updated.shape[0])])
		population = 0.0
		# import ipdb; ipdb.set_trace()
		for ii in closest:
			if population<machine_choice:
				try:
					population += int(self.total_population_updated[ii])
				except:
					import ipdb; ipdb.set_trace()
				# highlight[ii] = 'useful'
				highlight[ii] = 1.0
				villages_covered.append(self.village_updated[ii])
			else:
				break
		highlight[idx[0][0]] = 0.5
		return highlight, villages_covered, lat, lon


class Update:
	def __init__(self):
		self.india_corr = "Villages : []"
		self.districts_map = None
		self.village = Village()

	def __call__(self, select_district, machine_choice, select_village):
		if self.village.select_district != select_district:
			self.village.set_village(select_district)
			return self.update_map(machine_choice_dict[machine_choice], self.village.village_updated[0])

		try:
			return self.update_map(machine_choice_dict[machine_choice], select_village)
		except:
			return self.districts_map, [{"value": x, "label": x} for x in self.village.village_updated]

	def update_map(self, machine_choice, select_village):
		highlight, villages_covered, lat, lon = self.village.find_closest_villages(select_village, machine_choice)
		# print(highlight)

		data = {'villages': self.village.village_updated, 'random': highlight, 'lat': self.village.locs_updated[:,0], 'lon': self.village.locs_updated[:,1], 'population': self.village.total_population_updated}
		df_state = DataFrame(data)

		# self.districts_map = px.choropleth(df_state,
		# 						geojson=self.village.json_data,
		# 						featureidkey='properties.VILLNAME',
		# 						locations='villages',
		# 						color='random',
		# 						# color_continuous_scale='gray',
		# 						hover_data=['population']
		# 						)

		self.districts_map = go.Figure(go.Choroplethmapbox(customdata=df_state,
							geojson=self.village.json_data,
							featureidkey='properties.VILLNAME',
							locations=df_state.villages,
							text = df_state['population'],
							z=df_state['random'],
							hovertemplate = '<b>State</b>: <b>%{customdata[0]}</b>'+
											'<br><b>Population</b>: <b>%{customdata[4]}</b><br>',
							marker_opacity=0.5,
							# color=self.df_india.variable,
							colorscale=[[0.0, 'rgb(255,255,255)'], [0.5, 'rgb(255,0,0)'], [1.0, 'rgb(0,0,0)']],
							# hoverinfo=hover_data,
							))
		self.districts_map.update_layout(mapbox_style="carto-positron",
				mapbox_zoom=9, mapbox_center = {"lat": lon, "lon": lat}, 
				height=550, width=1200,
				title='Sanitary Pads Machine App', title_font_size=20,
				margin_l=160, margin_r=0, margin_t=40, margin_b=0)
		# self.districts_map.update_geos(fitbounds='locations',visible=False)
		# self.districts_map.update_layout(autosize=True)
		return self.districts_map, [{"value": x, "label": x} for x in self.village.village_updated]

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
					 ], style={'width': '28%', 'display': 'inline-block'}),
			html.Div([html.P("Machine Choice:"), 
					 ], style={'width': '33%', 'display': 'inline-block'}),
			html.Div([html.P("Select Village:"), 
					 ], style={'width': '33%', 'display': 'inline-block', 'float': 'right'}),

			html.Div([html.P(""), 
					 ], style={'width': '5%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='select_district',
					options = [{"value": x, "label": x} for x in district_names],
					value = 'Dhule',
					style={'width': "90%"}),], 
					style={'width': '28%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='machine_choice',
					options = [{"value": x, "label": x} for x in list(machine_choice_dict.keys())],
					value = 'machine1',
					style={'width': "90%"}),], 
					style={'width': '33%', 'display': 'inline-block'}),
			html.Div([dcc.Dropdown(id='select_village',
					options = [{"value": x, "label": x} for x in uc.village.village_updated],
					value = uc.village.village_updated[0],
					style={'width': "90%"}),], 
					style={'width': '33%', 'display': 'inline-block', 'float': 'right'}),			
			])

@app.callback([Output(component_id='map', component_property='figure'),
			   Output(component_id='select_village', component_property='options'),],
			  [Input("select_district", "value"),
			   Input("machine_choice", "value"),
			   Input(component_id='select_village', component_property='value'),])
			   # Input('submit-val', 'n_clicks')])


def update_map(select_district, machine_choice, select_village):
	# print(select_village)
	return uc(select_district, machine_choice, select_village)

app.run_server()