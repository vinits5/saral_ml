import os
import csv
import json
import requests
import pandas as pd
import plotly.express as px
from urllib.request import urlopen

import numpy as np
from pandas import DataFrame
import plotly.io as pio
import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from scipy.stats import pearsonr, linregress

# TODOs:
# 1. Add option for multi-variate regression + non-linear model.


state_vars = {
	'Toilet_Facility': 24,
	'Cooking_Fuel': 25,
	'Electricity': 22,
	'Drinking_Water': 23,
	'Women_Literacy': 8,
	'Women_Illiteracy': 7,
	'Women_Married_Below18': 11,
	'Men_Married_Below21': 12,
	'Spontaneous_Abortion': 21,
	# Output Vars.
	'Women_Heard_RTI/STI': 2,
	'Women_Heard_HIV/AIDS': 3,
	'Women_Symptoms_RTI/STI': 4,
	'Women_Symptoms_HIV/AIDS': 5
}
state_vars_op = 9

india_csv = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/Indian_Data.csv"

district_vars = {
	'Geographical_Area': 4,
	'Population': 5,
	'Govt_Primary_Schools': 6,
	'Treated Tap Water': 61,
	'Untreated Tap Water': 62,
	'Mobile Phone Coverage': 74,
	'Public Call': 73,
	'Internet Cafe': 75,
	'Public Buses': 76,
	'Railway Station': 78,
	'Non-agri Area': 92,
	# Output Vars.
	'Primary_Health_Centers': 31,
	'Toilet with Bath': 70,
	'Toilet w/o Bath': 71,
	'Asha Workers': 88
}
district_vars_op = 11

mh_csv = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/MH_District_Data.csv"

district_json_files_df = pd.read_csv('https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/village_json_files.csv')
district_names = district_json_files_df['districts'].to_numpy()
district_json_files = district_json_files_df['raw_github_locs'].to_numpy()

class Dataset:
	def __init__(self, file_path):
		self.file_path = file_path
		self.df = pd.read_csv(self.file_path)
		self.fields = self.find_fields()
		self.district_names_id = 1
		self.state_names_id = 0
		self.village_names_id = 3

	def read_data(self, required_ids):
		# Arguments:
			# file_path:        Location of csv dataset. (String)
			# required_ids:     Integer or List of column numbers to find data. (Can be found in the first row of the dataset.)
		# Output:
			# data:             Array of data for each given column id. (size of required_ids x number of data points)

		try:
			required_ids = [str(x) for x in required_ids]
		except:
			required_ids = str(required_ids)
		data = self.df[required_ids].to_numpy()[1:]
		return data.T

	def find_fields(self):
		return self.df.iloc[0].to_numpy()

	def find_ids(self, field):
		ids = int(np.where(field == self.fields)[0])
		return ids

	@staticmethod
	def replaceNA(data, replace_value):
		idx = np.where(np.isnan(data))
		data[idx[0]] = replace_value
		return data

	def avg_up_state_variables(self, states, variables):
		# states:		List of states available in plotly map.
		# variables:	List of 1D datasets. (ex. [Toilet_Facility, Water_Resource, Literacy])
		data_states = self.read_data(self.state_names_id)
		results = np.zeros((len(variables), len(states)))
		# results = {}
		# [results.update({x: []}) for x in range(len(variables))]

		for state_id, state in enumerate(states):
			if state == 'Andaman & Nicobar': state = 'Andaman and Nicobar Islands'
			if state == 'Puducherry': state = 'Pudducherry'
			idxs = np.where(data_states == state)
			if np.size(idxs) > 0:
				for var_id in range(len(variables)):
					# results[var_id].append(np.mean(variables[var_id][idxs[0]]))
					results[var_id, state_id] = np.mean(variables[var_id][idxs[0]])
			else:
				for var_id in range(len(variables)):
					# results[var_id].append(0.0)
					results[var_id, state_id] = 0.0
		
		# results = [np.array(x) for x in list(results.values())]
		results = [results[x,:] for x in range(results.shape[0])]
		return results

	def avg_up_district_variables(self, districts, variables):
		data_districts = self.read_data(self.district_names_id)
		# result = []
		results = np.zeros((len(variables), len(districts)))
		for district_id, district in enumerate(districts):
			if district == 'Ahmednagar': district = 'Ahmadnagar'
			if district == 'Garhchiroli': district = 'Gadchiroli'
			idxs = np.where(data_districts == district)
			if np.size(idxs) > 0:
				# result.append(np.mean(variables[idxs[0]]))
				for var_id in range(len(variables)):
					# results[var_id].append(np.mean(variables[var_id][idxs[0]]))
					results[var_id, district_id] = np.mean(variables[var_id][idxs[0]])
			else:
				# result.append(0.0)
				for var_id in range(len(variables)):
					# results[var_id].append(0.0)
					results[var_id, district_id] = 0.0
		results = [results[x,:] for x in range(results.shape[0])]
		return results

	def find_village_data(self, villages, variables, district):
		if district == 'Nasik': district = 'Nashik'
		data_districts = self.read_data(self.district_names_id)
		data_villages = self.read_data(self.village_names_id)

		result = []
		d_idx = np.where(data_districts == district)
		
		for village in villages:
			v_idx = np.where(data_villages == village)
			if np.size(v_idx) == 0:
				result.append(0)
			else:
				added = False
				for ii in v_idx[0]:
					if ii in d_idx[0]:
						result.append(variables[ii])
						added = True
						break
				if not added:
					result.append(0)

		return np.array(result).reshape(-1)

class Regressor:
	def __init__(self, dataset, vars_dict, app, category='state'):
		self.dataset = dataset
		self.vars_dict = vars_dict
		self.app = app
		self.category = category
		if self.category == 'state': 
			self.func = self.dataset.avg_up_state_variables
			self.map_boundaries = self.app.states
		elif self.category == 'district': 
			self.func = self.dataset.avg_up_district_variables
			self.map_boundaries = self.app.districts

	def __call__(self, input_id, output_id, metric=False):
		y_id = self.vars_dict[output_id]
		if len(input_id) == 0:
			x_id = list(self.vars_dict.values())[0]
			x_data, y_data, y_pred, corr = self.find_regression(x_id, y_id)
			x_data, y_data, y_pred = self.func(self.map_boundaries, [x_data, y_data, y_pred])
			x_data = x_data.reshape(-1,1)
		elif len(input_id) == 1:
			x_id = self.vars_dict[input_id[0]]
			x_data, y_data, y_pred, corr = self.find_regression(x_id, y_id)
			x_data, y_data, y_pred = self.func(self.map_boundaries, [x_data, y_data, y_pred])
			x_data = x_data.reshape(-1,1)
		else:
			x_id = [self.vars_dict[ii] for ii in input_id]
			x_data, y_data, y_pred, corr = self.find_multivariate_regression(x_id, y_id)
			x_data = np.array([self.func(self.map_boundaries, [x]) for x in x_data])
			y_data, y_pred = self.func(self.map_boundaries, [y_data, y_pred])
			x_data = x_data.T
			x_data = x_data[:,0,:]
		return x_data, y_data, y_pred, corr

	def find_multivariate_regression(self, x_id, y_id, metric=False):
		x_data = self.dataset.read_data(x_id)
		x_data = np.array([self.dataset.replaceNA(x.astype(np.float32), 0.0) for x in x_data])
		y_data = self.dataset.replaceNA(self.dataset.read_data(y_id).astype(np.float32), 0.0)

		from sklearn.linear_model import LinearRegression
		model = LinearRegression().fit(x_data.T, y_data)
		y_pred = model.predict(x_data.T)
		return x_data, y_data, y_pred, model.score(x_data.T, y_data)

	def find_regression(self, x_id, y_id, correlation=False):
		x_data = self.dataset.replaceNA(self.dataset.read_data(x_id).astype(np.float32), 0.0)
		y_data = self.dataset.replaceNA(self.dataset.read_data(y_id).astype(np.float32), 0.0)

		slope, intercept, corr, _, _ = linregress(x_data, y_data)
		y_pred = slope*x_data + intercept
		if correlation: print("Correlation: {}".format(corr))
		return x_data, y_data, y_pred, corr



class App:
	def __init__(self):
		self.india_json = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
		self.india_choro_csv = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/active_cases_2020-07-17_0800.csv"
		
		self.mh_json = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/maharashtra_districts.json"
		self.ip_variables = ['Toilet Facilities', 'Women\'s Literacy', 'Cooking Fuel']

		# Allocate colors.
		self.colorscales = px.colors.named_colorscales()
		self.load_json_files()
		self.load_csv()
		self.define_app_structure()

	def load_json_files(self):
		with urlopen(self.india_json) as response:
			self.india_states_json = json.load(response)

		resp = requests.get(self.mh_json)
		self.mh_districts_json = json.loads(resp.text)

	def load_csv(self):
		self.districts = [self.mh_districts_json['features'][i]['properties']['name_2'] for i in range(len(self.mh_districts_json['features']))]
		df = pd.read_csv(self.india_choro_csv)
		lists = [[row[col] for col in df.columns] for row in df.to_dict('records')]
		self.states = [x[0] for x in lists]

	@staticmethod
	def create_dataframe(data):
		df = DataFrame(data)
		return df

	def define_app_structure(self):
		self.app = Dash(__name__)
		self.app.layout = html.Div([
			html.H1("Choropleth Maps for Linear Regression", style={'text-align': 'center'}),
			html.Br(),
			html.Div([
				html.P("Input Variables for India: "),
				html.P("Prediction Variable for India: "),
			], style={'width': '24%', 'display': 'inline-block'}),
			
			html.Div([
				dcc.Dropdown(id='input_id_india',
						options = [{'label': variable, 'value': variable} for variable in list(state_vars.keys())[:state_vars_op]],
						multi=True,
						value=[list(state_vars.keys())[0]],
						style={'width':"100%"}
						),
				html.Br(),
				dcc.Dropdown(id='output_id_india',
						options = [{'label': variable, 'value': variable} for variable in list(state_vars.keys())[state_vars_op:]],
						multi=False,
						value=list(state_vars.keys())[state_vars_op],
						style={'width':"100%"}
						),
			], style={'width': '75%', 'float': 'right', 'display': 'inline-block'}),

			html.Hr(),
			
			html.Div([
				html.P("Select IP/OP/Prediction: "),
				html.P("Select Color: ")
			], style={'width': '48%', 'display': 'inline-block'}),

			html.Div([
				dcc.RadioItems(id='display_choice', 
						options=[{'label': x, 'value': x} for x in ['Input', 'Prediction']],
						value='Prediction',
						style={'width':"47%"},
						labelStyle={'display': 'inline-block'}
						),
				html.Br(),
				dcc.Dropdown(id='colorscale', 
						options=[{"value": x, "label": x} for x in self.colorscales],
						value='greys',
						style={'width':"47%"}
						),
			], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
			html.Br(),
			html.Hr(),
			html.Br(),
			html.P("The Correlation Factor is []", id='india_corr'),
			dcc.Graph(id='India_Map', style={'width': '120'}, figure={}),

			html.Hr(),
			html.Div([
				html.P("Input Variables for Maharashtra: "),
				html.P("Prediction Variable for Maharashtra: "),
			], style={'width': '24%', 'display': 'inline-block'}),

			html.Div([
				dcc.Dropdown(id='input_id_mh',
						options = [{'label': variable, 'value': variable} for variable in list(district_vars.keys())[:district_vars_op]],
						multi=True,
						value=[list(district_vars.keys())[0]],
						style={'width':"100%"}
						),
				html.Br(),
				dcc.Dropdown(id='output_id_mh',
						options = [{'label': variable, 'value': variable} for variable in list(district_vars.keys())[district_vars_op:]],
						multi=False,
						value=list(district_vars.keys())[district_vars_op],
						style={'width':"100%"}
						),
			], style={'width': '75%', 'float': 'right', 'display': 'inline-block'}),

			html.Hr(),
			# html.Br(),
			html.Div([
				html.P("The Correlation Factor is []", id='state_corr'),
				html.Br(),
				dcc.Graph(id='State_Map', style={'width': '120'}, figure={}),
			], style={'width': '48%', 'float': 'left'}),
			html.Div([
				dcc.Dropdown(id='select_district',
					options = [{"value": x, "label": x} for x in district_names],
					value = 'Dhule',
					style={'width': "90%"}),
				dcc.Dropdown(id='input_id_district',
					options = [{'label': variable, 'value': variable} for variable in list(district_vars.keys())],
					multi=False,
					value=list(district_vars.keys())[0],
					style={'width': "90%"}),
				dcc.Graph(id='District_Map', style={'width': '120'}, figure={})
			], style={'width': '48%', 'float': 'right'}),
		])


class UpdateClass:
	def __init__(self, app):
		self.app = app
		dataset = Dataset(mh_csv)
		self.mh_regressor = Regressor(dataset, district_vars, app, category='district')

		dataset = Dataset(india_csv)
		self.india_regressor = Regressor(dataset, state_vars, app, category='state')

		self.input_id_india = None
		self.output_id_india = None
		self.input_id_mh = None
		self.output_id_mh = None
		self.select_district = None
		self.input_id_district = list(district_vars.keys())[0]
		self.display_choice = 'Prediction'
		self.colorscale = 'greys'
		self.updated = False

	def __call__(self, input_id_india, output_id_india, input_id_mh, output_id_mh, 
				display_choice, colorscale, select_district, input_id_district):

		if input_id_india != self.input_id_india or output_id_india != self.output_id_india:
			self.update_india_map(input_id_india, output_id_india)
			self.input_id_india, self.output_id_india = input_id_india, output_id_india

		if input_id_mh != self.input_id_mh or output_id_mh != self.output_id_mh:
			self.update_state_map(input_id_mh, output_id_mh)
			self.input_id_mh, self.output_id_mh = input_id_mh, output_id_mh

		if display_choice != self.display_choice:
			self.update_display_choice(display_choice)
			self.display_choice = display_choice

		if colorscale != self.colorscale:
			self.update_colorscale(colorscale)
			self.colorscale = colorscale

		if self.select_district != select_district:
			self.update_village_map(select_district)
			self.select_district = select_district

		if self.input_id_district != input_id_district:
			self.update_village_map_data(input_id_district)
			self.input_id_district = input_id_district

		self.states_map, self.districts_map, self.villages_map = self.update_goes([self.states_map, self.districts_map, self.villages_map])
		self.states_map, self.districts_map, self.villages_map = self.update_margin([self.states_map, self.districts_map, self.villages_map])
		self.districts_map, self.villages_map = self.update_layout([self.districts_map, self.villages_map])

		return self.states_map, self.districts_map, self.villages_map, self.india_corr, self.state_corr

	def update_goes(self, inputs):
		return [x.update_geos(fitbounds='locations', visible=False) for x in inputs]

	def update_margin(self, inputs):
		return [x.update_layout(margin={"r":0,"t":0,"l":0,"b":0,"pad":0}) for x in inputs]

	def update_layout(self, inputs):
		return [x.update_layout(coloraxis_colorbar=dict(
				thicknessmode="pixels", thickness=10,
				lenmode="pixels", len=200, x=0.05, xpad=0
				), width=800, height=800) for x in inputs]

	def update_india_map(self, input_id_india, output_id_india):
		x_data, y_data, y_pred, corr = self.india_regressor(input_id_india, output_id_india)

		self.india_corr = "The Correlation Factor is {}".format(corr)

		data = {'state': self.app.states, 'variable': y_pred, 'output_pred': y_pred, 'input': x_data[:,0]}
		
		for idx, iid in enumerate(input_id_india):
			data[iid] = x_data[:,idx]
		
		self.df_india = App.create_dataframe(data)

		if self.display_choice == 'Input':
			hover_data = ['output_pred']
			self.df_india['variable'] = self.df_india['input']
		if self.display_choice == 'Prediction':
			hover_data = input_id_india
			self.df_india['variable'] = self.df_india['output_pred']


		self.states_map = px.choropleth(self.df_india,
							geojson=self.app.india_states_json,
							featureidkey='properties.ST_NM',
							locations='state',
							color='variable',
							color_continuous_scale=self.colorscale,
							hover_data=hover_data
							)

	def update_state_map(self, input_id_mh, output_id_mh):
		x_data, y_data, y_pred, corr = self.mh_regressor(input_id_mh, output_id_mh)
		self.state_corr = "The Correlation Factor is {}".format(corr)

		data = {'district': self.app.districts, 'variable': y_pred, 'output_pred': y_pred, 'input': x_data[:, 0]}
		for idx, iid in enumerate(input_id_mh):
			data[iid] = x_data[:,idx]

		self.df_state = App.create_dataframe(data)

		if self.display_choice == 'Input':
			hover_data = ['output_pred']
			self.df_state['variable'] = self.df_state['input']
		if self.display_choice == 'Prediction':
			hover_data = input_id_mh
			self.df_state['variable'] = self.df_state['output_pred']

		# import ipdb; ipdb.set_trace()

		self.districts_map = px.choropleth(self.df_state,
							geojson=self.app.mh_districts_json,
							featureidkey='properties.name_2',
							locations='district',
							color='variable',
							color_continuous_scale=self.colorscale,
							hover_data=hover_data
							)

	def update_village_map(self, select_district):
		district_idx = np.where(select_district == district_names)[0][0]
		district_json_file = district_json_files[district_idx]

		resp = requests.get(district_json_file)
		self.mh_villages_json = json.loads(resp.text)

		self.villages = [self.mh_villages_json['features'][i]['properties']['GPNAME'] for i in range(len(self.mh_villages_json['features']))]

		x_data = self.mh_regressor.dataset.read_data(district_vars[self.input_id_district])
		x_data = self.mh_regressor.dataset.replaceNA(x_data.astype(np.float32), 0.0)
		x_data = self.mh_regressor.dataset.find_village_data(self.villages, x_data, select_district)

		data = {'villages': self.villages, 'variable': x_data}

		self.df_village = App.create_dataframe(data)
		
		self.villages_map = px.choropleth(self.df_village,
							geojson=self.mh_villages_json,
							featureidkey='properties.GPNAME',
							locations='villages',
							color='variable',
							color_continuous_scale=self.colorscale,
							# hover_data=hover_data
							)

	def update_village_map_data(self, input_id_district):
		x_data = self.mh_regressor.dataset.read_data(district_vars[input_id_district])
		x_data = self.mh_regressor.dataset.replaceNA(x_data.astype(np.float32), 0.0)
		x_data = self.mh_regressor.dataset.find_village_data(self.villages, x_data, self.select_district)

		data = {'villages': self.villages, 'variable': x_data}

		self.df_village = App.create_dataframe(data)
		
		self.villages_map = px.choropleth(self.df_village,
							geojson=self.mh_villages_json,
							featureidkey='properties.GPNAME',
							locations='villages',
							color='variable',
							color_continuous_scale=self.colorscale,
							# hover_data=hover_data
							)		


	def update_display_choice(self, display_choice):
		if display_choice == 'Input':
			hover_data = ['output_pred']
			self.df_india['variable'] = self.df_india['input']
		if display_choice == 'Prediction':
			hover_data = self.input_id_india
			self.df_india['variable'] = self.df_india['output_pred']

		self.states_map = px.choropleth(self.df_india,
							geojson=self.app.india_states_json,
							featureidkey='properties.ST_NM',
							locations='state',
							color='variable',
							color_continuous_scale=self.colorscale,
							hover_data=hover_data
							)

		if display_choice == 'Input':
			hover_data = ['output_pred']
			self.df_state['variable'] = self.df_state['input']
		if display_choice == 'Prediction':
			hover_data = self.input_id_mh
			self.df_state['variable'] = self.df_state['output_pred']

		self.districts_map = px.choropleth(self.df_state,
							geojson=self.app.mh_districts_json,
							featureidkey='properties.name_2',
							locations='district',
							color='variable',
							color_continuous_scale=self.colorscale,
							hover_data=hover_data
							)

	def update_colorscale(self, colorscale):
		if self.display_choice == 'Input':
			hover_data = ['output_pred']
		if self.display_choice == 'Prediction':
			hover_data = self.input_id_india

		self.states_map = px.choropleth(self.df_india,
							geojson=self.app.india_states_json,
							featureidkey='properties.ST_NM',
							locations='state',
							color='variable',
							color_continuous_scale=colorscale,
							hover_data=hover_data
							)

		if self.display_choice == 'Prediction':
			hover_data = self.input_id_mh

		self.districts_map = px.choropleth(self.df_state,
							geojson=self.app.mh_districts_json,
							featureidkey='properties.name_2',
							locations='district',
							color='variable',
							color_continuous_scale=colorscale,
							hover_data=hover_data
							)

		self.villages_map = px.choropleth(self.df_village,
							geojson=self.mh_villages_json,
							featureidkey='properties.GPNAME',
							locations='villages',
							color='variable',
							color_continuous_scale=colorscale,
							# hover_data=hover_data
							)

app = App()
update = UpdateClass(app)

@app.app.callback(
			[Output(component_id='India_Map', component_property='figure'),				# India's Map.
			 Output(component_id='State_Map', component_property='figure'),				# Currently MH Map.
			 Output(component_id='District_Map', component_property='figure'),			# Any district in MH.
			 Output(component_id='india_corr', component_property='children'),			# Correlation factor for Indian Map
			 Output(component_id='state_corr', component_property='children')],			# Correlation factor for MH Map
			[Input(component_id='input_id_india', component_property='value'),			# Input variable for India Map
			 Input(component_id='output_id_india', component_property='value'),			# Prediction variable for India Map
			 Input(component_id='input_id_mh', component_property='value'),				# Input variable for MH Map
			 Input(component_id='output_id_mh', component_property='value'),			# Prediction variable for MH Map
			 Input(component_id='display_choice', component_property='value'),			# Display Input or Prediction Variable
			 Input(component_id='colorscale', component_property='value'),				# Choose color combination
			 Input(component_id='select_district', component_property='value'),			# Choose district from MH Map
			 Input(component_id='input_id_district', component_property='value')]		# Variable to be displayed on MH Map
		)

def update_graph(input_id_india, output_id_india, input_id_mh, output_id_mh, 
				 display_choice, colorscale, select_district, input_id_district):
	return update(input_id_india, output_id_india, input_id_mh, output_id_mh, 
				  display_choice, colorscale, select_district, input_id_district)

app.app.run_server()