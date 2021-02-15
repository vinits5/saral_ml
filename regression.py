import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, linregress

state_vars = {
	'Toilet_Facility': 24,
	'Cooking_Fuel': 25,
	'Women_Literacy': 8
}

district_vars = {
	'Geographical_Area': 4,
	'Population': 5,
	'Govt_Primary_Schools': 6,
	'Untreated_Tap_Water': 62,
}


class Regression:
	def __init__(self, data='state'):
		if data=='state':
			csv_path = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/Indian_Data.csv"
			csv_path = 'datasets/Indian_Data.csv'
			self.vars = state_vars
		if data=='district':
			csv_path = "https://raw.githubusercontent.com/vinits5/saral_ml/main/datasets/MH_District_Data.csv"
			csv_path = 'datasets/MH_District_Data.csv'
			self.vars = district_vars

		self.data_df = pd.read_csv(csv_path, low_memory=False, dtype=str)

	def __call__(self, x_label, y_label, regr_type='logitstic'):
		self.regr_type = regr_type
		self.x_label, self.y_label = x_label, y_label
		x_id, y_id = self.vars[x_label], self.vars[y_label]
		data_pts = self.data_df[[str(x_id), str(y_id)]].to_numpy()[1:]
		data_pts = data_pts.T 
		x_data, y_data = data_pts[0], data_pts[1]
		self.x_data = self.replaceNA(x_data.astype(np.float32), 0)

		if regr_type == 'linear':
			self.y_data = self.replaceNA(y_data.astype(np.float32), 0)
			self.y_pred, self.corr = self.linear_regression(self.x_data, self.y_data)
		elif regr_type == 'logitstic':
			self.y_data = self.replaceNA(y_data.astype(np.float32), 2)
			self.y_data = (self.y_data - 2)*(-1)
			self.y_data = self.y_data.astype(np.int32)
			self.y_pred, self.corr = self.logitstic_regression(self.x_data, self.y_data)
		return self.x_data, self.y_data, self.y_pred, self.corr

	@staticmethod
	def linear_regression(x_data, y_data):
		slope, intercept, corr, _, _ = linregress(x_data, y_data)
		y_pred = slope*x_data + intercept
		return y_pred, corr

	@staticmethod
	def logitstic_regression(x_data, y_data):
		model = LogisticRegression().fit(x_data.reshape(-1,1), y_data)
		y_pred = model.predict(x_data.reshape(-1,1))
		corr = accuracy_score(y_data, y_pred)
		return y_pred, corr

	@staticmethod
	def replaceNA(data, replace_value):
		idx = np.where(np.isnan(data))
		data[idx[0]] = replace_value
		return data

	def plot(self, save=False):
		import matplotlib.pyplot as plt
		fig=plt.figure(figsize=(12, 8), dpi= 100, facecolor='w', edgecolor='k')

		if self.regr_type == 'linear':
			plt.plot(self.x_data, self.y_pred, color='r', linewidth=5)
			plt.scatter(self.x_data, self.y_data, s=50)
			plt.title('Pearson\'s Correlation: {}'.format(self.corr), fontsize=20)
		elif self.regr_type == 'logitstic':
			colors = ['g' if cc==True else 'r' for cc in self.y_pred == self.y_data]
			plt.scatter(self.x_data, self.y_pred, s=50, c=colors)
			plt.title('Model Accuracy: {}'.format(self.corr), fontsize=20)
			plt.plot([25550, 25550], [0, 1], color='b', linewidth=3, ls='--')
			# plt.xscale('log', base=2.718281)

		plt.tick_params(labelsize=15, width=3, length=10)
		plt.grid(True)
		plt.xlabel(self.x_label, fontsize=20)
		plt.ylabel(self.y_label, fontsize=20)

		if save:
			filename = self.x_label + '_' + self.y_label + '.png'
			filename = os.path.join(save_filename, filename)
			try:
				plt.savefig(filename)
			except OSError as exc:
				if exc.errno == 36:
					filename = x_label[-10:] + '_' + y_label[-10:] + '.png'
					filename = os.path.join(save_filename, filename)
					plt.savefig(filename)
				else:
					raise  # re-raise previously caught exception
		plt.show()

if __name__ == '__main__':
	regr = Regression(data='district')
	x_data, y_data, y_pred, corr = regr('Population', 'Untreated_Tap_Water', regr_type='logitstic')
	print(corr)
	regr.plot()
