from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("xgboost.pkl", "rb"))

@app.route('/')
def home():
	return render_template('index.html')

def one_hot(value, total_features):
	lst = [0 for i in range(total_features)]
	try:
		lst[int(value)] = 1
	except:
		pass
	return lst

@app.route('/predict', methods=['POST'])
def predict():
	prediction = ''
	if request.method == 'POST':
		age = [int(request.form['age'])]
		workclass = one_hot(request.form['FormWorkCLassSelect'], 6)
		education = [int(request.form['FormEducationSelect']) + 1]
		education_num = [int(request.form['education_num'])]
		marital_status = one_hot(request.form['FormMartialStatusSelect'], 6)
		occupation = one_hot(request.form['FormOccupationSelect'], 13)
		relationship = one_hot(request.form['FormRelationSelect'], 5)
		gender = one_hot(request.form['FormGenderSelect'], 1)
		hours_per_week = [int(request.form['hours_per_week'])]

		features = age + workclass + education + education_num + marital_status + occupation + relationship + gender + hours_per_week
		
		result = model.predict([features])
		if result[0] == 0:
			prediction = 'Your Salary Will be LESS than 50K as per Statistics'
		else:
			prediction = 'Your Salary Will be MORE than 50K as per Statistics'
		return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
	app.run(debug=True)









