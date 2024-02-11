from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def dashboard():
    return render_template('Dashboard.html', data="dashboard")


@app.route('/BrainStrokeDetector')
def BrainStrokeDetector():
    return render_template('BrainStrokeDetector.html')


@app.route('/Dashboard')
def Dashboard():
    return render_template('Dashboard.html')


@app.route('/BrainTumourDetector')
def BrainTumourDetector():
    return render_template('BrainTumourDetector.html')


@app.route('/AlzheimerDiseaseDetector')
def AlzheimerDiseaseDetector():
    return render_template('AlzheimerDiseaseDetector.html')


@app.route('/ReportGenerator')
def ReportGenerator():
    return render_template('ReportGenerator.html')


@app.route('/Register')
def Register():
    return render_template('Register.html')


@app.route('/login')
def login():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)
