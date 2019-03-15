from flask import Flask,redirect, request, render_template, session
import os
app = Flask(__name__)

app.secret_key = (os.urandom(16))


@app.route('/')
def home_page(name=None):
    return render_template('index.html', name=name)
