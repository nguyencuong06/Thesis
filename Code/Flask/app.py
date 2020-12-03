import os
import re
import logging
import requests
#import yaml
import datetime
from flask import Flask, render_template, url_for, g, request, send_from_directory, abort, request_started, jsonify
from flask_cors import CORS
from flask_restful import Resource, reqparse
from flask_restful_swagger_2 import Api, swagger, Schema
from flask_json import FlaskJSON, json_response
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import Neo4jError, ServiceUnavailable

import neo4j.time

DATABASE_USERNAME = ('neo4j')
DATABASE_PASSWORD = ('seventies-shipping-checkouts')
DATABASE_URL = ('bolt://100.25.31.222:33089')
driver = GraphDatabase.driver(DATABASE_URL, auth=basic_auth(DATABASE_USERNAME, str(DATABASE_PASSWORD)))
app = Flask(__name__)
api = Api(app, title='DemoNeo4j', api_version='1.1.2')
app.config['SECRET_KEY'] = ('super secret guy')

'''current_path = os.getcwd()
print("Current directory is: " + current_path)
path_to_yaml = os.path.join(current_path + 'deploy_web_config.yml')
print("Path to yaml" + path_to_yaml)
try:
    with open('./deploy_web_config.yml', 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print("Error reading  the config file")
#set filename
pipeline1_filename = config['file_names']['pipeline1_filename']
pipeline2_filename = config['file_names']['pipeline2_filename']
model_filename = config['file_names']['model_filename']

#other params
debug_on = config['general']['debug_on']
logging_level = config['general']['logging_level']
BATCH_SIZE = config['general']['BATCH_SIZE']

def get_path(subpath):
    rawpath = os.getcwd()
    path = os.path.abspath(os.path.join(rawpath,'..',subpath))

pipeline_path = get_path('pipeline')    
pipeline1_path = os.path.join(pipeline_path, pipeline1_filename)
pipeline2_path = os.path.join(pipeline_pth, pipeline2_filename)
model_path = os.path.join(get_path('models'), models_filename)'''

@app.route('/', methods=['GET'])
def final():
    #session =  driver.session()
    #friends = session.read_transaction(get_data, "Cameron Crowe")        
    return render_template('base.html')

@app.route('/')
def about():
    #Get value from front-end
    value = {}
    value['Head'] = request.args.get('head')
    value['Relation'] = request.args.get('rela')
    value['Tail'] = request.args.get('tail')
    for ele in value:
        logging.warning("value for "+ele+" is: "+str(value[ele]))

if __name__ == "__main__":
    app.run(debug=True)  