from flask import Blueprint, render_template, jsonify
from GaiaBotFlask.service import controllerMethods

controls = Blueprint('controls', __name__)

@controls.route('/turn/off', methods=['GET'])
def turnOff():
    result = controllerMethods.turnOff()
    print(result)
    return jsonify(result)
    
    
@controls.route('/turn/on', methods=['GET'])
def turnOn():
    print("hit")
    result = controllerMethods.turnOn()
    print(result)
    return jsonify(result)