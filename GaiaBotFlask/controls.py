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

@controls.route('/turn/on/base', methods=['POST'])
def turnOnBase():
    # Get JSON data from the request body
    json_data = request.json
    
    if not json_data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Extract variables from JSON data
    variable1 = json_data.get('variable1')
    variable2 = json_data.get('variable2')
    result = controllerMethods.turnOnBase(variable1, variable2)
    print(result)
    return jsonify(result)


@controls.route('/turn/off/base', methods=['GET'])
def turnOffBase():
    # Get JSON data from the request body

    result = controllerMethods.turnOffBase()
    print(result)
    return jsonify(result)
