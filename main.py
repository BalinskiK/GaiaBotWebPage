from GaiaBotFlask import create_app
from flask_socketio import SocketIO

app = create_app()
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_to_pi')
def send_request_to_pi(data):
    print('Sending request to Raspberry Pi:', data)
    # Add logic to handle the request and control the robot

if __name__ == '__main__':
    socketio.run(app, debug=True, host="0.0.0.0", port=8000)
