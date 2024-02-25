import time
import datetime
from azure.iot.device import IoTHubDeviceClient, MethodResponse

# Required install
# pip install azure-iot-hub

# Additonal info : resp_status : 200 = success
#                              : 500 = unsuccessful
#                              : 404 = method not found                               

# Do not modify
# Connection string responsible for securing connection for azure iot c2d connection
CONNECTION_STRING = "HostName=GaiaBot-rasberryPi.azure-devices.net;DeviceId=rasberryPi2024;SharedAccessKey=iBvdRg7UgMhgOb6fR/NUF1yYGXo1MVz8rAIoTMi0WKg="
 
# Client responsible for initialsing a opening on the device
def create_client():
    
    # Instantiate the client
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

    # Define the handler for method requests
    def method_request_handler(method_request):
        if method_request.name == "StartDevice":
            # Act on the method by starting the device
            print("Starting Device")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            
            # Make sure property names have no spaces in them
            reported_props = {"startTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # For now it sends back the current time to the cloud
            #client.patch_twin_reported_properties(reported_props)
            #print( "Device twins updated with latest start time")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull return 500
            resp_status = 200
            
            # Payload has to be a json property
            resp_payload = {"Response": "This is the response from the device",
                            "startTime": current_time} 
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
            
        elif method_request.name == "StopDevice":
             # Act on the method by stopping the device
            print("Stopping Device")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            # reported_props = {"stopTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # client.patch_twin_reported_properties(reported_props)
            # print("Device twins updated with latest rebootTime")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull set resp_status to 500
            resp_status = 200
            
            # Look above for on method to see info about payload
            resp_payload = {"Response": "This is the response from the device",
                            "Time": current_time}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
            
        # Add elif to continue expanding number of methods / for multiple methods switch to switch rather than if's
            
        else:
            # Create a method response indicating the method request was for an unknown method
            resp_status = 404
            resp_payload = {"Response": "Unknown method"}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)

        # Send the method response
        client.send_method_response(method_response)

    try:
        # Attach the handler to the client
        client.on_method_request_received = method_request_handler
    except:
        # In the event of failure, clean up
        client.shutdown()

    return client

def main():
    print ("Starting the Iot hub for the device")
    client = create_client()

    print ("Waiting for commands, press Ctrl-C to exit")
    try:
        #This should be set to false when the device stops/loses connection
        # Wait for program exit
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        print("IoTHubDeviceClient sample stopped")
    finally:
        # Graceful exit
        print("Shutting down IoT Hub Client")
        client.shutdown()

if __name__ == '__main__':
    main()
    
# Iot client should always aim to be shutdown to avoid issues