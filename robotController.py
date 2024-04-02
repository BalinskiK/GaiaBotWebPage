import threading
import json
import multiprocessing
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


"""
This is the main controller of Gaiabot and should act as the centeral control unit
This code should run on boot of the bot

- On boot goals
- Set up connection to azure 
- Set up object detection

- On listener event goals
- Start the main program
- Exit the main program
- Return the robot safely

- Dependency injections are not used as we do not require multi reuse

"""

"""
Obejcts
"""

"""
Object that store and deals with object detection
"""

class ObjectDetection:
    
    """
    Constructor
    """
    
    def __init__(self):
        self.LatestDetection = 0
        
        # Start the object detection loop - will need to be made asyncrounous/ await for a return when fully booted
        self.DetectionStart = 0
        
        
    
    """
    END: Constructor
    """    
    
    """
    Getter(s)/Setter(s)
    """ 
    
    def setLatestDetection(self, coordinates):
        self.LatestDetection = coordinates
        
    def setLatestDetection(self):
        self.LatestDetection = null    
     
    def getLatestDetection(self, coordinates):
        return self.LatestDetection
     
    """
    END: Getter(s)/Setter(s)
    """ 
     
    """
    Method(s)
    """ 
    
    
    """
    END: Method(s)
    """
    

"""
END: Obejcts
"""

"""
Method(s)
"""


def start_thread():
    global thread
    thread = threading.Thread(target=mainProgram)
    thread.start()

# Function to stop the threaded process
def stop_thread():
    global thread
    if thread:
        thread.join()  # Wait for the thread to finish
        print("Thread stopped.")
    else:
        print("Thread is not running.")

# The main
def mainProgram():
    
    #Runs on a while loop
    
    #Get to first bush

    while True:
        # start the process
        
        # Have all the strawberries been collected?
        # or collection goal / additional info
        if(True):
            # Base drives off to drop off point
            
            #Trigger door to empty
            
            #End the process
            return
        else:
            # Is basket full 
            if(True):
                # Base drives off to drop off point
            
                #Trigger door to empty
            
                #Go back to previous postion
                return
            else:
                #Get latest object detection coordinates - make sure its the most latest one which will corespond to current position
                
                #Berry on bush - if point == null no berry on bush
                if(True):
                    #Collect
                    #Drop into basket
                    return
                    
                else:
                    #Move to next point
                    return
                        
                    
                
            
    

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
            
            
            # START
            
            #start_thread()
            
            
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
            
            #STOP
            
            #stop_thread();
            
            
            resp_status = 200
            
            # Look above for on method to see info about payload
            resp_payload = {"Response": "This is the response from the device",
                            "Time": current_time}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
            
        elif method_request.name == "StartBase":
             # Act on the method by stopping the device
            print("Starting base")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            # reported_props = {"stopTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # client.patch_twin_reported_properties(reported_props)
            # print("Device twins updated with latest rebootTime")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull set resp_status to 500
            
            #Startbase
            
            #CODE Goes here
            
            # Assuming method_request.payload is a string containing JSON data in dictionary format
            payload_dict = json.loads(method_request.payload)

            # Accessing values from the parsed dictionary
            variable1 = payload_dict["variable1"]
            variable2 = payload_dict["variable2"]


            
            
            
            resp_status = 200
            
            # Look above for on method to see info about payload
            resp_payload = {"Response": "This is the response from the device",
                            "Time": current_time}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
        
        
        elif method_request.name == "StopBase":
             # Act on the method by stopping the device
            print("Stopping Base")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            # reported_props = {"stopTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # client.patch_twin_reported_properties(reported_props)
            # print("Device twins updated with latest rebootTime")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull set resp_status to 500
            
            #StopBase
            
            #CODE Goes here
        
            
            resp_status = 200
            
            # Look above for on method to see info about payload
            resp_payload = {"Response": "This is the response from the device",
                            "Time": current_time}
            method_response = MethodResponse(method_request.request_id, resp_status, resp_payload)
         
            
        # Add elif to continue expanding number of methods / for multiple methods switch to switch rather than if's
            
        elif method_request.name == "StartArm":
             # Act on the method by stopping the device
            print("Starting arm")

            # ...and patching the reported properties
            current_time = str(datetime.datetime.now())
            # reported_props = {"stopTime": current_time}
            
            # This allows you to send back to the cloud without returning a payload - can be used for constant updates 
            # client.patch_twin_reported_properties(reported_props)
            # print("Device twins updated with latest rebootTime")

            # Create a method response indicating the method request was resolved
            # If the function was unsuccessfull set resp_status to 500
            
            #StartBase
            
            #CODE Goes here
        
            
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
    #Boot and initialise object detection
    objectDetection = ObjectDetection()
    
    print ("Starting the IoT Hub Python jobs sample...")
    client = create_client()
    

    print ("IoTHubDeviceClient waiting for commands, press Ctrl-C to exit")
    try:
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        print("IoTHubDeviceClient sample stopped!")
    finally:
        # Graceful exit
        print("Shutting down IoT Hub Client")
        client.shutdown()


if __name__ == '__main__':
    main()
