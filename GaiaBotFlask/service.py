import sys, time

from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod, CloudToDeviceMethodResult, Twin

# Do not modify
# In future for saftey related issue this should be stored as secrets
CONNECTION_STRING = "HostName=GaiaBot-rasberryPi.azure-devices.net;SharedAccessKeyName=service;SharedAccessKey=yoGFpkYvPSY3SiwO+LNbyGV0gl83LsQs1AIoTOVZm70="
DEVICE_ID = "rasberryPi2024"

def create_iothub_registry_manager():
    
    try: 
        registry_manager = IoTHubRegistryManager(CONNECTION_STRING)
    except Exception as ex:
        print ( "" )
        print ( "Unexpected error {0}".format(ex) )
        return
    except KeyboardInterrupt:
        print ( "" )
        print ( "IoTHubDeviceMethod sample stopped" )
        
    

def iothub_devicemethod_sample_run(METHOD_NAME, METHOD_PAYLOAD, TIMEOUT, WAIT_COUNT):
    try:
        # Create IoTHubRegistryManager
        registry_manager = IoTHubRegistryManager(CONNECTION_STRING)

        print ( "" )
        print ( "Invoking device to reboot..." )

        # Call the direct method.
        deviceMethod = CloudToDeviceMethod(method_name=METHOD_NAME, payload=METHOD_PAYLOAD)
        response = registry_manager.invoke_device_method(DEVICE_ID, deviceMethod)

        print ( "" )
        print ( "Successfully invoked the device to reboot." )

        print ( "" )
        print ( response.payload )

        # If we are expecting updates this runs repeat
        """
        while True:
            print ( "" )
            print ( "IoTHubClient waiting for commands, press Ctrl-C to exit" )

            status_counter = 0
            while status_counter <= WAIT_COUNT:
                twin_info = registry_manager.get_twin(DEVICE_ID)

                if twin_info.properties.reported.get("rebootTime") != None :
                    print ("Last reboot time: " + twin_info.properties.reported.get("rebootTime"))
                else:
                    print ("Waiting for device to report last reboot time...")

                time.sleep(5)
                status_counter += 1
        """

    except Exception as ex:
        print ( "" )
        print ( "Unexpected error {0}".format(ex) )
        return
    except KeyboardInterrupt:
        print ( "" )
        print ( "IoTHubDeviceMethod sample stopped" )

   


class controllerMethods:
    
    def testAndCreateConnection():
        return
    
    def turnOnBase(int1, int2):
        #call methods or change state to start the robot
        print ( "Starting the IoT Hub Service Client DeviceManagement Python sample..." )
        print ( "    Connection string = {0}".format(CONNECTION_STRING) )
        print ( "    Device ID         = {0}".format(DEVICE_ID) )
        
        METHOD_NAME = "StartBase"
        METHOD_PAYLOAD = "{\"variable1\":\"" + str(variable1) + "\", \"variable2\":\"" + str(variable2) + "\"}"
        TIMEOUT = 60
        WAIT_COUNT = 10
        
        iothub_devicemethod_sample_run(METHOD_NAME, METHOD_PAYLOAD, TIMEOUT, WAIT_COUNT)


        return True

    def turnOffBase():
        #call methods or change state to start the robot
        print ( "Starting the IoT Hub Service Client DeviceManagement Python sample..." )
        print ( "    Connection string = {0}".format(CONNECTION_STRING) )
        print ( "    Device ID         = {0}".format(DEVICE_ID) )
        
        METHOD_NAME = "StopBase"
        METHOD_PAYLOAD = "{\"method_number\":\"42\"}"
        TIMEOUT = 60
        WAIT_COUNT = 10
        
        iothub_devicemethod_sample_run(METHOD_NAME, METHOD_PAYLOAD, TIMEOUT, WAIT_COUNT)


        return True    
    
    def turnOn():
        #call methods or change state to start the robot
        print ( "Starting the IoT Hub Service Client DeviceManagement Python sample..." )
        print ( "    Connection string = {0}".format(CONNECTION_STRING) )
        print ( "    Device ID         = {0}".format(DEVICE_ID) )
        
        METHOD_NAME = "StartDevice"
        METHOD_PAYLOAD = "{\"method_number\":\"42\"}"
        TIMEOUT = 60
        WAIT_COUNT = 10
        
        iothub_devicemethod_sample_run(METHOD_NAME, METHOD_PAYLOAD, TIMEOUT, WAIT_COUNT)
    

        return True
    
    def turnOff():
        #call methods or change state to stop the robot - either kill or stop
        METHOD_NAME = "StopDevice"
        METHOD_PAYLOAD = "{\"method_number\":\"42\"}"
        TIMEOUT = 60
        WAIT_COUNT = 10
        
        print ( "Starting the IoT Hub Service Client DeviceManagement Python sample..." )
        print ( "    Connection string = {0}".format(CONNECTION_STRING) )
        print ( "    Device ID         = {0}".format(DEVICE_ID) )
        iothub_devicemethod_sample_run(METHOD_NAME, METHOD_PAYLOAD, TIMEOUT, WAIT_COUNT)

        return True
    