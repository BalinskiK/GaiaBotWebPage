//Js goes here


class gaiaBotController {
    /*
     * Constructor
    */


    constructor() {
        this.initialisation();
    }

    /*
     * END: Constructor
    */


    /*
     * Initialisation
    */

    async initialisation() {
        await this.initialiseVariables();
        await this.initialiseEvents();
    }

    initialiseVariables() {
        let self = this;

        return new Promise(function(resolve,reject){
            self.domItems =  {
                logTextarea : $("#log-text-area"),
                duringRunTime : $("#during-run-time"),
                connect : $("#connect-button"),
                connectSection : $("#connect-section"),
                on : $("#on-button"),
                off : $("#off-button"),
                kill : $("#kill-button"),
                test:{
                    on : $("#on-test-button"),
                    off : $("#off-test-button"),
                }
            }

            resolve(true)
        })
    }

    initialiseEvents(){
        let self = this;
        
        return new Promise(function(resolve, reject) {

            self.domItems.connect.on("click", async function() {
                await self.connect();
            });

            // Button 1
            self.domItems.on.on("click", async function() {
                await self.powerBotOn();
            });

            // Button 2
            self.domItems.off.on("click", async function() {
                await self.powerBotOff();
            });

            self.domItems.test.on.on("click", async function() {
                await self.turnOn();
            });

            // Button 2
            self.domItems.test.off.on("click", async function() {
                await self.turnOff();
            });

            $("#on-test-button-maia").on("click", async function() {
                await self.testBaseOn();
            });

            $("#off-test-button-maia").on("click", async function() {
                await self.testBaseOff();
            });

            $("#start-arm-on").on("click", async function() {
                await self.startArm();
            });

            resolve(true);
        });
    }

    /*
     * END: Initialisation
    */


    /*
     * Methods
    */

    connect(){
        let self = this;
        return new Promise(async function (resolve,reject) {

            //call test connection 
            //To implement

            self.domItems.connect.hide().promise().done(function (){
                self.domItems.on.show()
            })

            self.appendToOutputLog("Connected")

          })
    }

    powerBotOff(){
        let self = this;
        return new Promise(async function(resolve,reject){{
            try{
                let result = await self.turnOff()
                // Do something on success
                console.log(result)

                if (result === true){
                    self.domItems.duringRunTime.hide().promise().done(function(){
                        self.domItems.on.show()
                    })


                    self.appendToOutputLog("Powered Off")
                }

            }catch(result){
                // Do something on failure
            }
            resolve(true)
        }})
    }

    powerBotOn(){
        let self = this;
        return new Promise(async function(resolve,reject){{
            try{
                let result = await self.turnOn()
            
                // Do something on success

                if (result === true){
                    self.domItems.on.hide().promise().done(function(){
                        self.domItems.duringRunTime.show()
                    })

                    self.appendToOutputLog("Powered On")
                }
                console.log(result)
            }catch(result){
                // Do something on failure
            }
            resolve(true)
        }})
    }

    appendToOutputLog(content){
        let self = this;
        let logTextarea = self.domItems.logTextarea;
         console.log("init", logTextarea.val())


        if (logTextarea.val()){
            //
        } 
        logTextarea.val(logTextarea.val() + content + '\n' + "        ");
        console.log(logTextarea.val())


        // Scroll to the bottom to show the latest content
        logTextarea.scrollTop(logTextarea[0].scrollHeight);
    }

    /*
     * END: Methods
    */




    /*
     * END: Endpoints
    */
    testBaseOn() {
        let self = this;
        return new Promise(function(resolve, reject) {
            let uri = `../../../../../../turn/on/base`;
            let jsonData = {
                variable1: $("#variable1").val(),
                variable2: $("#variable2").val()
            };
            console.log(jsonData)
    
            $.ajax({
                type: 'POST', // Change type to 'POST'
                url: uri,
                contentType: 'application/json',
                data: JSON.stringify(jsonData), // Send JSON data in the request body
                success: function(result) {
                    resolve(result);
                },
                error: function(result) {
                    reject(result);
                }
            });
        });
    }
    


    testBaseOff(){
        let self = this;
        return new Promise(function(resolve, reject) {
            let uri = `../../../../../../turn/off/base`;
            $.ajax({
                type: 'GET', // Change type to 'GET'
                url: uri,
                contentType: 'application/json',
                success: function(result) {
                    resolve(result);
                },
                error: function(result) {
                    reject(result);
                }
            });
        });
    }

    turnOff() {
        let self = this;
        return new Promise(function(resolve, reject) {
            let uri = `../../../../../../turn/off`;
            $.ajax({
                type: 'GET', // Change type to 'GET'
                url: uri,
                contentType: 'application/json',
                success: function(result) {
                    resolve(result);
                },
                error: function(result) {
                    reject(result);
                }
            });
        });
    }
    
    turnOn() {
        let self = this;
        return new Promise(function(resolve, reject) {
            let uri = `../../../../../../turn/on`;
            $.ajax({
                type: 'GET', // Change type to 'GET'
                url: uri,
                contentType: 'application/json',
                success: function(result) {
                    // Will return a code
                    resolve(result);
                },
                error: function(result) {
                    reject(result);
                }
            });
        });
    }
    

    startArm() {
        let self = this;
        return new Promise(function(resolve, reject) {
            let uri = `../../../../../../start/arm`;
            $.ajax({
                type: 'GET', // Change type to 'GET'
                url: uri,
                contentType: 'application/json',
                success: function(result) {
                    // Will return a code
                    resolve(result);
                },
                error: function(result) {
                    reject(result);
                }
            });
        });
    }

    /*
     * END: Endpoints
    */
}


//Runs on start
$(function(){
    $(".nav .nav-item").on("click", function(){
        $(".nav").find(".active").removeClass("active");
        $(this).addClass("active");
     });

     
    let controls = new gaiaBotController();
})

