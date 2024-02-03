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
                on : $("#on-button"),
                off : $("#off-button")
            }

            resolve(true)
        })
    }

    initialiseEvents(){
        let self = this;
        
        return new Promise(function(resolve, reject) {
            // Button 1
            self.domItems.on.on("click", async function() {
                await self.turnOn();
            });

            // Button 2
            self.domItems.off.on("click", async function() {
                await self.turnOff();
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

    turnOffBot(){
        let self = this;
        return new Promise(async function(resolve,reject){{
            try{
                let result = await self.turnOff()
                // Do something on success
                console.log(result)

            }catch(result){
                // Do something on failure
            }
            resolve(true)
        }})
    }

    turnOnBot(){
        let self = this;
        return new Promise(async function(resolve,reject){{
            try{
                let result = await self.turnOn()
                // Do something on success
                console.log(result)
            }catch(result){
                // Do something on failure
            }
            resolve(true)
        }})
    }



    /*
     * END: Methods
    */




    /*
     * END: Endpoints
    */

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
    

    /*
     * END: Endpoints
    */
}


//Runs on start
$(function(){
    let controls = new gaiaBotController();
})
