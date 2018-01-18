/* -----------------------------------------------------------------------------
 * Example .ino file for arduino, compiled with CmdMessenger.h and
 * CmdMessenger.cpp in the sketch directory. 
 *----------------------------------------------------------------------------*/

#include "CmdMessenger.h"
#include <Servo.h> 


/* Define available CmdMessenger commands */
enum {
    connection,
    is_connected,
    pan_angle,
    pan_angle_set,
    tilt_angle,
    tilt_angle_set,
    acknowledge,
    error,
};

/* Initialize CmdMessenger -- this should match PyCmdMessenger instance */
const int BAUD_RATE = 9600;
CmdMessenger c = CmdMessenger(Serial,',',';','/');

/* Initialize servos */
int pan_servo_pin = 8; 
int tilt_servo_pin = 7; 

/* Create a servo objects */
Servo pan_servo; 
Servo tilt_servo; 


/* Create callback functions to deal with incoming messages */


/* callback */
void on_connection(void){
  c.sendBinCmd(is_connected,1);
}

/* callback */
void on_pan_angle(void){
   
    /* Get servo control angle */
    int angle = c.readBinArg<int>();
    /* Write angle */
    pan_servo.write(angle);

    /* send confirmation */
    c.sendBinCmd(pan_angle_set,angle);
}

/* callback */
void on_tilt_angle(void){
   
    /* Get servo control angle */
    int angle = c.readBinArg<int>();
    /* Write angle */
    tilt_servo.write(angle);

    /* send confirmation */
    c.sendBinCmd(tilt_angle_set,angle);
}

/* callback */
void on_unknown_command(void){
    c.sendCmd(error,"Command without callback.");
}

/* Attach callbacks for CmdMessenger commands */
void attach_callbacks(void) { 
    c.attach(connection, on_connection);
    c.attach(pan_angle,on_pan_angle);
    c.attach(tilt_angle,on_tilt_angle);
    c.attach(on_unknown_command);
}

void setup() {
    /* attach servos to pins */
    pan_servo.attach(pan_servo_pin);
    tilt_servo.attach(tilt_servo_pin);

  
    Serial.begin(BAUD_RATE);
    attach_callbacks();    
}

void loop() {
    c.feedinSerialData();
}

