  /* -----------------------------------------------------------------------------
   Example .ino file for arduino, compiled with CmdMessenger.h and
   CmdMessenger.cpp in the sketch directory.
  ----------------------------------------------------------------------------*/

#include "CmdMessenger.h"
#include <Servo.h>

/* Constants */
// min and max puls width for 270 degree servo (tilt)
const int MIN_P_270 = 500;
const int MAX_P_270 = 2600;
// define rotation speed for 270 degree servo (tilt)
const int ROT_SPEED_270 = 30;

// min and max puls width for 270 degree servo (tilt)
const int MIN_P_180 = 650;
const int MAX_P_180 = 2000;
// define rotation speed for 270 degree servo (tilt)
const int ROT_SPEED_180 = 30;

/* Initialize servos */
int pan_servo_pin = 8;
int tilt_servo_pin = 7;

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
CmdMessenger c = CmdMessenger(Serial, ',', ';', '/');

/* Create a servo objects */
Servo pan_servo;
Servo tilt_servo;


/* initialize some storage values */
int current_tilt_angle = 0;
int current_pan_angle = 0;


/* Create callback functions to deal with incoming messages */


/* callback */
void on_connection(void) {
  c.sendBinCmd(is_connected, 1);
}

/* callback */
void on_pan_angle(void) {

  /* Get servo control angle */
  int angle = c.readBinArg<int>();


 // rule of three to define microseconds per one degree
  const double SIG = (MAX_P_180 - MIN_P_180) / 180.0;


  /* send confirmation */
  c.sendBinCmd(pan_angle_set, angle);


  // move counter clock wise
  if (current_pan_angle < angle) {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_180 + SIG * current_pan_angle); pos <= MIN_P_180 + angle * SIG; pos += SIG) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_180);
      pan_servo.writeMicroseconds(pos);
    }

  } // move clock wise
  else {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_180 + SIG * current_pan_angle); pos >= MIN_P_180 + angle * SIG; pos -= SIG) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_180);
      pan_servo.writeMicroseconds(pos);
    }


  }
  // save current angle to achieve a smooth turning
  current_pan_angle = angle;





  
}

/* callback */
void on_tilt_angle(void) {

  /* Get servo control angle */
  int angle = c.readBinArg<int>();
  /* Write angle */

  // rule of three to define microseconds per one degree
  const double SIG = (MAX_P_270 - MIN_P_270) / 270.0;


  /* send ack back */
  c.sendBinCmd(tilt_angle_set, angle);


  // move counter clock wise
  if (current_tilt_angle < angle) {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_270 + SIG * current_tilt_angle); pos <= MIN_P_270 + angle * SIG; pos += SIG) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_270);
      tilt_servo.writeMicroseconds(pos);
    }

  } // move clock wise
  else {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_270 + SIG * current_tilt_angle); pos >= MIN_P_270 + angle * SIG; pos -= SIG) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_270);
      tilt_servo.writeMicroseconds(pos);
    }


  }
  // save current angle to achieve a smooth turning
  current_tilt_angle = angle;


}

/* callback */
void on_unknown_command(void) {
  c.sendCmd(error, "Command without callback.");
}

/* Attach callbacks for CmdMessenger commands */
void attach_callbacks(void) {
  c.attach(connection, on_connection);
  c.attach(pan_angle, on_pan_angle);
  c.attach(tilt_angle, on_tilt_angle);
  c.attach(on_unknown_command);
}


void reset_all_servos() {
  tilt_servo.writeMicroseconds(MIN_P_270);
  pan_servo.writeMicroseconds(MIN_P_180);
}

void setup() {
  /* attach servos to pins */
  pan_servo.attach(pan_servo_pin,MIN_P_180, MAX_P_180);
  // set correct pulse width for 270 degree servo
  tilt_servo.attach(tilt_servo_pin, MIN_P_270, MAX_P_270);


  /* reset servo angle */
  reset_all_servos();


  Serial.begin(BAUD_RATE);
  attach_callbacks();
}

void loop() {
  c.feedinSerialData();
}

