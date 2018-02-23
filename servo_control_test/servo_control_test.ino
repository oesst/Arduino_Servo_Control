  /* -----------------------------------------------------------------------------
   Example .ino file for arduino, compiled with CmdMessenger.h and
   CmdMessenger.cpp in the sketch directory.
  ----------------------------------------------------------------------------*/

#include "CmdMessenger.h"
#include <Servo.h>

/* Constants */
// min and max puls width for 270 degree servo (tilt)
const int MIN_P_270 = 500;
const int MAX_P_270 = 2500;
// define rotation speed for 270 degree servo (tilt)
const int ROT_SPEED_270 = 80;

// min and max puls width for 10 degree servo (tilt)
// Perfectly fitting for trafo power supply 6.4V 
const int MIN_P_180 = 800;
const int MAX_P_180 = 2150;
// define rotation speed for 180 degree servo (tilt)
const int ROT_SPEED_180 = 100;

// rule of three to define microseconds per one degree
const double SIG_180 = (MAX_P_180 - MIN_P_180) / 180.0;

// rule of three to define microseconds per one degree
const double SIG_270 = (MAX_P_270 - MIN_P_270) / 270.0;

/* Initialize servos */
int servo_180_pin = 8;
int servo_270_pin = 7;

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
Servo servo_270;
Servo servo_180;


/* initialize some storage values */
int current_pan_angle = 0;
int current_tilt_angle = 0;


/* Create callback functions to deal with incoming messages */


/* callback */
void on_connection(void) {
  c.sendBinCmd(is_connected, 1);
}

/* callback */
void on_pan_angle(void) {

  /* Get servo control angle */
  int angle = c.readBinArg<int>();

  // move counter clock wise
  if (current_tilt_angle < angle) {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_180 + SIG_180 * current_tilt_angle); pos <= MIN_P_180 + angle * SIG_180; pos += SIG_180) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_180);
      servo_270.writeMicroseconds(pos);
    }

  } // move clock wise
  else {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_180 + SIG_180 * current_tilt_angle); pos >= MIN_P_180 + angle * SIG_180; pos -= SIG_180) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_180);
      servo_270.writeMicroseconds(pos);
    }


  }
  // save current angle to achieve a smooth turning
  current_tilt_angle = angle;

  /* send confirmation */
  c.sendBinCmd(pan_angle_set, angle);
}

/* callback */
void on_tilt_angle(void) {

  /* Get servo control angle */
  int angle = c.readBinArg<int>();
  /* Write angle */
  // move counter clock wise
  if (current_pan_angle < angle) {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_270 + SIG_270 * current_pan_angle); pos <= MIN_P_270 + angle * SIG_270; pos += SIG_270) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_270);
      servo_180.writeMicroseconds(pos);
    }

  } // move clock wise
  else {
    // add min_p since we start at the min pulse width
    for (int pos = (MIN_P_270 + SIG_270 * current_pan_angle); pos >= MIN_P_270 + angle * SIG_270; pos -= SIG_270) {
      // delay to get proper rotation speed
      delay(1000.0 / ROT_SPEED_270);
      servo_180.writeMicroseconds(pos);
    }


  }
  // save current angle to achieve a smooth turning
  current_pan_angle = angle;

  /* send ack back */
  c.sendBinCmd(tilt_angle_set, angle);


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
  servo_180.writeMicroseconds(MIN_P_270);
  servo_270.writeMicroseconds(MIN_P_180);
}

void setup() {
  /* attach servos to pins */
  servo_270.attach(servo_180_pin,MIN_P_180, MAX_P_180);
  // set correct pulse width for 270 degree servo
  servo_180.attach(servo_270_pin, MIN_P_270, MAX_P_270);


  /* reset servo angle */
  reset_all_servos();


  Serial.begin(BAUD_RATE);
  attach_callbacks();
}

void loop() {
  c.feedinSerialData();
}
