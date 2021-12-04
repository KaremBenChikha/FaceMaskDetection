int inByte = 0;  // initialize the variable inByte
const int ledPin = 13;       // pin that the LED is attached to

void setup(){
 
  pinMode(ledPin, OUTPUT);  // initialize the LED pin as an output
  Serial.begin(57600);  // set serial monitor to same speed
}
void loop(){
  if (Serial.available()>0) {  // check if any data received
    inByte = Serial.read(); // yes, so read it from incoming buffer
    if (inByte == 1){
    digitalWrite(ledPin, HIGH);
  }
  else {
    digitalWrite(ledPin,LOW);
    }
  } 
}