#define IN1 9  // H-Bridge Input 1
#define IN2 10 // H-Bridge Input 2
#define ENA 5  // H-Bridge Enable pin (PWM for speed control)
#define POTPIN A3

int x; 
void setup() { 
	Serial.begin(9600); 
	Serial.setTimeout(1); 
  digitalWrite(IN1, LOW);
	digitalWrite(IN2, LOW);
} 
void moveUp(){
  int potValue = analogRead(POTPIN);
  float position = map(potValue, 0, 1023, 0, 12);
  Serial.println("MOVING MONITOR UP");
  
  if(potValue > 630){
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    delay(900);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }
  else {
    Serial.println("monitor can't move up! Max height reached");
    Serial.println(potValue);
  }
}

void moveDown(){
  Serial.println("MOVING MONITOR DOWN");
  digitalWrite(IN1, HIGH);
	digitalWrite(IN2, LOW);
  delay(900);
  digitalWrite(IN1, LOW);
	digitalWrite(IN2, LOW);
}
void vibrate(){
  Serial.println("MONITOR VIBRATING");
  digitalWrite(IN1, HIGH);
	digitalWrite(IN2, LOW);
  delay(400);
  digitalWrite(IN1, LOW);
	digitalWrite(IN2, HIGH);
  delay(400);
  digitalWrite(IN1, LOW);
	digitalWrite(IN2, LOW);
}
// movements = {"up" : 2, "down" : 3, "stop": 4}
void loop() { 
	while (!Serial.available()); 
	
	x = Serial.readString().toInt();
  Serial.print(x);
	if (x == 2) {
		moveDown();
	} else if (x == 3) {
		moveUp();
	} else if (x == 4) {
		digitalWrite(IN1, LOW);
		digitalWrite(IN2, LOW);
	} else if (x == 5) {
		vibrate();
	} else {
		Serial.print("Unexpected Errors");
	}

  Serial.println(Serial.readString());
	x = Serial.readString().toInt();  
}



