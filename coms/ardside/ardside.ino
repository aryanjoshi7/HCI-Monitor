#define IN1 9  // H-Bridge Input 1
#define IN2 10 // H-Bridge Input 2
#define ENA 5  // H-Bridge Enable pin (PWM for speed control)


int x; 
void setup() { 
	Serial.begin(115200); 
	Serial.setTimeout(1); 
} 

// movements = {"up" : 2, "down" : 3, "stop": 4}
void loop() { 
	while (!Serial.available()); 
	
	x = Serial.readString().toInt();
  Serial.print(x);
	if (x == 2) {
		digitalWrite(IN1, LOW);
		digitalWrite(IN2, HIGH);
	} else if (x == 3) {
		digitalWrite(IN1, HIGH);
		digitalWrite(IN2, LOW);
	} else if (x == 4) {
		digitalWrite(IN1, LOW);
		digitalWrite(IN2, LOW);
  }
	// } else {
	// 	Serial.print("Unexpected Errors");
	// }

	delay(2000);
	digitalWrite(IN1, LOW);
	digitalWrite(IN2, LOW);
  // Serial.println(Serial.readString());
	// x = Serial.readString().toInt();  
} 
