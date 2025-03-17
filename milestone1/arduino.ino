#define IN1 9  // H-Bridge Input 1
#define IN2 10 // H-Bridge Input 2
#define ENA 5  // H-Bridge Enable pin (PWM for speed control)

void setup() {
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(ENA, OUTPUT);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
}

void loop() {
    
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    delay(2000);
    digitalWrite(IN2, LOW);
    digitalWrite(IN1, HIGH);
    delay(2000);
    
}