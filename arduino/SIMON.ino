#include <Adafruit_NeoPixel.h>

#include <FastLED.h>  //https://github.com/FastLED/FastLED

//Constants
#define LED_COUNT 4
#define BRIGHTNESS 600
#define LED_TYPE WS2811
#define COLOR_ORDER BRG  //RGB
#define FASTLED_ALLOW_INTERRUPTS 0
#define FASTLED_INTERRUPT_RETRY_COUNT 1

//Parameters
const int digitalPin = 5;

// Define delay times
const int TURN_DELAY = 1000;   // Delay between turns
const int FLASH_DELAY = 2000;  // Delay for flashing LEDs

// Define the maximum number of steps in the game
const int MAX_STEPS = 20;

// Array to store the sequence of steps
int sequence[MAX_STEPS];
int sequenceLength = 0;
int answerIndex = 0;


//Objects
CRGB leds[LED_COUNT];                                                         // Store strip led state
CRGB colors_array[4] = { CRGB::Green, CRGB::Red, CRGB::Yellow, CRGB::Blue };  // Store available colors for each LED

void setup() {
  Serial.begin(9600);
  Serial.println("Initialize System");


  //Initialize FastLED
  FastLED.addLeds<LED_TYPE, digitalPin, COLOR_ORDER>(leds, LED_COUNT);
  FastLED.setBrightness(BRIGHTNESS);
  FastLED.show();

  // Seed random number generator
  randomSeed(analogRead(0));

  //Turn off everything to start in a clean state
  turnOffEverything();

  // Start the game
  startGame();
}

void loop() {
  // Add a new Step to the sequence
  addStepToSequence();
  playSequence();

  turnOnEverything();

  Serial.println("Check");
  // Wait for input from Python program
  while (!Serial.available()) {
  }

  // Read the input from Python program
  char buttonPressed = Serial.read();
  validateInput(buttonPressed);
  delay(2000);
}

// Function to turn off every LED
void turnOffEverything() {
  for (int i = 0; i < LED_COUNT; i++) {
    leds[i].setRGB(0, 0, 0);  // Set LED color to off
  }

  FastLED.show();  // Update the LED strip

}

void turnOnEverything() {
  lightLED(0, colors_array[0]);
  lightLED(1, colors_array[1]);
  lightLED(2, colors_array[2]);
  lightLED(3, colors_array[3]);
}

// Function to start the game
void startGame() {
  Serial.println("Starting game");
  answerIndex = 0;

  // Starts the game with a sequence of one step and plays it
  sequenceLength = 0;
}

// Function to add a new step to the sequence
void addStepToSequence() {
  sequence[sequenceLength] = random(0, LED_COUNT);  // Generate random step (LED index)
  sequenceLength++;
}

// Function to play the sequence of steps
void playSequence() {
  answerIndex = 0;
  for (int i = 0; i < sequenceLength; i++) {
    flashLED(sequence[i], colors_array[sequence[i]], 1);
    delay(TURN_DELAY);
  }
}

//Function to light up a LED
void lightLED(int ledIndex, CRGB color) {
  leds[ledIndex] = color;
  FastLED.show();  // Update the LED strip
}

// Function to flash a single LED
void flashLED(int ledIndex, CRGB color, int times) {

  for (int i = 0; i < times; i++) {
    lightLED(ledIndex, color);
    delay(FLASH_DELAY);
    turnOffLED(ledIndex);
  }
}

void flashOnlyLED(int ledIndex){
  for (int i = 0; i < 4; i++){
    if(i != ledIndex){
      leds[i] = CRGB::Black;
    }
  }
  FastLED.show();
  delay(FLASH_DELAY);

  turnOnEverything();
}

void flashAllLED( int times) {
    for (int i = 0; i < times; i++) {
    lightLED(0, colors_array[0]);
    lightLED(1, colors_array[1]);
    lightLED(2, colors_array[2]);
    lightLED(3, colors_array[3]);

    delay(200);
    turnOffEverything();
    delay(200);
  }
}

void flashAllLED(CRGB color, int times) {
  delay(500);

  for (int i = 0; i < times; i++) {
    lightLED(0, color);
    lightLED(1, color);
    lightLED(2, color);
    lightLED(3, color);

    delay(200);
    turnOffEverything();
    delay(200);
  }
}

void turnOffLED(int ledIndex) {
  leds[ledIndex] = CRGB::Black;
  FastLED.show();  // Update the LED strip
}

bool validateInput(char buttonPressed) {
  int answer = buttonPressed - '0';
  flashOnlyLED(answer);
  int correctAnswer = sequence[answerIndex];

  if (answer == correctAnswer) {
    answerIndex++;
    if (answerIndex >= sequenceLength) {
      flashAllLED(CRGB::Green, 3);
    }else{
    
      Serial.println("Check");
      // Wait for input from Python program
      while (!Serial.available()) {
      }

      // Read the input from Python program
      char buttonPressed = Serial.read();
      validateInput(buttonPressed);
    }
  } else {
    flashAllLED(CRGB::Red, 3);
    delay(500);
    flashAllLED(5);
    startGame();
  }
  // if (buttonPressed == sequence[sequenceLength - 1]) {
  //   sequenceLength++;
  //   if (sequenceLength > sizeof(sequence)) {
  //     sequenceLength = 0;
  //   }
  // } else {
  //   sequenceLength = 0;
  // }
}


void serialEvent() {
  while (Serial.available()) {
    char buttonPressed = Serial.read();
    validateInput(buttonPressed);
  }
}

bool waitForAnswer() {
  while (!Serial.available()) {
    char buttonPressed = Serial.read();
    validateInput(buttonPressed);
  }
}