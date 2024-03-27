#include <Adafruit_NeoPixel.h>

#include <FastLED.h> //https://github.com/FastLED/FastLED

//Constants
#define LED_COUNT 4
#define BRIGHTNESS 600
#define LED_TYPE WS2811
#define COLOR_ORDER BRG//RGB
#define FASTLED_ALLOW_INTERRUPTS 0
#define FASTLED_INTERRUPT_RETRY_COUNT 1

//Parameters
const int digitalPin  = 5;

// Define delay times
const int TURN_DELAY = 1000; // Delay between turns
const int FLASH_DELAY = 2000; // Delay for flashing LEDs

// Define the maximum number of steps in the game
const int MAX_STEPS = 20;

// Array to store the sequence of steps
int sequence[MAX_STEPS];
int sequenceLength = 0;


//Objects
CRGB leds[LED_COUNT]; // Store strip led state
CRGB colors_array[4] = {CRGB::Green, CRGB::Red,  CRGB::Yellow, CRGB::Blue}; // Store available colors for each LED

void setup() {
  Serial.begin(9600);
  Serial.println("Initialize System");


  //Initialize FastLED
  FastLED.addLeds<LED_TYPE, digitalPin, COLOR_ORDER>(leds, LED_COUNT);
  FastLED.setBrightness(  BRIGHTNESS );
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

  // Play the sequence of LEDs
  playSequence();
}

// Function to turn off every LED
void turnOffEverything(){
  Serial.println("Turning off everything");
  for (int i = 0; i < LED_COUNT; i++) {
     leds[i].setRGB(0, 0, 0); // Set LED color to off
  }
  
  FastLED.show();// Update the LED strip

  Serial.println("Everything turned off");
}

// Function to start the game
void startGame() {
  Serial.println("Starting game");

  // Starts the game with a sequence of one step and plays it
  sequenceLength = 0;
  addStepToSequence();
  playSequence();

}

// Function to add a new step to the sequence
void addStepToSequence() {
  sequence[sequenceLength] = random(0, LED_COUNT); // Generate random step (LED index)
  sequenceLength++;
}

// Function to play the sequence of steps
void playSequence() {
  Serial.print("Playing sequence ");
  Serial.println(sequenceLength);
  for (int i = 0; i < sequenceLength; i++) {
    flashLED(sequence[i]);
    delay(TURN_DELAY);
  }
}

//Function to light up a LED
void lightLED(int ledIndex){
  leds[ledIndex] = colors_array[ledIndex];
  FastLED.show();// Update the LED strip
}

// Function to flash a single LED
void flashLED(int ledIndex) {
  
  Serial.print("Flashing Led ");
  Serial.println(ledIndex);

  lightLED(ledIndex);

  
  delay(FLASH_DELAY);
  
  leds[ledIndex] = CRGB::Black;
  FastLED.show();// Update the LED strip


  Serial.print("Led ");
  Serial.print(ledIndex);
  Serial.println(" Flashed");
}