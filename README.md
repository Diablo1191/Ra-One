# Ra-One
Jee-Van 
This project uses a custom YOLOv10 model for crop counting, weed detection, wheet deyection and yield prediction from drone footage. The application provides a graphical user interface (GUI) to start crop analysis, toggle the display of yield prediction on and off, and manage the app's lifecycle.

Features
Crop Counting: Detects and counts crops in real-time using drone footage.
Yield Prediction: Estimates crop yield based on green cover percentage in the footage.
Toggle Yield Display: Allows users to toggle the yield prediction on and off during analysis.
User-Friendly Interface: GUI for easy control of the application.
Prerequisites
Before running the application, ensure you have the following installed:

Python 3.8 or later
Virtual Environment (recommended)
Required Python libraries (see below)
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/Diablo1191/Ra-One.git

Create and activate a virtual environment (recommended):

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
Download the YOLOv10 model:

Place your custom YOLOv10 model file (best.pt) in the project directory.
File Structure
maincode.py: The core logic for crop detection, counting, and yield prediction.
tracker.py: A custom object tracker that assigns unique IDs to crops for accurate counting.
appui.py: The graphical user interface for starting the analysis, toggling yield display, and exiting the application.
config.py: A configuration file to manage global variables such as the yield display toggle.
Running the Application
Start the application:

bash
Copy code
python appui.py
Using the GUI:

Welcome Screen: Choose to start the analysis or exit the app.
Toggle Yield Display: Turn the yield prediction display on or off using the "Toggle Yield Display" button.
Real-Time Crop Counting: After starting the analysis, you will see the crop count, and if enabled, the predicted yield on the screen.
Configuration
Yield Display: The display of yield prediction can be toggled during the analysis. The config.py file manages the state of the yield display.
Future Improvements
Add support for additional crop types or conditions.
Improve the yield prediction model for more accuracy.
Implement a "Stop Analysis" button to terminate the analysis without closing the application.
Contributing
If you would like to contribute to the project, feel free to fork the repository and submit a pull request. You can also open an issue for any bug reports or feature requests.
