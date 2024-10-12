# Electric Radiator Assembly Vision System

## About
This project aims to improve the assembly process of electric radiators by using a vision system 
that accurately estimates the positions of mounting holes on radiator covers and radiators. 
The system leverages stereoscopic vision to provide precise feedback to robots, allowing for 
adjustments to the cover placement. This ensures accurate alignment despite variations that may 
occur during fabrication.

## Project Structure
```plaintext
CoverAlignCV/  
├── assets/      
│   └── 3dmodels/  
│       └── couvercle.stl  
├── src/  
│   ├── detection.py  
│   ├── render.py   
│   └── simulation.py     
├── README.md   
└── requirements.txt  
```

## Getting Started

### Prerequisites
To run this project, you'll need ```Python 3.x``` installed on your machine.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gramos93/CoverAlignCV.git
cd CoverAlignCV
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

1. Extract png file from stl file.
```bash
python src/render.py
```

2. Detect edges from 'top_view.png' image.
```bash
python src/detection.py
```

3. Simulation & Video recording from stl file.
```bash
python src/simulation.py
```