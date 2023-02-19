# Cuboid Detector
3D Reconstruction of a cuboid from an RGB image.
### How to run
1. Install requirements with: <br>
    <code>pip install -r requirements.txt</code>
2. Run the script with: <br>
    <code>python cuboid_detector.py</code> <br>

### Pipeline
Cuboid RGB image: <br>
![Alt cuboid](cuboid.png) <br>
Cuboid grayscale image: <br>
![Alt grayscale](grayscale.png) <br>
Edges: <br>
![Alt edges](edges.png) <br>
Lines computed from edges: <br>
![Alt lines](lines.png) <br>
Cuboid surfaces from lines: <br>
![Alt surface_0](surface_0.png) <br>
![Alt surface_1](surface_1.png) <br>
![Alt surface_2](surface_2.png) <br>
Final computed cuboid points: <br>
![Alt points](points.png) <br>
Reconstructed cuboid: <br>
![Alt render](render.png) <br>
