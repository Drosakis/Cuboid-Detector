# Cuboid Detector
3D Reconstruction of a cuboid from an RGB image.
### How to run
1. Install requirements with: <br>
    <code>pip install -r requirements.txt</code>
2. Run the script with: <br>
    <code>python cuboid_detector.py</code> <br>

### Pipeline
Cuboid RGB image: <br>
<p align="center">
  <img src="cuboid.png"">
</p>
Cuboid grayscale image: <br>
<p align="center">
  <img src="grayscale.png"">
</p>
Edges: <br>
<p align="center">
  <img src="edges.png"">
</p>
Lines computed from edges: <br>
<p align="center">
  <img src="lines.png"">
</p>
Cuboid surfaces from lines: <br>
<p align="center">
  <img src="surface_2.png"">
  <img src="surface_1.png"">
  <img src="surface_0.png"">
</p>
Final computed cuboid points: <br>
<p align="center">
  <img src="points.png"">
</p>
Reconstructed cuboid: <br>
<p align="center">
  <img src="render.png"">
</p>
