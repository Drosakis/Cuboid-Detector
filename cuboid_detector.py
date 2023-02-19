# -*- coding: utf-8 -*-
"""
Created on Thu Feb 3 22:19:09 2022

@author: Drosakis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import trimesh
import pyrender
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import polar
from collections import defaultdict

def render(model_vertices, faces, R=None, P=None):
    """Renders 3D mesh in a seperate window"""
    mesh_t = trimesh.Trimesh(vertices=model_vertices, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(mesh_t)
    scene = pyrender.Scene()
    if R is None:
        R = np.eye(4)
    scene.add(mesh, pose=R)
    pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)
    
def render_spheres(points, radius, color):
    """Renders 3D spheres in a seperate window"""
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (len(points), 1, 1))
    tfs[:,:3,3] = points
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    return m

def plot_lines(lines, image_size):
    """Displays lines on an image plane"""
    lines_image = np.zeros((image_size[1], image_size[0]))
    for x1,y1,x2,y2 in lines[:]:
        cv2.line(lines_image, (x1,y1), (x2,y2), 255, 1)
    plt.imshow(lines_image, cmap='gray')
    plt.show()
    cv2.imwrite("lines.png", lines_image)
    return lines_image

def draw(model, R, P, image_size):
    """Draws cuboid points on an image plane"""
    image_width = image_size[0]
    image_height = image_size[1]
    S = image_width / 2
    
    model = homogeneous(model)
    model_world = np.dot(model, R)
    model_picture = np.dot(model_world, P)
    
    for i in range(model_picture.shape[0]):
        model_picture[i, :] = model_picture[i, :] / model_picture[i, 3]
    model_picture = inhomogeneous(model_picture)
    
    model_picture = model_picture * S + S
    model_picture = model_picture.astype(int)
    
    image = np.zeros((image_height, image_width))
    for i in range(model_picture.shape[0]):
        cv2.circle(image, (model_picture[i, 1], model_picture[i, 2]), 5, 255, -1)
        
    plt.imshow(image, cmap='gray')
    plt.show()
    return image
    
def unique_points_min_dist(lines, min_dist):
    """Merges close line endpoints together"""
    start_points = np.array(lines[:, :2])
    end_points = np.array(lines[:, 2:])
    points = np.concatenate((start_points, end_points))
    
    D = squareform(pdist(points))
    
    total_points = points.shape[0]
    unique_points = []
    counts = []
    non_unique_indices = []
    for i in range(total_points):
        if i in non_unique_indices:
            continue
        count = 1
        for j in range(i, total_points):
            if D[i][j] < min_dist and i is not j and not j in non_unique_indices:
                non_unique_indices.append(j)
                count += 1
                points[j] = points[i]
        unique_points.append(points[i])
        counts.append(count)
    
    new_start_end = np.split(points, 2)
    lines[:, :2] = new_start_end[0]
    lines[:, 2:] = new_start_end[1]
    lines = unique_lines(lines)
    points = np.unique(points, axis=0)
    return lines, points, counts
    
def unique_lines(lines):
    """Removes duplicate lines"""
    lines = np.unique(lines, axis=0)
    total_lines = lines.shape[0]
    non_unique_indices = []
    unique_lines_list = []
    for i in range(total_lines):
        if i in non_unique_indices:
            continue
        for j in range(total_lines):
            x1a,y1a,x2a,y2a = lines[i]
            x1b,y1b,x2b,y2b = lines[j]
            if x1a == x2b and y1a == y2b and x2a == x1b and y2a == y1b and not j in non_unique_indices:
                non_unique_indices.append(j)
        unique_lines_list.append(lines[i])
    lines = np.array(unique_lines_list)
    return lines
    
def compute_lines(edges):
    """Computes lines from edges"""
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 30, minLineLength=30, maxLineGap=30)
    if lines is None:
        return None, None, None
    if lines.shape[0] == 1:
        return lines[0], np.array([lines[0][0][0:2], lines[0][0][2:4]]), np.array([1, 1])
    lines = np.squeeze(lines)
    lines, _, _ = unique_points_min_dist(lines, 10)
    lines = unique_lines(lines)
    lines, points, counts = unique_points_min_dist(lines, 10)
    return lines, points, counts
    
def get_index_of_point(points, p):
    """Finds index of point"""
    for i in range(points.shape[0]):
        if points[i][0] == p[0] and points[i][1] == p[1]:
            return i

def construct_graph_from_lines(lines, points):
    """Creates graph from lines"""
    graph = defaultdict(list)
    for i in range(points.shape[0]):
        x, y = points[i]
        for j in range(lines.shape[0]):
            x1,y1,x2,y2 = lines[j]
            index1 = get_index_of_point(points, [x1, y1])
            index2 = get_index_of_point(points, [x2, y2])
            if x1 == x and y1 == y:
                addEdge(graph, i, index2)
                addEdge(graph, index2, i)
            if x2 == x and y2 == y:
                addEdge(graph, index1, i)
                addEdge(graph, i, index1)
    return graph

def addEdge(graph,u,v):
    """Graph utility function to add an edge"""
    if not v in graph[u]:
        graph[u].append(v)

def find_path(graph, start, end, path=[]):
    """Graph utility function to find path"""
    path = path + [start]
    if start == end:
        return path
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
            
def dfs(graph, start, end):
    """Graph utility function to perform Depth First Search"""
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))
            
def find_cycles(graph):
    """Graph utility function to find cycles"""
    cycles = [[node]+path for node in graph for path in dfs(graph, node, node)]
    return cycles

def plot_surface(surface, points, image_size):
    """Plot surface (graph cycle)"""
    surface_image = np.zeros((image_size[1], image_size[0]))
    for i in range(len(surface) - 1):
        p1 = surface[i]
        p2 = surface[i+1]
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        cv2.line(surface_image, (x1,y1), (x2,y2), 255, 1)
    plt.imshow(surface_image, cmap='gray')
    plt.show()
    return surface_image

def remove_duplicate_cycles(cycles):
    """Removes dupicate cycles"""
    unique_cycles = []
    non_unique_indices = []
    for i in range(len(cycles)):
        if i in non_unique_indices:
            continue
        for j in range(len(cycles)):
            if len(cycles[i]) != len(cycles[j]):
                continue
            if (np.array(cycles[i]) == np.flip(cycles[j])).all():
                non_unique_indices.append(j)
        unique_cycles.append(cycles[i])
    return unique_cycles

def compute_surfaces(lines, points):
    """Computes cuboid surfaces from lines"""
    graph = construct_graph_from_lines(lines, points)

    cycles = find_cycles(graph)
    cycles = remove_duplicate_cycles(cycles)

    surfaces = []
    for i in range(points.shape[0]):
        surfaces.append([])
        
    for cycle in cycles:
        for i in range(points.shape[0]):
            if cycle[0] == i and len(cycle) == 5:
                surfaces[i].append(cycle)

    ref_point = -1
    for i in range(len(surfaces)):
        if len(surfaces[i]) == 3:
            ref_point = i
    
    return surfaces[ref_point], ref_point

def find_topological_point_pairs(model, points, surfaces, ref_point):
    """Computes model-picture point correspondences given the reference point"""    
    # Model point connecting all three surfaces
    model_ref_point = 1
    
    # Model points connecting two seperate surfaces
    model_left_right_connection = 0
    model_left_up_connection = 3
    model_right_up_connection = 5
    
    # Independent model points of each surface
    model_independent_left_point = 2
    model_independent_right_point = 4
    model_independent_up_point = 7
    
    # Ordering of the surfaces doesn't matter because the model is a cube
    surface_left = surfaces[0][:4]
    surface_right = surfaces[1][:4]
    surface_up = surfaces[2][:4]
    
    # Remove ref points to compute connecting points with intersections
    surface_left.remove(ref_point)
    surface_right.remove(ref_point)
    surface_up.remove(ref_point)
    
    # Compute connecting points
    left_right_connection = np.intersect1d(surface_left, surface_right)[0]
    left_up_connection = np.intersect1d(surface_left, surface_up)[0]
    right_up_connection = np.intersect1d(surface_right, surface_up)[0]
    
    # Compute independent points
    surface_left.remove(left_right_connection)
    surface_left.remove(left_up_connection)
    independent_left_point = surface_left[0]
    
    surface_right.remove(left_right_connection)
    surface_right.remove(right_up_connection)
    independent_right_point = surface_right[0]
    
    surface_up.remove(left_up_connection)
    surface_up.remove(right_up_connection)
    independent_up_point = surface_up[0]
    
    # Allocate model and picture point arrays
    model_points = np.zeros((7, 3))
    picture_points = np.zeros((7, 2))
    
    # Ref point correspondence
    model_points[0] = model[model_ref_point]
    picture_points[0] = points[ref_point]
    
    # Connection points correspondence
    model_points[1] = model[model_left_right_connection]
    picture_points[1] = points[left_right_connection]
    
    model_points[2] = model[model_left_up_connection]
    picture_points[2] = points[left_up_connection]
    
    model_points[3] = model[model_right_up_connection]
    picture_points[3] = points[right_up_connection]
    
    # Independent point correspondence
    model_points[4] = model[model_independent_left_point]
    picture_points[4] = points[independent_left_point]
    
    model_points[5] = model[model_independent_right_point]
    picture_points[5] = points[independent_right_point]
    
    model_points[6] = model[model_independent_up_point]
    picture_points[6] = points[independent_up_point]
    
    return picture_points, model_points

def normalized_image_space(points, image_size):
    """Normalizes the points to a 0-1 range"""
    height = image_size[0]
    width = image_size[1]
    S = max(width / 2, height / 2)
    points = (points - S) / S
    return points
    
def homogeneous(points):
    """Adds a column of ones to a set of vectors, converting them to homogeneous coordinates"""
    ones_col = np.ones((points.shape[0], 1))
    points = np.concatenate((points, ones_col), axis=1)
    return points

def inhomogeneous(points):
    """Normalizes a set of vectors with the homogeneous dimension and then removes it"""
    for i in range(points.shape[0]):
        points[i, :] = points[i, :] / points[i, 3]
    points = points[:, :3]
    return points
    
def orthogonal(M):
    """Enforces an orthogonality constraint to a matrix"""
    U, D, V = np.linalg.svd(M, full_matrices=True)
    M = np.dot(U, V)
    return M
    
def compute_transformation_no_perspective(model_points, picture_points):
    """Computes transformation from model picture correspondences without the perspective component"""
    indices = [1, 3, 4, 5]
    A = homogeneous(model_points[indices])
    B = homogeneous(picture_points[indices])
    A_inv = np.linalg.inv(A)
    H = np.dot(A_inv, B)
    E = np.linalg.norm(np.dot(homogeneous(model_points), H) - homogeneous(picture_points))
    print("Error: " + str(E))
    return H

def compute_transformation(model_points, picture_points):
    """Computes full transformation from model picture correspondences"""
    A = homogeneous(model_points)
    B = homogeneous(picture_points)
    I = np.eye(A.shape[0])
    G = np.dot(np.dot(A, np.linalg.inv(np.dot(A.T, A))), A.T) - I
    Q = np.dot(B, B.T)
    S = G * Q.T
    _, _, V = np.linalg.svd(S, full_matrices=True)
    d = V[-1]
    D = np.diag(d)
    H = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), D), B)
    E = np.linalg.norm(np.dot(A, H) - np.dot(D, B))
    print("Error: " + str(E))
    return H

def rotation_matrix_to_euler(R):
    """Converts a rotation matrix to euler angles"""
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    euler = np.array([x, y, z])
    return euler

def translation_matrix(t):
    """Constructs a translation matrix from a translation vector"""
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [t[0], t[1], t[2], 1]])
    return T

def rotation_matrix(euler):
    """Construct a rotation matrix from euler angles"""
    theta_x = euler[0]
    theta_y = euler[1]
    theta_z = euler[2]
    Rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x), 0],
                    [0, np.sin(theta_x), np.cos(theta_x), 0],
                    [0, 0, 0, 1]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                    [0, 0, 0, 1]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                    [np.sin(theta_z), np.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R

def scale_matrix(s):
    """Constructs a scale matrix from a scale vector"""
    S = np.array([[s[0], 0, 0, 0],
                  [0, s[1], 0, 0],
                  [0, 0, s[2], 0],
                  [0, 0, 0, 1]])
    return S

def is_orthogonal(M):
    """Tests whether a matrix is orthogonal"""
    r = np.allclose(np.dot(M, M.T), np.eye(3))
    c = np.allclose(np.dot(M.T, M), np.eye(3))
    return r or c

def has_orthogonal_rows(R):
    """Tests whether a matrix has orthogonal rows"""
    v1 = R[0, :3]
    v2 = R[1, :3]
    v3 = R[2, :3]
    d1 = np.dot(v1, v2)
    d2 = np.dot(v1, v3)
    d3 = np.dot(v2, v3)
    epsilon = 0.0000001
    return d1 < epsilon and d2 < epsilon and d3 < epsilon

def normalize(v):
    """Normalizes a vector"""
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def decompose_transformation(H):
    """Decomposes full transformation into a affine trasformation and a perspective transformation"""
    zeros_col = np.zeros((H.shape[0], 1))
    H = np.concatenate((zeros_col, H), axis=1)
    H = H / H[3, 3]
    
    R = H.copy()
    w = H[:3, 3]
    R[:3, 3] = 0
    
    # Make top three rows of R orthogonal to remove skew variations (TODO: check for better solution to remove skew variations)
    k1 = -R[0, 1] * R[1, 1] - R[0, 2] * R[1, 2]
    k2 = -R[0, 1] * R[2, 1] - R[0, 2] * R[2, 2]
    k3 = -R[1, 1] * R[2, 1] - R[1, 2] * R[2, 2]
    if k1 != 0 and k2 != 0 and k3 != 0:
        x1 = np.sign(k3) * np.sqrt(np.abs(k1)) * np.sqrt(np.abs(k2)) / np.sqrt(np.abs(k3))
        x2 = np.sign(k2) * np.sqrt(np.abs(k1)) * np.sqrt(np.abs(k3)) / np.sqrt(np.abs(k2))
        x3 = np.sign(k1) * np.sqrt(np.abs(k2)) * np.sqrt(np.abs(k3)) / np.sqrt(np.abs(k1))
        R[:3, 0] = [x1, x2, x3]
    else:
        print("Error. Skew variation removal conditions are not met.")
    
    r = - w[0] / R[0, 0]
    P = np.array([[1, 0, 0, -r],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    #print(has_orthogonal_rows(R[:3, :3]))
    return R, P

def decompose_TRS(A):
    """Decomposes an affine transformation into translation, rotation and scale vectors"""
    t = A[:3, 3]
    
    L = A.copy()
    L[:3, 3] = 0
    R, S = polar(L)
    
    if np.linalg.det(R) < 0:
        R[:3,:3] = -R[:3,:3]
        S[:3,:3] = -S[:3,:3]
        
    r = rotation_matrix_to_euler(R)
    
    s = [S[0, 0], S[1, 1], S[2, 2]]
    
    return t, r, s

class DrawLineWidget(object):
    """Widget for a user to draw lines in an image"""
    def __init__(self, image):
        self.original_image = image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
        self.lines = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            #print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            self.lines.append([self.image_coordinates[0][0], self.image_coordinates[0][1], self.image_coordinates[1][0], self.image_coordinates[1][1]])
            
            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
    
    def close(self):
        cv2.destroyAllWindows()
    
def get_lines_from_user_input(image):
    """Get line draws from a user on an image"""
    draw_line_widget = DrawLineWidget(image)
    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        if cv2.waitKey(1) == ord('q'):
            break
    draw_line_widget.close()
    lines = np.array(draw_line_widget.lines)
    return lines

def animate(model, t, r, s, wait_for_input=True, save_to_file=False):
    """Animates cuboid points in a series of images"""
    if wait_for_input:
        input()
    for i in range(65):
        r = r + [0, 0, np.pi/32]
        
        T = translation_matrix(t)
        R = rotation_matrix(r)
        S = scale_matrix(s)
        
        R = R.T
        
        A = np.dot(np.dot(S, R), T)
        image = draw(model, A, P, image_size)
        if save_to_file:
            cv2.imwrite('images/' + str(i) + '.png', image)

# Cube model
model = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])
model = model - np.array([0.5, 0.5, 0.5])

# Cube faces (required for 3D rendering)
faces = np.array([[1, 3, 0],
                  [4, 1, 0],
                  [0, 3, 2],
                  [2, 4, 0],
                  [1, 7, 3],
                  [5, 1, 4],
                  [5, 7, 1],
                  [3, 7, 2],
                  [6, 4, 2],
                  [2, 7, 6],
                  [6, 5, 4],
                  [7, 5, 6]])

# Render cube model 3D mesh
#render(model, faces)

# Read image
image_name = "cuboid.png"
image = cv2.imread(image_name)
image_size = (image.shape[1], image.shape[0])
I_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(I_rgb)
plt.show()
plt.imshow(I, cmap='gray')
plt.show()
cv2.imwrite("grayscale.png", I)

# Compute edges from image
edges = cv2.Canny(image=I, threshold1=100, threshold2=200)
plt.imshow(edges, cmap='gray')
plt.show()
cv2.imwrite("edges.png", edges)

# Compute lines from edge pixels
lines, points, counts = compute_lines(edges)
if lines is None:
    print('No lines detected.')
    raise SystemExit
print("Lines detected: " + str(lines.shape[0]))
lines_image = plot_lines(lines, image_size)
cv2.imwrite("lines.png", lines_image)

# Match line end points to model points
surfaces, ref_point = compute_surfaces(lines, points)
if ref_point == -1:
    print('Cuboid not found.')
    raise SystemExit
print("Reference point: " + str(points[ref_point]) + "  (index: " + str(ref_point) + ")")
for i, surface in enumerate(surfaces):
    surface_image = plot_surface(surface, points, image_size)
    cv2.imwrite("surface_" + str(i) + ".png", surface_image)
picture_points, model_points = find_topological_point_pairs(model, points, surfaces, ref_point)
picture_points_normalized = normalized_image_space(picture_points, image_size)

# Compute transformation from model points to picture points
H = compute_transformation(model_points[:6], picture_points_normalized[:6])
R, P = decompose_transformation(H)

# Display computed cuboid points
points = draw(model, R, P, image_size)
cv2.imwrite("points.png", points)

# Decompose R into translation, rotation and scale and animate cuboid points
#t, r, s = decompose_TRS(R.T)
#animate(model, t, r, s)

# Render cuboid 3D mesh
render(model, faces, R.T, P)