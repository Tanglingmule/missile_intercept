import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
import numpy as np
import sys
import time
import math
import colorsys
from simulation import MissileInterceptorSimulation, Vector3, Interceptor
import random

# Initialize pygame font
pygame.font.init()

def init_gl(width: int, height: int):
    """Initialize OpenGL settings"""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Enable depth testing and lighting
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    
    # Set material properties
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_COLOR_MATERIAL)
    
    # Enable blending for transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Set background color (dark blue)
    glClearColor(0.1, 0.1, 0.2, 1.0)

def draw_grid(size=100, step=10):
    """Draw a grid on the XZ plane"""
    glBegin(GL_LINES)
    glColor3f(0.3, 0.3, 0.3)
    for i in range(-size, size + 1, step):
        # X-axis lines
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
        # Z-axis lines
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
    glEnd()

def draw_missile(position: Vector3, color=None, is_target=False):
    """Draw a missile at the given position"""
    glPushMatrix()
    glTranslatef(position.x, position.y, position.z)
    
    quadric = gluNewQuadric()
    
    if is_target:
        # Draw target missile (red)
        glColor3f(1.0, 0.0, 0.0)  # Red
        gluSphere(quadric, 2.0, 10, 10)
    else:
        # Draw interceptor with custom color or default blue
        if color is None:
            glColor3f(0.0, 0.5, 1.0)  # Default light blue
        else:
            glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0)  # Use provided color
        gluSphere(quadric, 1.0, 10, 10)
    
    gluDeleteQuadric(quadric)
    glPopMatrix()

def draw_cram(position: Vector3, rotation: Vector3):
    """Draw the C-RAM system at the given position with rotation"""
    glPushMatrix()
    glTranslatef(position.x, position.y, position.z)
    
    # Convert rotation from radians to degrees and invert Y rotation for correct orientation
    yaw_deg = math.degrees(rotation.y)
    pitch_deg = math.degrees(rotation.x)
    
    # Rotate to face the target
    glRotatef(yaw_deg, 0, 1, 0)  # Yaw (left/right)
    glRotatef(pitch_deg, 1, 0, 0)  # Pitch (up/down)
    
    # Create a quadric object for drawing
    quadric = gluNewQuadric()
    
    # Base (dark gray)
    glColor3f(0.3, 0.3, 0.3)
    glPushMatrix()
    glScalef(3, 0.5, 3)
    # Draw a cube using quads
    glBegin(GL_QUADS)
    # Front face
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    # Back face
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    # Top face
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)
    # Bottom face
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    # Right face
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    # Left face
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glEnd()
    glPopMatrix()
    
    # Turret (medium gray)
    glColor3f(0.5, 0.5, 0.5)
    glPushMatrix()
    glTranslatef(0, 0.5, 0)
    gluSphere(quadric, 0.8, 16, 16)
    
    # Barrel (dark gray)
    glColor3f(0.2, 0.2, 0.2)
    glRotatef(90, 0, 1, 0)  # Rotate to point along X axis
    gluCylinder(quadric, 0.3, 0.3, 3, 8, 1)  # Draw barrel as a cylinder
    
    glPopMatrix()
    
    # Clean up quadric
    gluDeleteQuadric(quadric)
    glPopMatrix()

def draw_trajectory(points, color=(1.0, 1.0, 0.0, 0.5)):
    """Draw a trajectory line from a list of points"""
    if len(points) < 2:
        return
    
    glDisable(GL_LIGHTING)
    glBegin(GL_LINE_STRIP)
    glColor4f(*color)
    for point in points:
        glVertex3f(point.x, point.y, point.z)
    glEnd()
    glEnable(GL_LIGHTING)

def render_text(x, y, text, font_size=20, color=(255, 255, 255, 255)):
    """Render text to the screen using pygame's font rendering"""
    font = pygame.font.SysFont('Arial', font_size)
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    width, height = pygame.display.get_surface().get_size()
    glOrtho(0, width, height, 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glRasterPos2i(x, y)
    glDrawPixels(
        text_surface.get_width(), 
        text_surface.get_height(),
        GL_RGBA, 
        GL_UNSIGNED_BYTE, 
        text_data
    )
    
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def main():
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (1280, 720)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Missile Interception Simulation - 15 Targets")
    
    # Initialize OpenGL
    init_gl(display[0], display[1])
    
    # Create simulation with 15 targets
    sim = MissileInterceptorSimulation(num_targets=15)
    
    # Camera settings
    camera_distance = 300.0
    camera_angle_x = 30.0
    camera_angle_y = 0.0
    
    # Trajectory history
    target_trajectories = [[] for _ in range(sim.num_targets)]
    interceptor_trajectory = []
    max_trajectory_points = 100
    
    # Main game loop
    running = True
    last_time = time.time()
    last_mouse_pos = (0, 0)
    
    # Trajectory tracking
    target_trajectory = []
    interceptor_trajectory = [[] for _ in range(25)]  # One trajectory per interceptor
    max_trajectory_points = 500  # Keep trajectories shorter for performance
    
    # Camera state
    smooth_camera_target = Vector3(0, 0, 0)
    is_dragging = False
    last_mouse_pos = (0, 0)
    
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reset simulation
                    sim = MissileInterceptorSimulation(num_targets=15)
                    target_trajectories = [[] for _ in range(sim.num_targets)]
                    interceptor_trajectory = [[] for _ in range(25)]
                elif event.key == pygame.K_SPACE:  # Pause/Resume
                    sim.time_scale = 0.0 if sim.time_scale > 0 else 1.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    is_dragging = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    is_dragging = False
            elif event.type == pygame.MOUSEMOTION and is_dragging:
                # Rotate camera based on mouse movement
                x, y = event.pos
                dx, dy = x - last_mouse_pos[0], y - last_mouse_pos[1]
                last_mouse_pos = (x, y)
                
                # Adjust camera angles based on mouse movement
                camera_angle_x -= dx * 0.5
                camera_angle_y = max(10.0, min(80.0, camera_angle_y - dy * 0.5))
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out with mouse wheel
                camera_distance = max(50.0, min(500.0, camera_distance - event.y * 10.0))
        
        # Get time delta
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Update simulation
        sim.update(dt)
        
        # Update target trajectories
        for i, target in enumerate(sim.targets):
            if not target.destroyed:
                if i >= len(target_trajectories):
                    target_trajectories.append([])
                target_trajectories[i].append(Vector3(target.position.x, target.position.y, target.position.z))
                if len(target_trajectories[i]) > max_trajectory_points:
                    target_trajectories[i].pop(0)
        
        # Update interceptor trajectories
        for i, interceptor in enumerate(sim.interceptors):
            if not interceptor.destroyed:
                if i >= len(interceptor_trajectory):
                    interceptor_trajectory.append([])
                interceptor_trajectory[i].append(Vector3(interceptor.position.x, interceptor.position.y, interceptor.position.z))
                if len(interceptor_trajectory[i]) > max_trajectory_points:
                    interceptor_trajectory[i].pop(0)
        
        # Camera remains static by default
        # Only update camera target if user is dragging
        if not is_dragging:
            camera_target = Vector3(0, 0, 0)  # Default view center
            camera_distance = 300.0  # Fixed distance
        
        # Reset smooth camera target when not dragging
        if not is_dragging:
            smooth_camera_target = Vector3(camera_target.x, camera_target.y, camera_target.z)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera
        glLoadIdentity()
        
        # Convert spherical coordinates to cartesian
        rad_azimuth = math.radians(camera_angle_x)
        rad_elevation = math.radians(camera_angle_y)
        
        # Calculate camera position
        camera_x = smooth_camera_target.x + camera_distance * math.cos(rad_elevation) * math.sin(rad_azimuth)
        camera_y = smooth_camera_target.y + camera_distance * math.sin(rad_elevation)
        camera_z = smooth_camera_target.z + camera_distance * math.cos(rad_elevation) * math.cos(rad_azimuth)
        
        # Set up the view
        gluLookAt(
            camera_x, camera_y, camera_z,  # Camera position
            smooth_camera_target.x, smooth_camera_target.y, smooth_camera_target.z,  # Look at point
            0, 1, 0  # Up vector
        )
        
        # Draw scene
        draw_grid()
        
        # Draw all target missiles and their trajectories
        for i, target in enumerate(sim.targets):
            if not target.destroyed:
                # Get target color based on index
                hue = (i / len(sim.targets)) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                
                # Draw target trajectory
                if i < len(target_trajectories) and len(target_trajectories[i]) > 1:
                    draw_trajectory(target_trajectories[i], (r, g, b, 0.5))
                
                # Draw target missile with unique color
                glPushMatrix()
                glTranslatef(target.position.x, target.position.y, target.position.z)
                glColor3f(r, g, b)
                gluSphere(gluNewQuadric(), 3.0, 10, 10)  # Draw target as a sphere
                glPopMatrix()
                
                # Draw target's velocity vector
                if hasattr(sim, 'draw_velocity') and sim.draw_velocity:
                    draw_vector(target.position, target.velocity, (r, g, b, 1.0))
        
        # Draw C-RAM system
        if hasattr(sim, 'cram') and hasattr(sim.cram, 'rotation'):
            draw_cram(Vector3(0, 0, 0), sim.cram.rotation)
        
        # Draw all interceptors
        for i, interceptor in enumerate(sim.interceptors):
            if not interceptor.destroyed:
                # Draw interceptor with color based on speed
                speed = interceptor.velocity.magnitude()
                speed_ratio = min(1.0, (speed - 100) / 300)  # Normalize speed to 0-1 range (100-400 m/s)
                r = speed_ratio  # Red increases with speed
                g = 0.5 - speed_ratio * 0.5  # Green decreases
                b = 1.0 - speed_ratio  # Blue decreases
                draw_missile(interceptor.position, is_target=False, color=(r, g, b, 1.0))
                
                # Draw interceptor's velocity vector
                if hasattr(sim, 'draw_velocity') and sim.draw_velocity:
                    draw_vector(interceptor.position, interceptor.velocity, (0.5, 0.5, 1.0, 1.0))            
                # Draw trajectory
                if i < len(interceptor_trajectory):
                    draw_trajectory(interceptor_trajectory[i], (speed_ratio, 0, 1 - speed_ratio, 0.5))
        
        # Display status text
        active_targets = sum(1 for t in sim.targets if not t.destroyed)
        active_interceptors = sum(1 for i in sim.interceptors if not i.destroyed)
        total_interceptors_launched = len(sim.interceptors)
        
        # Calculate interception statistics
        total_targets = len(sim.targets)
        intercepted_targets = sum(1 for t in sim.targets if hasattr(t, 'was_intercepted') and t.was_intercepted)
        missed_targets = sum(1 for t in sim.targets if t.destroyed and not (hasattr(t, 'was_intercepted') and t.was_intercepted))
        
        # Calculate success rate based only on intercepted vs total targets
        success_rate = (intercepted_targets / total_targets * 100) if total_targets > 0 else 0.0
        
        # Calculate interceptor efficiency (interceptors per successful interception)
        interceptor_efficiency = (total_interceptors_launched / intercepted_targets) if intercepted_targets > 0 else 0.0
        
        status_text = [
            f"=== Simulation Status ===",
            f"Time: {sim.time:.1f}s | Status: {'PAUSED' if sim.time_scale == 0 else 'RUNNING'}",
            f"Targets: {intercepted_targets} intercepted, {missed_targets} missed of {total_targets}",
            f"Interceptors: {active_interceptors} active, {total_interceptors_launched} launched",
            f"Interceptor Efficiency: {interceptor_efficiency:.1f} per hit",
            f"Interception Rate: {success_rate:.1f}%",
            "",
            "=== Controls ===",
            "Left-click + Drag: Rotate view",
            "Mouse Wheel: Zoom in/out",
            "R: Reset simulation",
            "SPACE: Pause/Resume",
            "ESC: Quit"
        ]
        
        # Render each line of text
        for i, text in enumerate(status_text):
            render_text(10, 20 + i * 25, text, 20, (255, 255, 255, 255))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)  # Cap at 60 FPS
        
        # Display FPS in window title
        fps = int(clock.get_fps())
        pygame.display.set_caption(f"Missile Intercept Simulation - FPS: {fps}")

        print('Bullets shot: ', len(sim.interceptors))
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
