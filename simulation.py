import numpy as np
from typing import Tuple, List, Optional
import math
import random
import time

class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        """Return the magnitude (length) of the vector"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag == 0:
            return Vector3()
        return self * (1.0 / mag)
    
    def dot(self, other: 'Vector3') -> float:
        """Return the dot product of this vector with another vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z
        
    def cross(self, other: 'Vector3') -> 'Vector3':
        """Return the cross product of this vector with another vector"""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
        
    def rotate_x(self, angle: float) -> 'Vector3':
        """Rotate the vector around the X axis by the given angle in radians"""
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        y = self.y * cos_theta - self.z * sin_theta
        z = self.y * sin_theta + self.z * cos_theta
        return Vector3(self.x, y, z)
        
    def rotate_y(self, angle: float) -> 'Vector3':
        """Rotate the vector around the Y axis by the given angle in radians"""
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        x = self.x * cos_theta + self.z * sin_theta
        z = -self.x * sin_theta + self.z * cos_theta
        return Vector3(x, self.y, z)
        
    def rotate_z(self, angle: float) -> 'Vector3':
        """Rotate the vector around the Z axis by the given angle in radians"""
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        x = self.x * cos_theta - self.y * sin_theta
        y = self.x * sin_theta + self.y * cos_theta
        return Vector3(x, y, self.z)
        
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

class Missile:
    def __init__(self, position: Vector3, velocity: Vector3, is_target: bool = False):
        self.position = position
        self.velocity = velocity
        self.acceleration = Vector3()
        self.is_target = is_target
        self.destroyed = False
        self.radius = 1.0  # For collision detection
        
    def update(self, dt: float, gravity: Vector3, wind_resistance: float = 0.01):
        if self.destroyed:
            return
            
        # Apply gravity
        self.acceleration = gravity
        
        # Apply wind resistance (proportional to velocity squared, opposite direction)
        if self.velocity.magnitude() > 0:
            drag = self.velocity.normalized() * (self.velocity.magnitude() ** 2) * -wind_resistance
            self.acceleration += drag
        
        # Update velocity and position using semi-implicit Euler integration
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Ground collision (simplified)
        if self.position.y < 0:
            self.position.y = 0
            self.destroyed = True

class Interceptor(Missile):
    def __init__(self, position: Vector3, velocity: Vector3, target_position: Vector3 = None, 
                 target_velocity: Vector3 = None, accuracy: float = 1.0, speed_factor: float = 1.0):
        super().__init__(position, velocity, is_target=False)
        self.target_position = target_position or Vector3(0, 0, 0)
        self.target_velocity = target_velocity or Vector3(0, 0, 0)
        self.accuracy = accuracy  # 0.0 to 1.0, affects guidance
        self.speed_factor = speed_factor  # Speed multiplier
        self.time_since_launch = 0.0
        self.destroyed = False
        self.collision_radius = 2.5  # meters
        self.target_id = -1  # ID of the target this interceptor is tracking
        
    def update(self, dt: float, gravity: 'Vector3', wind_resistance: float):
        super().update(dt, gravity, wind_resistance)
        self.time_since_launch += dt

class CRAMSystem:
    def __init__(self, position: Vector3):
        self.position = position
        self.rotation = Vector3(0, 0, 0)  # pitch, yaw, roll
        self.shot_interval = 0.2  # seconds between shots (faster for multiple targets)
        self.last_shot_time = -10  # initialize to allow immediate first shot
        self.max_interceptors = 100  # Increased for multiple targets
        self.active_interceptors = 0
        self.engagement_range = 1000  # Increased engagement range
        self.target_history = {}  # Track targets being engaged
        
    def can_shoot(self, current_time: float) -> bool:
        """Check if the system can fire a new interceptor"""
        return (current_time - self.last_shot_time >= self.shot_interval and 
                self.active_interceptors < self.max_interceptors)
                
    def aim_at_target(self, target_position: Vector3):
        """Aim the CRAM system at the target"""
        # Calculate direction to target (pitch and yaw only)
        direction = target_position - self.position
        self.rotation.y = math.degrees(math.atan2(direction.x, direction.z))  # Yaw
        
        # Calculate pitch (limit to reasonable elevation)
        horizontal_distance = math.sqrt(direction.x**2 + direction.z**2)
        self.rotation.x = -math.degrees(math.atan2(direction.y, horizontal_distance))  # Pitch
        
    def get_best_target(self, targets: List['Missile']) -> Optional[Tuple[Vector3, Vector3, int]]:
        """Select the best target to engage based on proximity and threat level"""
        if not targets:
            return None
            
        best_target = None
        best_score = -float('inf')
        current_time = time.time()
        
        for i, target in enumerate(targets):
            if target.destroyed:
                continue
                
            # Calculate target metrics
            distance = (target.position - self.position).magnitude()
            time_since_engaged = current_time - self.target_history.get(i, 0)
            
            # Calculate threat score (higher is more threatening)
            # Prioritize closer targets and those not recently engaged
            score = (1000 / max(1, distance)) * (1 + time_since_engaged)
            
            # Bonus for targets heading towards the C-RAM
            direction_to_cram = (self.position - target.position).normalized()
            approach_angle = math.degrees(math.acos(
                direction_to_cram.dot(target.velocity.normalized()) / 
                (direction_to_cram.magnitude() * target.velocity.magnitude())
            ))
            
            if approach_angle < 45:  # Target is heading towards C-RAM
                score *= 1.5
                
            if score > best_score:
                best_score = score
                best_target = (target.position, target.velocity, i)
        
        return best_target
        
    def launch_interceptor(self, target_position: Vector3, target_velocity: Vector3, 
                          target_id: int, current_time: float) -> Optional['Interceptor']:
        """Launch an interceptor missile at the specified target"""
        if not self.can_shoot(current_time):
            return None
            
        # Calculate intercept point with lead
        to_target = target_position - self.position
        distance = to_target.magnitude()
        time_to_intercept = distance / 350  # Approximate time to target
        
        # Predict target position at intercept time
        predicted_pos = target_position + target_velocity * time_to_intercept
        
        # Calculate initial direction with lead
        direction = (predicted_pos - self.position).normalized()
        
        # Add some randomness to initial velocity (reduced spread for better accuracy)
        spread = 0.05  # radians (reduced from 0.1)
        spread_x = random.uniform(-spread, spread)
        spread_y = random.uniform(-spread, spread)
        
        # Apply rotation to direction
        if hasattr(direction, 'rotate_x') and hasattr(direction, 'rotate_y'):
            direction = direction.rotate_x(spread_x).rotate_y(spread_y).normalized()
        
        # Vary speed slightly for more realistic behavior
        speed = random.uniform(300, 400)  # m/s
        
        # Create interceptor with high accuracy (95-100%)
        accuracy = 0.95 + random.random() * 0.05
        speed_factor = 0.9 + random.random() * 0.2  # 0.9-1.1x speed
        
        interceptor = Interceptor(
            position=Vector3(self.position.x, self.position.y + 2, self.position.z),
            velocity=direction * speed,
            target_position=predicted_pos,
            target_velocity=target_velocity,
            accuracy=accuracy,
            speed_factor=speed_factor
        )
        interceptor.target_id = target_id  # Track which target this interceptor is after
        
        # Update state
        self.last_shot_time = current_time
        self.active_interceptors += 1
        self.target_history[target_id] = current_time
        
        # Clean up old target history
        self.target_history = {
            tid: t for tid, t in self.target_history.items() 
            if current_time - t < 30  # Keep targets for 30 seconds
        }
        
        return interceptor

class MissileInterceptorSimulation:
    def __init__(self, num_targets=15):
        # Simulation parameters
        self.gravity = Vector3(0, -9.81, 0)  # m/s²
        self.wind_resistance = 0.0
        self.time_scale = 1.0
        
        # Create multiple target missiles
        self.targets = []
        self.num_targets = num_targets
        self.create_targets()
        
        # Create C-RAM system
        self.cram = CRAMSystem(position=Vector3(0, 0, 0))
        
        # Create interceptors list (now managed by C-RAM)
        self.interceptors = []
        
        # Simulation state
        self.time = 0.0
        self.running = True
        self.interception_count = 0
        self.last_shot_time = 0
        
    def create_targets(self):
        """Create multiple target missiles with random positions and velocities"""
        for i in range(self.num_targets):
            # Random starting position (in a wide arc)
            angle = (i / self.num_targets) * math.pi * 1.5 + math.pi/4  # 90 degree arc
            distance = 200 + random.uniform(-50, 50)
            height = 50 + random.uniform(0, 100)
            
            x = math.cos(angle) * distance
            z = math.sin(angle) * distance * 0.5  # Flatten the arc
            
            # Velocity towards origin with some randomness
            speed = 30 + random.uniform(-10, 10)
            vel_x = -x / distance * speed
            vel_y = -height / distance * speed * 0.5
            vel_z = -z / distance * speed
            
            initial_pos = Vector3(x, height, z)
            initial_vel = Vector3(vel_x, vel_y, vel_z)
            
            self.targets.append(Missile(initial_pos, initial_vel, is_target=True))
    
    def create_interceptors(self):
        import random
        num_interceptors = 50
        
        for i in range(num_interceptors):
            # Start interceptors in a more strategic spread pattern
            x_offset = random.uniform(-120, -30)  # Closer to target
            y_offset = random.uniform(20, 80)     # Better vertical coverage
            
            # Higher base accuracy (0.85 to 1.0)
            accuracy = 0.85 + random.random() * 0.15
            
            # Higher base speed (1.2x to 2.0x)
            speed_factor = 1.2 + random.random() * 0.8
            
            # Create interceptor
            pos = Vector3(x_offset, y_offset, 0)
            vel = Vector3(100 * speed_factor, 0, 0)  # Initial velocity
            
            interceptor = Interceptor(pos, vel, accuracy, speed_factor)
            self.interceptors.append(interceptor)
        
    def update(self, dt: float):
        """Update the simulation"""
        if not self.running:
            return
            
        # Scale dt by time_scale to allow for slow-mo or fast-forward
        scaled_dt = dt * self.time_scale
        self.time += scaled_dt
        
        # Update all targets
        active_targets = [t for t in self.targets if not t.destroyed]
        for target in active_targets:
            target.update(scaled_dt, self.gravity, self.wind_resistance)
        
        # Let C-RAM select the best target to engage
        if active_targets and hasattr(self, 'cram'):
            # Get best target and its index
            target_info = self.cram.get_best_target(active_targets)
            
            if target_info:
                target_pos, target_vel, target_idx = target_info
                
                # Aim C-RAM at selected target
                self.cram.aim_at_target(target_pos)
                
                # Check if we should launch a new interceptor
                if self.cram.can_shoot(self.time):
                    interceptor = self.cram.launch_interceptor(
                        target_pos,
                        target_vel,
                        target_idx,  # Pass target index for tracking
                        self.time
                    )
                    if interceptor:
                        interceptor.target_id = target_idx
                        self.interceptors.append(interceptor)
        
        # Update all interceptors
        active_interceptors = 0
        for interceptor in self.interceptors:
            if not interceptor.destroyed:
                active_interceptors += 1
                
                # Find this interceptor's target
                target = None
                if hasattr(interceptor, 'target_id') and interceptor.target_id < len(self.targets):
                    target = self.targets[interceptor.target_id]
                
                # Apply guidance if target exists and is active
                if target and not target.destroyed:
                    self.guide_interceptor(interceptor, target, scaled_dt)
                
                # Update physics
                interceptor.update(scaled_dt, Vector3(0, 0, 0), 0.0)
                
                # Apply gravity
                interceptor.velocity.y += self.gravity.y * scaled_dt
                
                # Check for collisions with all active targets
                for target in active_targets:
                    if not target.destroyed and self.check_collision(interceptor, target):
                        self.on_collision(interceptor, target)
                        break  # Only one target can be hit per frame
        
        # Update C-RAM active interceptors count
        self.cram.active_interceptors = active_interceptors
    
    def guide_interceptor(self, interceptor: 'Interceptor', target: 'Missile', dt: float):
        """Enhanced proportional navigation with better target prediction"""
        if target.destroyed or interceptor.destroyed:
            return
            
        # Get current positions and velocities
        r = target.position - interceptor.position  # Relative position
        v = interceptor.velocity  # Interceptor velocity
        v_t = target.velocity  # Target velocity
        
        # Calculate relative velocity
        v_r = v_t - v
        
        # Calculate line of sight rate (LOS rate) - magnitude of the cross product gives the angular rate
        los = r.normalized()
        los_rate = v_r.cross(los).magnitude() / max(1.0, r.magnitude())
        
        # Calculate time to closest approach
        los_normalized = los
        closing_speed = -(los_normalized.x * v_r.x + los_normalized.y * v_r.y + los_normalized.z * v_r.z)
        time_to_go = r.magnitude() / max(1.0, abs(closing_speed))
        
        # Add some inaccuracy based on the interceptor's accuracy
        accuracy_noise = (1.0 - interceptor.accuracy) * 0.3  # Reduced noise
        noise = Vector3(
            (random.random() * 2 - 1) * accuracy_noise * 50,  # Reduced noise magnitude
            (random.random() * 2 - 1) * accuracy_noise * 50,
            0
        )
        
        # Predict future position with lead and noise
        future_pos = target.position + v_t * (time_to_go * 0.8) + noise  # 80% lead for more aggressive intercept
        
        # Calculate desired direction to future position
        to_future = future_pos - interceptor.position
        distance = to_future.magnitude()
        
        if distance < 5.0:  # Increased close range threshold
            return
            
        # Calculate desired speed based on distance to target
        base_speed = 250 * interceptor.speed_factor  # Higher base speed
        desired_speed = base_speed * min(1.5, 1.0 + distance / 200.0)  # Faster when further
        
        # Calculate desired velocity vector with lead
        desired_velocity = to_future.normalized() * desired_speed
        
        # Enhanced proportional navigation with LOS rate feedback
        nav_constant = 8.0  # More aggressive steering
        accel = (desired_velocity - v) * nav_constant
        
        # Add gravity compensation with some lead
        accel = accel - (self.gravity * 1.2)
        
        # Add terminal homing when close
        if distance < 100:  # Within 100 meters
            # Increase navigation constant for terminal phase
            accel = accel * 1.5
            # Add direct line-of-sight correction
            accel = accel + los * 50
        
        # Apply acceleration (with higher max limit)
        max_accel = 500.0 * interceptor.speed_factor  # Higher max acceleration
        if accel.magnitude() > max_accel:
            accel = accel.normalized() * max_accel
        
        # Enhanced vertical guidance
        height_diff = target.position.y - interceptor.position.y
        if abs(height_diff) > 2:  # If there's significant height difference
            # Scale vertical correction with distance
            vertical_correction = height_diff * (1.0 + distance / 200.0)
            accel.y += vertical_correction * 2.0
        
        # Prevent ground collision with stronger correction
        if interceptor.position.y < 15 and interceptor.velocity.y < 0:
            accel.y = max(accel.y, 30.0)  # Stronger upward correction
        
        interceptor.acceleration = accel
    
    def check_collision(self, interceptor: 'Interceptor', target: 'Missile') -> bool:
        """Check if interceptor has collided with a target"""
        if interceptor.destroyed or target.destroyed:
            return False
            
        # Simple distance-based collision detection
        distance = (interceptor.position - target.position).magnitude()
        return distance < (interceptor.radius + target.radius)
    
    def on_collision(self, interceptor: 'Interceptor', target: 'Missile'):
        self.interception_count += 1
        print(f"Interceptor {self.interception_count} hit at time {self.time:.2f}s!")
        target.destroyed = True
        interceptor.destroyed = True
        
        # End simulation if all interceptors are done
        if all(i.destroyed for i in self.interceptors):
            self.running = False
