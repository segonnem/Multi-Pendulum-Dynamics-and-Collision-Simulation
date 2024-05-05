import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
import scipy.linalg
import matplotlib.patches as patches
from matplotlib.patches import Circle
import sys

# Increase the recursion limit
sys.setrecursionlimit(3000) 

class Boule:
    def __init__(self, x, y, radius, vx, vy, ax, ay, color, masse):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.color = color
        self.masse = masse

    def update_position(self, dt):
        # Update the velocity with acceleration
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Update the position with the velocity
        self.x += self.vx * dt
        self.y += self.vy * dt

    def update_velocity_after_collision(self, other):
        # Calculate the difference in positions
        dx = self.x - other.x
        dy = self.y - other.y
        dist_squared = dx**2 + dy**2

        # Calculate the difference in velocities
        dvx = self.vx - other.vx
        dvy = self.vy - other.vy

        dot_product = dvx * dx + dvy * dy

        if dot_product < 0:  # Les boules se rapprochent, on évite un flip et donc des boules qui s'accrochent
            
            dist_squared = dx**2 + dy**2

            self.vx -= (2 * other.masse * (dvx*dx + dvy*dy) / (self.masse + other.masse)) / dist_squared * dx
            self.vy -= (2 * other.masse * (dvx*dx + dvy*dy) / (self.masse + other.masse)) / dist_squared * dy

            other.vx -= (2 * self.masse * (dvx*dx + dvy*dy) / (self.masse + other.masse)) / dist_squared * (-dx)
            other.vy -= (2 * self.masse * (dvx*dx + dvy*dy) / (self.masse + other.masse)) / dist_squared * (-dy)

    def collides_with(self, other):
        # Calculer la distance entre les centres des deux boules
        dx = self.x - other.x
        dy = self.y - other.y
        distance = (dx**2 + dy**2)**0.5

        # Vérifier si la distance est inférieure à la somme des rayons
        return distance <= (self.radius + other.radius)
            

class PendulumSystem:
    def __init__(self, theta, omega, lengths, masses, xstart, ystart, radius, colors):
        self.theta = np.array(theta)
        self.omega = np.array(omega)
        self.lengths = np.array(lengths)
        self.masses = np.array(masses)
        self.xstart = xstart
        self.ystart = ystart
        self.radius = radius
        self.colors = colors

    def sigma(self, j, k):
        return 0 if j > k else 1

    def phi(self, j, k):
        return 0 if j == k else 1

    def compute_b(self):
        n = len(self.theta)
        b = np.zeros(n)
        g = 9.81  # Constante gravitationnelle

        for j in range(n):
            term1 = term2 = term3 = 0.0

            for k in range(n):
                sum_for_k = sum(self.masses[q] * self.sigma(j, q) for q in range(k, n))

                term1 += g * self.lengths[j] * np.sin(self.theta[j]) * self.masses[k] * self.sigma(j, k)
                term2 += sum_for_k * self.lengths[j] * self.lengths[k] * np.sin(self.theta[j] - self.theta[k]) * self.omega[j] * self.omega[k]
                term3 += sum_for_k * self.lengths[j] * self.lengths[k] * np.sin(self.theta[k] - self.theta[j]) * (self.omega[j] - self.omega[k]) * self.omega[k]

            b[j] = -(term1 + term2 + term3)

        return b

    def compute_A(self):
        n = len(self.theta)
        A = np.zeros((n, n))

        for j in range(n):
            for k in range(n):

                sum_for_q = sum(self.masses[q] * self.sigma(j, q) for q in range(k, n))
                value = sum_for_q * self.lengths[j] * self.lengths[k] * self.phi(j, k) * np.cos(self.theta[j] - self.theta[k])

                if j == k:
                    sum_for_k = sum(self.masses[kk] * self.sigma(j, kk) for kk in range(n))
                    value += sum_for_k * self.lengths[j] ** 2

                A[j, k] = value

        return A
    
    def rk4_step(self, dt):
        # K1
        k1 = PendulumSystem(dt * self.omega, [], self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)
        b1 = self.compute_b()
        A1 = self.compute_A()
        k1.omega = dt * scipy.linalg.solve(A1, b1)

        # État intermédiaire 1
        mid1 = PendulumSystem(self.theta + 0.5 * k1.theta, self.omega + 0.5 * k1.omega, self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)

        # K2
        k2 = PendulumSystem(dt * mid1.omega, [], self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)
        b2 = mid1.compute_b()
        A2 = mid1.compute_A()
        k2.omega = dt * scipy.linalg.solve(A2, b2)

        # État intermédiaire 2
        mid2 = PendulumSystem(self.theta + 0.5 * k2.theta, self.omega + 0.5 * k2.omega, self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)

        # K3
        k3 = PendulumSystem(dt * mid2.omega, [], self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)
        b3 = mid2.compute_b()
        A3 = mid2.compute_A()
        k3.omega = dt * scipy.linalg.solve(A3, b3)

        # État intermédiaire 3
        end = PendulumSystem(self.theta + k3.theta, self.omega + k3.omega, self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)

        # K4
        k4 = PendulumSystem(dt * end.omega, [], self.lengths, self.masses, self.xstart, self.ystart, self.radius, self.colors)
        b4 = end.compute_b()
        A4 = end.compute_A()
        k4.omega = dt * scipy.linalg.solve(A4, b4)

        # Nouvel état
        new_theta = self.theta + (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta) / 6
        new_omega = self.omega + (k1.omega + 2 * k2.omega + 2 * k3.omega + k4.omega) / 6

        return new_theta, new_omega
    

    def update(self, dt):
        # Mise à jour de l'état du pendule en utilisant la méthode rk4_step
        new_theta, new_omega = self.rk4_step(dt)

        # Mettre à jour l'état actuel avec les nouvelles valeurs
        self.theta, self.omega = new_theta, new_omega

        return self.theta, self.omega

    def get_mass_positions(self):
        # Calcule la position de chaque masse du pendule
        positions = []
        x = self.xstart
        y = self.ystart
        for length, theta in zip(self.lengths, self.theta):
            x += length * np.sin(theta)
            y -= length * np.cos(theta)
            positions.append((x, y))
        return positions
    
    def get_tige_positions(self):
        # Calcule la position de chaque tige du pendule
        tige_positions = []
        x_prev, y_prev = self.xstart, self.ystart
        for length, theta in zip(self.lengths, self.theta):
            x_next = x_prev + length * np.sin(theta)
            y_next = y_prev - length * np.cos(theta)
            tige_positions.append(((x_prev, y_prev), (x_next, y_next)))
            x_prev, y_prev = x_next, y_next
        return tige_positions

    def S(self,p,i):
        n = len(self.theta)
        if p <= i:
            return sum(self.masses[k] for k in range(i, n))
        else:
            return sum(self.masses[k] for k in range(p, n))

class Rectangle:

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.boules = []

    def contains(self, boule):
        # Calcule la distance entre le centre de la boule et le rectangle
        circleDistance_x = abs(boule.x - (self.x + self.width / 2))
        circleDistance_y = abs(boule.y - (self.y + self.height / 2))

        # Vérifie si la boule est trop loin du rectangle pour se chevaucher
        if circleDistance_x > (self.width / 2 + boule.radius):
            return False
        if circleDistance_y > (self.height / 2 + boule.radius):
            return False

        # Vérifie si la boule est suffisamment proche pour garantir un chevauchement
        if circleDistance_x <= (self.width / 2):
            return True
        if circleDistance_y <= (self.height / 2):
            return True

        # Vérifie le chevauchement dans les coins du rectangle
        cornerDistance_sq = ((circleDistance_x - self.width / 2) ** 2 +
                             (circleDistance_y - self.height / 2) ** 2)

        return cornerDistance_sq <= (boule.radius ** 2)
    
    def intersects(self, other):
        # Vérifie si ce rectangle (self) se chevauche avec un autre rectangle (other)
        return (self.x < other.x + other.width and
                self.x + self.width > other.x and
                self.y < other.y + other.height and
                self.y + self.height > other.y)
    
quadtree_patches = []


class QTNode:
	def __init__(self):
		self.children = []
	
	# returns true if a node is a leaf in the quadtree
	def isLeaf(self):
		for child in self.children:
			if child is not None:
				return False
		return True

# Quad Tree data structure
class QuadTree:
    def __init__(self, boundingBox):
        self.root = None
        self.boundingBox = boundingBox 
        self.points = []
        self.maxLimit = 4
        self.patches = []

    # returns true if quadtree is empty
    def isEmpty(self):
        return self.root is None

   
    def find(self, qtree, boule):
        if qtree.root.isLeaf(): 
            return self
        for i in range(4):
            q = qtree.root.children[i]
            if self.boundingBox.contains(boule):
                if q.isEmpty():
                    return q
                return q.find(q, boule)

        return self
    
    def draw(self, ax):
        rect = patches.Rectangle((self.boundingBox.x, self.boundingBox.y), self.boundingBox.width, self.boundingBox.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        quadtree_patches.append(rect)  # Add this patch to the global list

        if self.root and self.root.children:  # Ensure root and children are not None
            for child in self.root.children:
                if child:
                    child.draw(ax)


    def insert(self, boule):
        
        if not self.boundingBox.contains(boule):
            return
        
        if self.isEmpty():
            self.root = QTNode()
            self.points.append(boule)
            return

        
        if self.root.isLeaf():

            self.points.append(boule)
            if len(self.points) > self.maxLimit:
                self.split()
        else:
            
            for child in self.root.children:
                if child.boundingBox.contains(boule):
                    child.insert(boule)
                    

    def split(self):
        if self.isEmpty():
            return

        x = self.boundingBox.x
        y = self.boundingBox.y
        w = self.boundingBox.width
        h = self.boundingBox.height

        
        q1 = QuadTree(Rectangle(x, y, w / 2, h / 2))
        q2 = QuadTree(Rectangle(x + w / 2, y, w / 2, h / 2))
        q3 = QuadTree(Rectangle(x, y + h / 2, w / 2, h / 2))
        q4 = QuadTree(Rectangle(x + w / 2, y + h / 2, w / 2, h / 2))

        assert len(self.points) > self.maxLimit

        
        for boule in self.points:
            if q1.boundingBox.contains(boule):
                q1.insert(boule)
            if q2.boundingBox.contains(boule):
                q2.insert(boule)
            if q3.boundingBox.contains(boule):
                q3.insert(boule)
            if q4.boundingBox.contains(boule):
                q4.insert(boule)

        self.root.children = [q1, q2, q3, q4]

    # Method to detect collisions for the quadtree built for the frame
    def check_collision(self):
        # DFS
        if self.isEmpty():
            return 
        if self.root.isLeaf():
            # A leaf can contain a maximum of maxLimit number of objects.
            # So, an n^2 approach here is okay.
            i = 0
            numPoints = len(self.points)
            
            
            while i < numPoints:
                j = i + 1
                while j < numPoints:

                    if self.points[i].collides_with(self.points[j]):
                        
                        self.points[i].update_velocity_after_collision(self.points[j])

                    j += 1
                i += 1
            return
            

        # Recurse on children
        for i in range(4):
            self.root.children[i].check_collision()

    
       

class Carre:
    def __init__(self, width, height, pendules):
        self.width = width
        self.height = height
        self.boules = []
        self.pendules = pendules
        self.lines = [plt.plot([], [], 'o-', lw=2, color = "black")[0] for _ in pendules]  
        self.quadtree = QuadTree(Rectangle(0, 0, width, height))
        self.max_radius = 0

    def add_boule(self, boule):
        self.boules.append(boule)
        if boule.radius > self.max_radius :
            self.max_radius = boule.radius
        self.quadtree.max_radius = self.max_radius

    def update(self, dt):
        self.check_collisions()
        for boule in self.boules:
            boule.update_position(dt)


    def handle_double_boule_pendulum_collision(self):

        
        pendules = self.pendules
        if len(self.pendules) < 1:
            return
        for pi in range(len(pendules)):
            for pj in range(pi + 1, len(pendules)):
                pendule1 = pendules[pi]
                pendule2 = pendules[pj]
                
                if pendule1 != pendule2 :

                    pendulum_positions1 = pendule1.get_mass_positions()
                    pendulum_positions2 = pendule2.get_mass_positions()

                    for i1, (pendulum_x1, pendulum_y1) in enumerate(pendulum_positions1):
                        for i2, (pendulum_x2, pendulum_y2) in enumerate(pendulum_positions2):


                            # Rayon de la masse du pendule
                            pendulum_radius1 = pendule1.radius[i1]
                            pendulum_radius2 = pendule2.radius[i2]

                            # Calculez la distance entre le centre de la boule et la masse du pendule
                            distance = np.hypot(pendulum_x1 - pendulum_x2, pendulum_y1 - pendulum_y2)

                            # Vérifiez si la distance est inférieure à la somme des rayons
                            if distance <= pendulum_radius2 + pendulum_radius1:
                                
                                vx01,vy01 = 0,0
                                for indice in range(i1+1):
                                    
                                    vx01,vy01 = vx01 + pendule1.lengths[indice] * np.cos(pendule1.theta[indice]) * pendule1.omega[indice], vy01 + pendule1.lengths[indice] * np.sin(pendule1.theta[indice]) * pendule1.omega[indice]

                                vx02,vy02 = 0,0
                                for indice in range(i2+1):
                                    
                                    vx02,vy02 = vx02 + pendule2.lengths[indice] * np.cos(pendule2.theta[indice]) * pendule2.omega[indice], vy02 + pendule2.lengths[indice] * np.sin(pendule2.theta[indice]) * pendule2.omega[indice]

                                pendulum_boule1 = Boule(pendulum_x1, pendulum_y1, pendulum_radius1, vx01, vy01, 0, 0, "color", pendule1.masses[i1])
                                pendulum_boule2 = Boule(pendulum_x2, pendulum_y2, pendulum_radius2, vx02, vy02, 0, 0, "color", pendule2.masses[i2])
                                pendulum_boule1.update_velocity_after_collision(pendulum_boule2) #collision libre
                                
                                vx1,vy1 = pendulum_boule1.vx , pendulum_boule1.vy
                                vx2,vy2 = pendulum_boule2.vx , pendulum_boule2.vy
                                m1 = pendule1.masses[i1]
                                m2 = pendule2.masses[i2]
                                
                                P1 = m1*(vx1-vx01, vy1-vy01) 
                                P2 = m2*(vx2-vx02, vy2-vy02)
                                
                                rond_r1 = (pendule1.lengths[i1] * np.cos(pendule1.theta[i1]),pendule1.lengths[i1] * np.sin(pendule1.theta[i1]))
                                rond_r2 = (pendule2.lengths[i2] * np.cos(pendule2.theta[i2]),pendule2.lengths[i2] * np.sin(pendule2.theta[i2]))
                                prod1 = P1[0] * rond_r1[0] + P1[1] * rond_r1[1]
                                prod2 = P2[0] * rond_r2[0] + P2[1] * rond_r2[1]
                                
                                n1 = len(pendule1.theta) 
                                n2 = len(pendule2.theta) 
                                # Initialiser les matrices A et b
                                A1 = np.zeros((n1, n1))
                                b1 = np.zeros(n1)

                                A2 = np.zeros((n2, n2))
                                b2 = np.zeros(n2)

                                # Remplir les matrices A et b

                                for k in range(n1):
                                    
                                    theta_k = pendule1.theta[k]
                                    l_k = pendule1.lengths[k]
                        
                                    for l in range(n1):
                                        theta_l = pendule1.theta[l]
                                        l_l = pendule1.lengths[l]
                                        
                                        if k == l:
                                            A1[k, l] = pendule1.lengths[k]**2 * sum(pendule1.masses[indice] for indice in range(k, len(pendule1.masses)))

                                        else:
                                            A1[k, l] =  2 * sum(pendule1.masses[indice] for indice in range(max(k,l), len(pendule1.masses))) * l_l * l_k * np.cos(theta_k - theta_l)
                                    
                                    
                                    b1[k] = sum(A1[k,l]*pendule1.omega[l] for l in range(n1))
                                
                                # Résoudre le système d'équations Ax = b
                                b1[i1]+=prod1
                                x1 = np.linalg.solve(A1, b1)

                                for i in range(n1):
                                    
                                    pendule1.omega[i] = x1[i]
                                
                                for k in range(n2):
                                    
                                    theta_k = pendule2.theta[k]
                                    l_k = pendule2.lengths[k]
                        
                                    for l in range(n2):
                                        theta_l = pendule2.theta[l]
                                        l_l = pendule2.lengths[l]
                                        
                                        if k == l:
                                            A2[k, l] = pendule2.lengths[k]**2 * sum(pendule2.masses[indice] for indice in range(k, len(pendule2.masses)))

                                        else:
                                            A2[k, l] =  2 * sum(pendule2.masses[indice] for indice in range(max(k,l), len(pendule2.masses))) * l_l * l_k * np.cos(theta_k - theta_l)
                                    
                                    
                                    b2[k] = sum(A2[k,l]*pendule2.omega[l] for l in range(n2))
                                
                                # Résoudre le système d'équations Ax = b
                                b2[i2]+=prod2
                                x2 = np.linalg.solve(A2, b2)
                                
                                for i in range(n2):
                                    
                                    pendule2.omega[i] = x2[i]
                                
                                
    def check_tige_collision(self, tige, boule):
        # Vérifier la collision entre une tige et une boule
        (x1, y1), (x2, y2) = tige
        cx, cy, r = boule.x, boule.y, boule.radius

        # Calculer la distance la plus proche entre le centre de la boule et la tige
        dx, dy = x2 - x1, y2 - y1
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
        
        if t > 0 and t < 1 :
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

            # Vérifier la collision
            distance = np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
            return (distance < r, cx > closest_x)
        else :
            return (False, None)
    



                        
                                
    def check_collisions(self):

        self.quadtree = QuadTree(Rectangle(0, 0, self.width, self.height))
        for boule in self.boules:
            self.quadtree.insert(boule)

        # Quadtree use
        self.quadtree.check_collision()

        
        normal_left = [1, 0]  
        normal_right = [-1, 0] 
        normal_top = [0, -1]  
        normal_bottom = [0, 1]  
        # Check for wall collisions

        for boule in self.boules:
            
            
            if boule.x - boule.radius < 0 and np.dot([boule.vx, boule.vy], normal_left) < 0:
                boule.vx *= -1
            if boule.x + boule.radius > self.width and np.dot([boule.vx, boule.vy], normal_right) < 0:
                boule.vx *= -1
            if boule.y - boule.radius < 0 and np.dot([boule.vx, boule.vy], normal_bottom) < 0:
                boule.vy *= - 1
            if boule.y + boule.radius > self.height and np.dot([boule.vx, boule.vy], normal_top) < 0:
                boule.vy *= -1 


        # free ball and bob collision
        for boule in self.boules :
            for pendule in self.pendules :
                self.handle_boule_pendulum_collision(boule, pendule)
        

        self.handle_double_boule_pendulum_collision()
    
    def handle_boule_pendulum_collision(self, boule, pendule):

        pendulum_positions = pendule.get_mass_positions()
        
        # Rayon de la boule
        boule_radius = boule.radius

        for i, (pendulum_x, pendulum_y) in enumerate(pendulum_positions):

            # Rayon de la masse du pendule
            pendulum_radius = pendule.radius[i] 

            # Calculez la distance entre le centre de la boule et la masse du pendule
            distance = np.hypot(boule.x - pendulum_x, boule.y - pendulum_y)

            # Vérifiez si la distance est inférieure à la somme des rayons
            if distance <= boule_radius + pendulum_radius:
                
                vx0,vy0 = 0,0
                for indice in range(i+1):
                    
                    vx0,vy0 = vx0 + pendule.lengths[indice] * np.cos(pendule.theta[indice]) * pendule.omega[indice], vy0 + pendule.lengths[indice] * np.sin(pendule.theta[indice]) * pendule.omega[indice]

                
                pendulum_boule = Boule(pendulum_x, pendulum_y, pendulum_radius, vx0, vy0, 0, 0, "color", pendule.masses[i])
                
                boule.update_velocity_after_collision(pendulum_boule) #collision libre
                
                vx,vy = pendulum_boule.vx , pendulum_boule.vy
                
                m = pendule.masses[i]

                P = m*(vx-vx0, vy-vy0) #impulsion différence de vitesse

                rond_r = (pendule.lengths[i] * np.cos(pendule.theta[i]),pendule.lengths[i] * np.sin(pendule.theta[i]))

                prod = P[0] * rond_r[0] + P[1] * rond_r[1]
                
                n = len(pendule.theta) 

                # Initialiser les matrices A et b
                A = np.zeros((n, n))
                b = np.zeros(n)

                # Remplir les matrices A et b

                for k in range(n):
                    
                    theta_k = pendule.theta[k]
                    l_k = pendule.lengths[k]
        
                    for l in range(n):
                        theta_l = pendule.theta[l]
                        l_l = pendule.lengths[l]
                        
                        if k == l:
                            A[k, l] = pendule.lengths[k]**2 * sum(pendule.masses[indice] for indice in range(k, len(pendule.masses)))

                        else:
                            A[k, l] =  2 * sum(pendule.masses[indice] for indice in range(max(k,l), len(pendule.masses))) * l_l * l_k * np.cos(theta_k - theta_l)
                    
                    
                    b[k] = sum(A[k,l]*pendule.omega[l] for l in range(n))
                
                # Résoudre le système d'équations Ax = b
                b[i]+=prod
                x = np.linalg.solve(A, b)

                for i in range(n):
                    
                    pendule.omega[i] = x[i]
                

def update(frame_number, carre, circles_carre, circles_pendules, dt):

    global quadtree_patches

    # Remove existing patches from the axes to prevent clutter
    for p in quadtree_patches:
        p.remove()
    quadtree_patches = []
    carre.update(dt) #check collision + update

    ax = plt.gca()
    carre.quadtree.draw(ax)
    # Mise à jour des positions des cercles pour le carré
    for circle, boule in zip(circles_carre, carre.boules):
        circle.center = (boule.x, boule.y)
        
    k_p = 0

    for pendulum in carre.pendules :

        pendulum_line = carre.lines[k_p]
        pendulum.update(dt)  # Mettre à jour l'état du pendule
        circles_pendule = circles_pendules[k_p]
        theta = pendulum.theta
        xstart, ystart = pendulum.xstart, pendulum.ystart
        lengths = pendulum.lengths

        # Mise à jour de la position du pendule
        x = [xstart]
        y = [ystart]
        # Obtenir les axes actuels

        # Supprimer les anciens cercles du pendule
        for circle in circles_pendule:
            circle.remove()
        circles_pendule.clear()

        for i in range(len(pendulum.lengths)):
            x.append(x[-1] + lengths[i] * np.sin(theta[i]))
            y.append(y[-1] - lengths[i] * np.cos(theta[i]))
            # Ajouter un nouveau cercle pour le pendule
            circle = plt.Circle((x[-1], y[-1]), pendulum.radius[i], color = pendulum.colors[i], zorder=2)
            ax.add_patch(circle)
            circles_pendule.append(circle)  # Ajouter le cercle du pendule à la liste séparée

        pendulum_line.set_data(x, y)
        k_p+=1


def run_simulation(width, height, num_boules, dt):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)  
    ax.set_title("SEGONNES Mattéo", fontsize=16, fontweight='bold', pad=20)


    '''  Pendule 1'''

    xstart, ystart = 34, 60
    initial_theta = np.array([1,1,1,1])
    initial_omega = np.array([0,0,0,1])
    lengths = np.array([5,10,14,15])
    masses = np.array([10,7,20,6])
    radius = np.array([1.8,1.5,2.6,1.2])
    colors = np.random.rand(4, 3)
    pendulum = PendulumSystem(initial_theta, initial_omega, lengths, masses, xstart, ystart, radius, colors)

    #--------------------------------------------------------------------------------------------------------#


    '''  Pendule 2'''
    
    xstart_new, ystart_new = 70, 50
    initial_theta_new = np.array([1, 1, -1, -1])  
    initial_omega_new = np.array([1, 0, 0, 1])  
    lengths_new = np.array([10, 10, 20, 15])     
    masses_new = np.array([3, 10, 20, 10])       
    radius_new = np.array([1.8, 1.5, 2.6, 1.2]) 
    colors_new = np.random.rand(4, 3)         
    pendulum_new = PendulumSystem(initial_theta_new, initial_omega_new, lengths_new, masses_new, xstart_new, ystart_new, radius_new, colors_new)

    #--------------------------------------------------------------------------------------------------------#


    carre = Carre(width, height, [pendulum, pendulum_new])
   
    circles = []
    for _ in range(num_boules):
    
        new_radius = random.uniform(0.3, 3.6)
        # Générer des positions en tenant compte du rayon
        x = random.uniform(new_radius, width - new_radius)
        y = random.uniform(new_radius, height - new_radius)
        
        boule = Boule(
            x=x,
            y=y,
            radius=new_radius,
            vx=random.uniform(-1, 1)*30,
            vy=random.uniform(-1, 1)*40,
            ax=0,
            ay=0,
            color=np.random.rand(3,), # Couleurs aléatoires
            masse = new_radius*4
                
        )
        carre.add_boule(boule)


    for boule in carre.boules :
        # Create a circle and add it to the list and axes
        circle = plt.Circle((boule.x, boule.y), boule.radius, color=boule.color)
        
        ax.add_patch(circle)
        circles.append(circle)

    ani = animation.FuncAnimation(fig, update, fargs=(carre, circles, [[],[],[],[]], dt), interval=1)
    plt.show()

run_simulation(110, 80, 0, 0.03)