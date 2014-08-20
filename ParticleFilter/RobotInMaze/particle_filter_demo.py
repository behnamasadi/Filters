#!/usr/bin/python

# Particle Filter Demonstration
#
#   Imagine a robot whose environment is a floor tiled with a random
#   pattern of black or white tiles. The robot has a map of the floor
#   but can only sense - unreliably - whether it is above a black or
#   a white tile.
#
#   This program demonstrates how the robot can use a particle filter
#   to locate itself on the map. As you click each robot position, the
#   particle filter will update its sense of where the robot is. Over
#   a number of moves and measurements, the partcles converge and
#   provide a good estimate of the robot's true position which is
#   indicated by the cursor.
#
#   Written by Pete Marshall (petemarshall77@gmail.com) December 2011
#
#   Notes:
#
#      Written (well, hacked together) while trying to learn Python and
#      wxPython at the same time. Is probably non-idiomatic and full of
#      holes. Feel free to improve
#
import wx, random
random.seed()

map_width = 600            # map width in pixels 
map_height = 460           # map height in pixels
block_size = 20            # size of tile in pixels
particle_count = 5000      # number of partcles in filter

# Filter parameters
move_noise = 4             # particle moves will be +/- this amount
p_black_given_black = 0.8  # sensor reliablity
p_white_given_white = 0.9  # sensor reliability

# Set up the environment
#random.randint(a, b)
#Return a random integer N such that a <= N <= b.

environment = []
for x in range(map_width/block_size):
    x_vals = []
    for y in range(map_height/block_size):
        x_vals.append(random.randint(0,1))
    environment.append(x_vals)


class Map(wx.Frame):

    # Indicate we have no location for the robot
    current_x = -1
    current_y = -1

    # Array of particle values (and newly sampled particles
    particles =[None] * particle_count
    new_particles =[None] * particle_count
    
    # Initialize the wx.Frame
    def __init__(self, parent, title):
        super(Map, self).__init__(parent, title=title,
                                  size=(map_width, map_height))

        # Event handlers
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)

        # Show the window
        self.Centre()
        self.Show()

        # Initially the particles are located at random points on the map
        for count in range(particle_count):
            weight = 1.0/particle_count
            x_pos = random.randint(0, map_width-1)
            y_pos = random.randint(0, map_height-1)
            self.particles[count] = {'weight':weight, 'xpos':x_pos, 'ypos':y_pos}
            self.new_particles[count] = {'weight':0.0, 'xpos':0, 'ypos':0}
            
    # Paint the window        
    def OnPaint(self, e):
        dc = wx.PaintDC(self)
        dc.SetPen(wx.Pen('#ffffff'))

        # Draw the environment
        for x in range(map_width/block_size):
            for y in range(map_height/block_size):
                if environment[x][y] == 1:
                    dc.SetBrush(wx.Brush('#999999'))  # Black black
                else:
                    dc.SetBrush(wx.Brush('#ffffff'))  # White square
                    
                # Draw the rectangle
                dc.DrawRectangle(x*block_size,
                                 y*block_size,
                                 x*block_size+block_size-1,
                                 y*block_size+block_size-1)

        # Draw the particles
        dc.SetPen(wx.Pen('RED'))
        for count in range(particle_count):
            dc.DrawPoint(self.particles[count]['xpos'],
                         self.particles[count]['ypos'])

        # Draw a cursor at the current location
        dc.SetPen(wx.Pen('BLUE'))
        cursiz = 8
        if self.current_x != -1:
            dc.DrawLine(self.current_x-cursiz, self.current_y-1,
                        self.current_x, self.current_y-1)
            dc.DrawLine(self.current_x-1, self.current_y-cursiz,
                        self.current_x-1, self.current_y)
            dc.DrawLine(self.current_x+1, self.current_y-cursiz,
                        self.current_x+1, self.current_y)
            dc.DrawLine(self.current_x+cursiz, self.current_y-1,
                        self.current_x+1, self.current_y-1)
            dc.DrawLine(self.current_x-cursiz, self.current_y+1,
                        self.current_x, self.current_y+1)
            dc.DrawLine(self.current_x-1, self.current_y+cursiz,
                        self.current_x-1, self.current_y)
            dc.DrawLine(self.current_x+1, self.current_y+cursiz,
                        self.current_x+1, self.current_y)
            dc.DrawLine(self.current_x+cursiz, self.current_y+1,
                        self.current_x+1, self.current_y+1)

    # Process a mouse click
    def OnClick(self, e):

        point = e.GetPosition()

        # If first time here, just set the initial position
        if self.current_x == -1:
            self.current_x = point[0]
            self.current_y = point[1]
            self.OnPaint(self)
            return
        
        #----------------------------------
        # Particle filter code starts here
        #----------------------------------

        # Calculate how far the robot has moved
        # (This is exact, in the real world it would be noisy
        #  but this doesn't change the demo we're doing here.)
        delta_x = point[0] - self.current_x
        delta_y = point[1] - self.current_y

        # Calculate the new position for each particle, adding noise
        for count in range(particle_count):
            
            # Calculate new position    
            newx = self.particles[count]['xpos'] + delta_x + random.randint(-move_noise, move_noise)
            newy = self.particles[count]['ypos'] + delta_y + random.randint(-move_noise, move_noise)
        
            # If new position is out of bounds, pick new random position
            if (newx < 0) or (newx >= map_width) or (newy < 0) or (newy >= map_height):
                newx = random.randint(0, map_width-1)
                newy = random.randint(0, map_height-1)

            # Update the position    
            self.particles[count]['xpos'] = newx
            self.particles[count]['ypos'] = newy

        # Make the measurement - sense the environment
        measurement = environment[point[0]/block_size][point[1]/block_size]

        # Calculate the new particle weights
        total_weight = 0.0 # for normalization
        for count in range(particle_count):
            particle_state = environment[self.particles[count]['xpos']/block_size][self.particles[count]['ypos']/block_size]
            
            # Calculate the particle weight, factoring in
            # the sensor reliability
            # (note: measurement == 1 is a black square)
            if (measurement == 0) and (particle_state == 0):
                new_weight = p_white_given_white
            elif (measurement == 0) and (particle_state == 1):
                new_weight = 1-p_black_given_black
            elif (measurement == 1) and (particle_state == 0):
                new_weight = 1-p_white_given_white
            else:
                new_weight = p_black_given_black

            # Update the new particle weight and the running total
            self.particles[count]['weight'] = new_weight    
            total_weight = total_weight + new_weight
            
        # Now we have the new weights for the paricles, we need
        # to normalize them
        for count in range(particle_count):
            self.particles[count]['weight'] = self.particles[count]['weight']/total_weight

        # Resampling
        #    We need weighted sampling with replacement. To get
        #    replacement we just copy the sampled particles into
        #    a new array. To get the weighting, we create a new
        #    array which contains the cumulative weights of the
        #    particles. What we create, essentially, is a view of
        #    the number line between 0 and 1, segmented by each
        #    particle's weight: particles of greater weight will
        #    occupy more of the number line.
        #
        #    Then we pick a random weight between 0 and 1 and 
        #    search for that in the index. The position in the 
        #    index will give us the particle to sample.

        # Resample Phase 1 - create an index of the weights
        total_samples = 0
        cumulative_weight = 0.0
        weights_index = [None] * particle_count
        for count in range(particle_count):
            cumulative_weight = cumulative_weight + self.particles[count]['weight']
            weights_index[count] = cumulative_weight
        
        # Resample Phase 2 - create a list of the new samples
        sample_list = [None] * particle_count
        for count in range(particle_count):
            # Pick a random weight and find it in the
            # list (by binary chop)
            random_weight = random.random()
            indexLow = 0
            indexHigh = particle_count - 1
            while indexLow != indexHigh - 1:
                indexMiddle = indexLow + int((indexHigh - indexLow)/2)
                if weights_index[indexMiddle] < random_weight:
                    indexLow = indexMiddle
                else:
                    indexHigh = indexMiddle

            sample_list[count] = indexHigh

        # Resample Phase 3 - replace the current samples
        for count in range(particle_count):
            self.new_particles[count] = self.particles[sample_list[count]]

        for count in range(particle_count):
            self.particles[count]['xpos'] = self.new_particles[count]['xpos']
            self.particles[count]['ypos'] = self.new_particles[count]['ypos']
            self.particles[count]['weight'] = self.new_particles[count]['weight']
        # New state complete: save current robot location
        # and force redraw
        self.current_x = point[0]
        self.current_y = point[1]
        self.OnPaint(self)

# The action starts here
if __name__ == '__main__':
#    print  environment
    app = wx.App()
    Map(None, 'Particle Filter')
    app.MainLoop()