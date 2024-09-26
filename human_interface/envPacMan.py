import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe


""" --------------------------------------------------------------------------
Ghost - class
"""
class ghost:
    def __init__(self, x, y, map):
        self.init_pos = [x, y]
        self.reset()

        self.dir = 's'
        self.map = map
        
        self.P_change_direction = 0.5        
        return

    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = self.init_pos
        else:
            self.pos = [x,y]
            
        self.prepos = self.pos       
        return
    
    def move(self):
        #check if the same movement is possible
        newPos = self.newPos(self.pos, self.dir)
        cantMove = (newPos == self.pos)
        
        if cantMove or np.random.rand() < self.P_change_direction:
            newPos == self.pos.copy()
            while newPos == self.pos:
                r = np.random.rand()
                if  r < 1/4:
                    self.dir = 'n'
                elif r < 2/4:
                    self.dir = 's'
                elif r < 3/4:
                    self.dir = 'w'
                else:
                    self.dir = 'e'
                
                newPos = self.newPos(self.pos, self.dir)
            
        # actually move
        self.prepos = self.pos
        self.pos = newPos        
        return
    
    def newPos(self, currPos, dir):
        
        newPos = currPos.copy()      
        if dir == 'n':
            newPos[1] -= 1
        elif dir == 's':
            newPos[1] += 1
        elif dir == 'w':
            newPos[0] -= 1
        elif dir == 'e':
            newPos[0] += 1
        if self.map[newPos[1]][newPos[0]] == '#':
            newPos = currPos           
        return newPos

""" --------------------------------------------------------------------------
Pacman - class
"""
class pacman:
    def __init__(self, x, y, map):
        self.init_pos = [x, y]
        self.reset()
        
        self.map = map
        return
    
    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = self.init_pos
        else:
            self.pos = [x,y]
        self.prepos = self.pos
        return
    
    def move(self, action):
        
        self.prepos = self.pos
        self.pos = self.newPos(self.pos, action)    
        return
    
    def newPos(self, currPos, dir):
        
        newPos = currPos.copy()
        if dir == 'n':
            newPos[1] -= 1
        elif dir == 's':
            newPos[1] += 1
        elif dir == 'w':
            newPos[0] -= 1
        elif dir == 'e':
            newPos[0] += 1
    
        if self.map[newPos[1]][newPos[0]] == '#':
            newPos = currPos    
        return newPos

""" --------------------------------------------------------------------------
Pellets - class (Toplevel)
"""
class pellets:
    def __init__(self, pos_list):
        self.pos_list = np.array(pos_list)
        self.valid = np.array([ True for i in range(len(pos_list))])
        return
    
    def reset(self):
        self.valid[:] = True
    
    def remaining_pellets(self):
        return self.pos_list[self.valid]
    
    def number_remaining_pellets(self):
        return np.sum(self.valid)
    
    def eaten(self, pos):
        ret = False
        for i in range(len(self.valid)):
            if all( np.array(pos) == self.pos_list[i] ) and self.valid[i]:
                self.valid[i] = False
                ret = True
        return ret
    
""" --------------------------------------------------------------------------
env - class (Toplevel)
"""
class env:

    def __init__(self):
        self.map = list()
        self.map.append('#######')
        self.map.append('#     #')
        self.map.append('# ### #')
        self.map.append('# #   #')
        self.map.append('# ### #')
        self.map.append('#     #')
        self.map.append('#######')
                                
        self.pacman = pacman(1,1, self.map)
        self.ghost  = ghost(5,5, self.map)
        self.pellets = pellets([[3,3], [1,5]])

        self.obstacles = [[2, 2], [2, 3], [2, 4], [3, 2], [4, 2], [3, 4], [4, 4]]
        self.map_size_x = len(self.map[0]) - 2
        self.map_size_y = len(self.map) - 2
        self.num_pellets = self.pellets.number_remaining_pellets()

        self.pacman_img = image.imread('./sprites/pacman.png')
        self.ghost_img = image.imread('./sprites/ghost.png')
        self.pellet_img = image.imread('./sprites/cherry.png')
        self.walls = self.add_walls()
        return
    
    def wall_exists(self,coord1,coord2,walls):
        if [coord1,coord2] in walls or [coord2,coord1] in walls:
            return True
        else:
            return False
        
    def add_walls(self):
        walls = []
        margin = 0.5
        # plot boundaries of arena
        # walls.append([[margin,margin],[self.map_size_x+margin,margin]])
        # walls.append([[margin,self.map_size_y+margin],[self.map_size_x+margin,self.map_size_y+margin]])
        # walls.append([[margin,margin],[margin,self.map_size_y+margin]])
        # walls.append([[self.map_size_x+margin,margin],[self.map_size_x+margin,self.map_size_y+margin]])
        walls.append([[margin,margin],[self.map_size_x+margin,margin],
                      [margin,margin],[margin,self.map_size_y+margin],
                      [margin,self.map_size_y+margin],[self.map_size_x+margin,self.map_size_y+margin],
                      [self.map_size_x+margin,margin],[self.map_size_x+margin,self.map_size_y+margin]])
        

        for ob_a in self.obstacles:
            for ob_b in self.obstacles:
                if self.wall_exists(ob_a,ob_b,walls):
                    continue
                else:
                    if abs(ob_a[0] - ob_b[0]) == 1 and abs(ob_a[1] - ob_b[1]) == 0:
                        walls.append([ob_a,ob_b])
                    elif abs(ob_a[1] - ob_b[1]) == 1 and abs(ob_a[0] - ob_b[0]) == 0:
                        walls.append([ob_b,ob_a])
        return walls


    def plot(self):
        fig, ax = plt.subplots(figsize = (5,5),edgecolor='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        ax.set_aspect(1.0)
        # Plot vertical lines
        for i in range(1, self.map_size_x + 1):
            ax.plot([1, self.map_size_x],[i, i],  color='#1818A6', zorder=1, linewidth=2)
        for i in range(1, self.map_size_y + 1):
            ax.plot([i, i], [1, self.map_size_y], color='#1818A6', zorder=1, linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_facecolor('#2F2E2E')
        ax.set_facecolor('#000000')
        
        for wall in self.walls:
            x,y = zip(*wall)
            plt.plot(x,y,color='#000000',linewidth=10,path_effects=[pe.Stroke(linewidth=15, foreground='#4663FF'), pe.Normal()], zorder=2)
            

        # for obstacle in self.obstacles:
        #     plt.scatter(obstacle[0],obstacle[1],s = 200,c='k', zorder=2)

        pellet_imagebox = OffsetImage(self.pellet_img, zoom = 0.05)
        for pellet in self.pellets.remaining_pellets():
            a = AnnotationBbox(
                pellet_imagebox,
                (pellet[0], self.map_size_y+1-pellet[1]),
                frameon=False)
            ax.add_artist(a)
            # plt.scatter(pellet[0],self.map_size_y+1-pellet[1],s = 100,c='r')

        pacman_imagebox = OffsetImage(self.pacman_img, zoom = 0.025)
        a = AnnotationBbox(
            pacman_imagebox,
            (self.pacman.pos[0], self.map_size_y+1-self.pacman.pos[1]),
            frameon=False)
        ax.add_artist(a)
        ghost_imagebox = OffsetImage(self.ghost_img, zoom = 0.1)
        b = AnnotationBbox(
            ghost_imagebox,
            (self.ghost.pos[0], self.map_size_y+1-self.ghost.pos[1]),
            frameon=False)
        ax.add_artist(b)

        return fig, ax

    def reset(self, random=False):
        if random:
            xlim = len(self.map[0])
            ylim = len(self.map)
            
            # pacman location
            while True:
                px = np.random.randint(0, xlim)
                py = np.random.randint(0, ylim)
                if self.map[py][px] != '#':
                    self.pacman.reset(px,py)
                    break
            # ghost location
            while True:
                gx = np.random.randint(0, xlim)
                gy = np.random.randint(0, ylim)
                if self.map[gy][gx] != '#' and (px!=gx or py!=gy) and [gx,gy] not in [list(ppos) for ppos in list(self.pellets.pos_list)]:
                    
                    self.ghost.reset(gx,gy)
                    break
        else:
            self.pacman.reset()
            self.ghost.reset()
            
        self.pellets.reset()        
        return
    
    def nStates(self):
        return (self.map_size_x*self.map_size_y) * 4 * (2**self.num_pellets) * (self.map_size_x*self.map_size_y) # Ghost pos. x Ghost direction x pellets x Pacman pos.
        
    def action_list(self):
        return ['n', 's', 'e', 'w']
    
    def step(self, action):
    
        # init. return parameters
        rw = 0
        done = False

        # move pacman
        self.pacman.move(action)
        
        # move ghost
        self.ghost.move()
        
        # collison?
        if self.pacman.pos == self.ghost.pos or (self.pacman.prepos == self.ghost.pos and self.pacman.pos == self.ghost.prepos):
            # Game over
            # self.pacman.pos = self.pacman.prepos
            rw = -500
            done = True
        
        # eat pellet?
        eat = self.pellets.eaten(self.pacman.pos)
        if eat:
            rw += 10
            if self.pellets.number_remaining_pellets() == 0:
                # Clear Game
                rw += 500
                done = True
        else:
            rw -= 1
                
        return [self.st2ob(), rw, done]
    
    # generate display string
    def display(self):
        disp = self.map.copy()

        p_pos = self.pellets.remaining_pellets()
        for i in range(self.pellets.number_remaining_pellets()):
            disp[p_pos[i][1]] = self.replaceChar(disp[p_pos[i][1]], '*', p_pos[i][0])

        disp[self.pacman.pos[1]] = self.replaceChar(disp[self.pacman.pos[1]], 'P', self.pacman.pos[0])
        disp[self.ghost.pos[1]]   = self.replaceChar(disp[self.ghost.pos[1]],   'G', self.ghost.pos[0])
            
        return disp
    
    def replaceChar(self, st, c, idx):
        return st[0:idx] + c + st[idx+1:]
    
    # state to observation conversion
    def st2ob(self):
        gPosIdx = (self.ghost.pos[0]-1) + (self.ghost.pos[1]-1)*self.map_size_x
        gDirIdx = np.argmax( np.array(['n', 's', 'e', 'w']) == self.ghost.dir )
        pPosIdx = (self.pacman.pos[0]-1) + (self.pacman.pos[1]-1)*self.map_size_x
        peltIdx = np.sum( self.pellets.valid * 2**np.arange( len(self.pellets.valid) ) )
        return gPosIdx + gDirIdx*(self.map_size_x*self.map_size_y) + \
               pPosIdx*(self.map_size_x*self.map_size_y) * 4 + \
               peltIdx*(self.map_size_x*self.map_size_y) * 4 *(self.map_size_x*self.map_size_y)
