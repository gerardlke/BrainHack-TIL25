"""Manages the RL model."""
import cv2
import numpy as np
from einops import rearrange
import copy
import random

bit_ops = eval(open('bitops.txt').read())

class RLManager:

    def __init__(self, vis=True):

        self.vis = vis
        self.vis_tile_size = 100
        self.view_padding = 1 # when visualizing, you get a global view of larger axis padded by this number instead of whatever by whatever
        # e.g if your col is the larger dimension, you pad it by 1, you will visualize a 9 by 9 area. ofc all padded area will be 0 bit.
        self.fixed_view = True  # purely visualization sake. if so, and the guy rotates, we rotate the observation we get to fit the original view 
        assert self.vis_tile_size % 2 == 0, 'make your tile size divisible by 2 la'
        self.viewcone_row = 5
        self.viewcone_col = 7
        # dictionary of operations to perform, if the bit in that position is flipped.
        # keys: int representing position of bit, values are a list of hyperparams to draw 
        self.bit_ops = bit_ops

    def draw_function(self, bit_tile, template, coords):
        """
        Helper function to draw stuffs on a tile.
        Recieves a tile's information, and runs the relevant operations to draw on the template.

        Args:
            - bit_tile: 8-size binary vector e.g [1 1 0 0 0 0 0 0], each bit denoting some information about the tile.
            - template: the numpy object to draw on
            - coords: (row, col) 0-indexed coordinate of the current tile we are operating on.

        Returns: 
            - template
        """
        # coord stuff
        halfway_value = int(self.vis_tile_size / 2)
        # all this is (x, y)
        tile_top_left = (coords[1] * self.vis_tile_size, coords[0] * self.vis_tile_size)
        tile_top_right = ((coords[1] + 1) * self.vis_tile_size, coords[0] * self.vis_tile_size)
        tile_bot_left = (coords[1] * self.vis_tile_size, (coords[0] + 1) * self.vis_tile_size)
        tile_bot_right = ((coords[1] + 1) * self.vis_tile_size, (coords[0] + 1) * self.vis_tile_size)

        center = (coords[1] * self.vis_tile_size + halfway_value, coords[0] * self.vis_tile_size + halfway_value, )
        values = copy.deepcopy(list(self.bit_ops.values()))
        taken = [values[idx][i] for idx, i in enumerate(bit_tile[::-1]) if values[idx][i] is not None]  # list of operations to perform.

        # draw text bit value on the visualization
        bits = str(np.packbits(bit_tile)[0])
        # cv2.putText(img=template, text=bits, org=center, )
        # some additional logic is in place however

        if 'Black tile' in taken and len(taken) == 2:  # this is because the second index has ops for both bit states, so it will always demand an op
            # even though it is a Black tile. if there are other things to draw, 'Black tile' may have been set but we will not draw a black tile.
            # tile is out of bounds
            template[coords[0] * self.vis_tile_size:(coords[0] + 1) * self.vis_tile_size, coords[1] * self.vis_tile_size:(coords[1] + 1) * self.vis_tile_size, :] = 42
        else:
            
            for op in taken:
                if isinstance(op, dict):
                    func = op.pop('type')
                    if 'circle' in func.__name__:
                        op['center'] = center
                    elif 'rectangle' in func.__name__:
                        size = op.pop('size')
                        half_size = int(size / 2)
                        top_left = (center[0] - half_size, center[1] - half_size)
                        bot_right = (center[0] + half_size, center[1] + half_size)
                        op['pt1'] = top_left
                        op['pt2'] = bot_right
                    elif 'line' in func.__name__:
                        orientation = op.pop('orientation')
                        if orientation == 'right':
                            p1 = tile_top_right
                            p2 = tile_bot_right
                        if orientation == 'left':
                            p1 = tile_top_left
                            p2 = tile_bot_left
                        if orientation == 'bottom':
                            p1 = tile_bot_left
                            p2 = tile_bot_right
                        if orientation == 'top':
                            p1 = tile_top_right
                            p2 = tile_top_left
                        op['pt1'] = p1
                        op['pt2'] = p2
        
                    # we take the remaining op as a kwarg dict to throw at the func
                    func(img=template, **op)

        return template



    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        viewcone = observation['viewcone']
        viewcone = np.array(viewcone).transpose(1, 0)
        # print('viewcone', viewcone)

        action = random.randint(0, 4)
        if self.fixed_view:
            if action == 2:
                k = 1
            elif action == 3:
                k = -1
            else:
                k = 0
        #         # turned left, so rotate incoming observation right 



        # if self.vis:
        #     temp_viewcone = np.rot90(viewcone, k=k)  # rotated to fit original orientation, but because viewcone is skew forward, may not be the same shape
        #     row, col = temp_viewcone.shape
        #     pad_shorter = abs(col - row) // 2 + self.view_padding
        #     pad_longer = self.view_padding
        #     if row > col:
        #         pad_row, pad_col = pad_longer, pad_shorter
        #     else:
        #         pad_row, pad_col = pad_shorter, pad_longer

        #     padded_viewcone = np.pad(temp_viewcone, ((pad_row, pad_row), (pad_col, pad_col)), 
        #             mode='constant', constant_values=0)
        #     viewcone_shape = padded_viewcone.shape[0]
        #     print('padded_viewcone', padded_viewcone)
        #     print('temp_viewcone', temp_viewcone, temp_viewcone.shape)
        #     plate = np.full((viewcone_shape * self.vis_tile_size, 
        #                      viewcone_shape * self.vis_tile_size,
        #                      3), 255).astype(np.uint8)
        #     bit_planes = np.unpackbits(padded_viewcone.astype(np.uint8))
        #     bit_planes = rearrange(bit_planes, 
        #         '(R C B) -> (R C) B', 
        #         R=viewcone_shape, C=viewcone_shape, B=8)

        #     for idx, tile in enumerate(bit_planes):
        #         # use 1-indexing for convenience in draw function
        #         row = (idx) // viewcone_shape  # divide by length of row
        #         col = (idx) % viewcone_shape
        #         # print('row col tile', row, col, idx)
        #         plate = self.draw_function(tile, plate, (row, col))
            
        #     cv2.imshow('plate', plate)
        #     cv2.waitKey(1000)
                

        # Your inference code goes here. --ben no

        # just to visualize observations for now in a seperate cv2 window
        # 

        # cv2.imshow('plate', plate)
        # cv2.waitKey(1)
        
        # print('plate', plate.shape)

        return action
