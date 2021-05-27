import discord
from discord.ext import commands
import cv2 as cv
import numpy as np
import pathlib

from util import aliases, flagset

class Replace(commands.Cog):
    async def generic_replace(self, img, color, full_flag): # TODO: kwargs for rotation amt, scale    
        color_arr = np.array(color, dtype=np.uint8)
        mask = cv.inRange(img, color_arr, color_arr)

        gflag = cv.imread(f'flags/{full_flag}.png', cv.IMREAD_UNCHANGED)
        gflag = cv.resize(gflag, (img.shape[0], img.shape[1]))

        masked = cv.bitwise_and(gflag, gflag, mask=mask)

        # inverse binary threshold based on transparency >0
        _, negative = cv.threshold(masked[:,:,3], thresh=1, maxval=255, type=cv.THRESH_BINARY_INV)
        other = cv.bitwise_and(img, img, mask=negative)
        final = cv.bitwise_or(other, masked)

        # TODO: better antialiasing than this (make it less strict about replacing only the exact color in color_arr?)
        resized = cv.resize(final, (512, 512), interpolation=cv.INTER_AREA)

        return resized

    @commands.command()
    async def gaysper(self, ctx, flag_body, flag_outline=''):
        flags = [flag_body, flag_outline]
        fulls = []

        for flag in flags:
            full = aliases[flag] if flag in aliases else flag
            fulls.append(full)

            if full not in flagset and full != '':
                await ctx.send(f'Error: The flag `{flag}` was not recognized.')
                return

        if fulls[1] != '':
            output2 = '_' + fulls[1]
        else:
            output2 = ''
        output_path = f'output/gaysper_{fulls[0]}{output2}.png'

        if not pathlib.Path(output_path).exists(): # memoize gaysper generation
            img = cv.imread(f'images/gaysper.png', cv.IMREAD_UNCHANGED)
            replaced = await self.generic_replace(img, [51, 51, 51, 255], fulls[0])

            if fulls[1] != '': # replace outline also
                # TODO: doing this could cause problems if #2f2f2f is in the flag. also it's less efficient. probably best
                # to figure out a way to do the second loop within the generic_replace function
                replaced = await self.generic_replace(replaced, [47, 47, 47, 255], fulls[1])
            cv.imwrite(output_path, replaced)
        await ctx.send(file=discord.File(output_path))

def setup(bot):
    bot.add_cog(Replace(bot))
