import discord
from discord.ext import commands
import cairosvg
import cv2 as cv
import numpy as np
import os
import pathlib
import urllib

from util import aliases, config, flagset

class Emoji(commands.Cog):
    memo = {}

    async def findmost(self, em, unicode_key): # find the color that appears most in the emoji
        a = cv.resize(em, (128, 128)) # scale down emoji for finding color

        a2D = a.reshape(-1,a.shape[-1])
        col_range = (256, 256, 256, 256)
        a1D = np.ravel_multi_index(a2D.T, col_range)

        results = {}
        for x in a1D:
            if x not in results:
                results[x] = 1
            else:
                results[x] += 1

        del results[0] # remove transparent pixels from most frequent color list
        a_max = max(results, key=lambda x: results[x])

        unr = np.unravel_index(a_max, col_range)

        return np.array(unr)

    @commands.command()
    async def make(self, ctx, e, flag, *, options=''):
        # send notice about any unrecognized options. currently only blur is recognized.
        options = options.split()
        unrecog = ['`' + o + '`' for o in options if o != 'blur']
        if unrecog:
            await ctx.send(f'Warning: Unrecognized option(s): {", ".join(unrecog)}.')

        e = e.encode('utf-8')
        e = e[:-3] if e.endswith(b'\xef\xb8\x8f') else e # remove variation selector 16 if present
        e = e.decode('utf-8')

        # TODO: download/cache svg files
        unicode = '-'.join([f'{ord(utf):x}' for utf in e])
        req = urllib.request.Request(f'https://twemoji.maxcdn.com/v/latest/svg/{unicode}.svg', headers={'User-Agent' : 'Magic Browser'})

        try:
            con = urllib.request.urlopen(req)
        except urllib.error.HTTPError:
            await ctx.send('Error: The character you used is not a recognized emoji.')
            return

        imgbytes = cairosvg.svg2png(bytestring=bytes(con.read()), output_width=2048, output_height=2048)

        nparr = np.frombuffer(imgbytes, np.uint8)
        emoji = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)

        if flag in aliases:
            p = f'flags/{aliases[flag]}.png'
        elif flag in flagset:
            p = f'flags/{flag}.png'
        else:
            await ctx.send(f'Error: One or more of the flags you entered is not currently supported.')
            return

        most = await self.findmost(emoji, unicode)
        emoji_mask = cv.inRange(emoji, most, most)

        # create flag from its corresponding colors
        eflag = cv.imread(p, cv.IMREAD_UNCHANGED)
        eflag[:, :, 3] = 255

        if 'blur' in options:
            eflag = cv.blur(eflag, (500, 500))

        eflag = cv.resize(eflag, (emoji.shape[0], emoji.shape[1]))

        # add flag mask to the emoji
        masked = cv.bitwise_and(eflag, eflag, mask=emoji_mask)

        # inverse binary threshold based on transparency >0
        _, negative = cv.threshold(masked[:,:,3], thresh=1, maxval=255, type=cv.THRESH_BINARY_INV)

        other = cv.bitwise_and(emoji, emoji, mask=negative)

        final = cv.bitwise_or(other, masked)
        cv.imwrite('output/emoji.png', final)

        resized = cv.resize(final, (512, 512), interpolation=cv.INTER_AREA)

        # create output directory if it doesn't exist already
        pathlib.Path('output').mkdir(exist_ok=True)

        cv.imwrite('output/emoji.png', resized)
        await ctx.send(file=discord.File('output/emoji.png'))

def setup(bot):
    bot.add_cog(Emoji(bot))
