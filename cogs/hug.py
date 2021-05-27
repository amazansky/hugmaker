import discord
from discord.ext import commands
import cv2 as cv
import numpy as np
import urllib

from util import aliases, config, flagset, rotate

class Hug(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    ops = config['BOT_OPS']
    userhug = config['REQUIRE_USER_HUG'] # checks if user hug requirement is enabled

    # define the color for each of the people in the hug
    lightblue = np.array([238, 172, 85, 255], dtype = np.uint8)
    darkblue = np.array([153, 102, 34, 255], dtype = np.uint8)

    # define a member converter class which returns a flag if it in the list ...
    # otherwise it searches for a member using the parameter info.
    class MemberProfilePicture(commands.MemberConverter):
        async def convert(self, ctx, argument):
            # map aliases to flags

            # the second value in the return tuple represents whether the user is authorized to make
            # that hug. the user must be authorized for at least one of the two "people" in the hug
            # for the bot to send it.

            if argument in aliases:
                return (f'flags/{aliases[argument]}.png', 'flag')
            elif argument in flagset:
                return (f'flags/{argument}.png', 'flag')

            # if parameter is not a predefined flag, try interpreting it as a member
            else:
                member = await super().convert(ctx, argument)
                authorized = 'self' if member.id == ctx.author.id else 'other'
                return (str(member.avatar_url), authorized)

    @commands.command()
    async def hug(self, ctx, left: MemberProfilePicture, right: MemberProfilePicture):
        # check if author is in the hug OR if user hug requirement is disabled in config
        # TODO: simplify the if statement
        if 'self' in (left[1], right[1]) or (left[1] == 'flag' and right[1] == 'flag') or self.userhug == False: # user is authorized to make this hug
            img = cv.imread('images/hug_2048.png', cv.IMREAD_UNCHANGED)

            mask1 = cv.inRange(img, self.darkblue, self.darkblue)
            mask2 = cv.inRange(img, self.lightblue, self.lightblue)

            pflags = []
            for p in left[0], right[0]:
                if p[6:-4] in flagset: # image is a locally stored flag
                    pflag = cv.imread(p, cv.IMREAD_UNCHANGED)
                    pflag = cv.resize(pflag, (img.shape[0], img.shape[1]))
                else: # image is a Discord profile picture; fetch from online
                    # TODO: Change the user agent to something other than magic browser
                    req = urllib.request.Request(p, headers={'User-Agent' : 'Magic Browser'})
                    con = urllib.request.urlopen(req)
                    arr = np.asarray(bytearray(con.read()), dtype=np.uint8)
                    decoded = cv.imdecode(arr, -1)

                    # add alpha channel to Discord profile picture
                    # (some profile pictures already have an alpha channel; some don't yet.)
                    rgba = cv.cvtColor(decoded, cv.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = 255
                    pflag = cv.resize(rgba, (img.shape[0], img.shape[1]))

                pflags.append(pflag)

            if left[0] == right[0]: # darken left flag if they're the same
                pflags[0] = pflags[0] * 0.8
                pflags[0] = pflags[0].astype('uint8')

            # rotate right flag 5 degrees as long as it isn't a user profile picture
            if right[1] != 'self' and right[1] != 'other':
                pflags[1] = rotate(pflags[1], 5, 1.1)

            # use the people as masks for the flags
            person1 = cv.bitwise_and(pflags[0], pflags[0], mask=mask1)
            person2 = cv.bitwise_and(pflags[1], pflags[1], mask=mask2)

            people = cv.bitwise_or(person1, person2)

            # downscale for anti-aliasing. INTER_AREA worked the best out of the methods I tried.
            resized = cv.resize(people, (512, 512), interpolation=cv.INTER_AREA)

            cv.imwrite('output/hug.png', resized)
            await ctx.send(file=discord.File('output/hug.png'))

        else: # user is not authorized to make this hug
            msg = 'To prevent abuse of this feature, you must include yourself in any hug containing at least one profile picture.'
            if ctx.author.id in self.ops:
                msg += ' As a bot operator, you may change this by setting `REQUIRE_USER_HUG` to `false` in `config.yml`.'
            await ctx.send(msg)

def setup(bot):
    bot.add_cog(Hug(bot))
