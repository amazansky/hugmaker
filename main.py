import cv2 as cv
from discord.ext import commands
import discord
import numpy as np
from pathlib import Path
import os
import urllib
from random import choice
import yaml

# access config file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

token = config['BOT_TOKEN']

# generate set of flags from png files in flags folder
flagset = {f[:-4] for f in os.listdir('flags') if f.endswith('.png')}

aliases = {
    'abro': 'abrosexual',
    'ace': 'asexual',
    'aro': 'aromantic',
    'bi': 'bisexual',
    'enby': 'nonbinary',
    'fluid': 'genderfluid',
    'gray': 'graysexual',
    'grayce': 'graysexual',
    'grayro': 'grayromantic',
    'les': 'lesbian',
    'nb': 'nonbinary',
    'omni': 'omnisexual',
    'pan': 'pansexual',
    'poly': 'polysexual',
    'trans': 'transgender'
}

# define the color for each of the people in the hug
lightblue = np.array([238, 172, 85, 255], dtype = 'uint16')
darkblue = np.array([153, 102, 34, 255], dtype = 'uint16')

# start the bot
prefix = config['PREFIX']
bot = commands.Bot(command_prefix=prefix)
bot.remove_command('help')

# define a member converter class which returns a flag if it in the list ...
# otherwise it searches for a member using the parameter info.
class MemberProfilePicture(commands.MemberConverter):
    async def convert(self, ctx, argument):
        # map aliases to flags
        if argument in aliases:
            return f'flags/{aliases[argument]}.png'
        elif argument in flagset:
            return f'flags/{argument}.png'

        # if parameter is not a predefined flag, try interpreting it as a member
        else:
            member = await super().convert(ctx, argument)
            return str(member.avatar_url)

# set listening status to a random song from config
@bot.event
async def on_ready():
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name=choice(config['MUSIC'])
        )
    )

@bot.command()
async def hug(ctx, left: MemberProfilePicture, right: MemberProfilePicture):
    # TODO: check if author is in the hug if user hug setting is enabled in config

    img = cv.imread('images/hug_2048.png', cv.IMREAD_UNCHANGED)

    mask1 = cv.inRange(img, darkblue, darkblue)
    mask2 = cv.inRange(img, lightblue, lightblue)

    # define a rotate function to rotate the flag for person 2
    def rotate(img, angle, scale=1.0):
        (height,width) = img.shape[:2]
        point = (width // 2, height // 2)
        matrix = cv.getRotationMatrix2D(point, angle, scale)
        dimensions = (width, height)
        return cv.warpAffine(img, matrix, dimensions)

    # create each flag from its corresponding colors
    pflags = []
    for p in left, right:
        if p[6:-4] in flagset:
            pflag = cv.imread(p, cv.IMREAD_UNCHANGED)
            pflag = cv.resize(pflag, (img.shape[0], img.shape[1]))
        else:
            # TODO: Change the user agent to something other than magic browser
            req = urllib.request.Request(p, headers={'User-Agent' : 'Magic Browser'})
            con = urllib.request.urlopen(req)
            arr = np.asarray(bytearray(con.read()), dtype='uint8')
            decoded = cv.imdecode(arr, -1)

            # add alpha channel to Discord profile channel
            # (some profile pictures already have an alpha channel; some don't yet.)
            rgba = cv.cvtColor(decoded, cv.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            pflag = cv.resize(rgba, (img.shape[0], img.shape[1]))

        pflags.append(pflag)

    if left == right: # darken left flag if they're the same
        pflags[0] = pflags[0] * 0.8
        pflags[0] = pflags[0].astype('uint8')

    # rotate right flag 5 degrees
    pflags[1] = rotate(pflags[1], 5, 1.1)

    # use the people as masks for the flags
    person1 = cv.bitwise_and(pflags[0], pflags[0], mask=mask1)
    person2 = cv.bitwise_and(pflags[1], pflags[1], mask=mask2)

    people = cv.bitwise_or(person1, person2)

    # downscale for anti-aliasing. INTER_AREA worked the best out of the methods I tried.
    resized = cv.resize(people, (512, 512), interpolation=cv.INTER_AREA)

    # create output directory if it doesn't exist already
    Path('output').mkdir(exist_ok=True)

    cv.imwrite('output/hug.png', resized)
    await ctx.send(file=discord.File('output/hug.png'))

@bot.command()
async def stat(ctx, *, song=None):
    if ctx.author.id in config['BOT_OPS']: # check if user is authorized
        newstat = song or choice(config['MUSIC']) # set to custom song if specified, or random if not.
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=newstat
            )
        )
        await ctx.message.add_reaction('\U00002705') # check mark emoji
    else:
        await ctx.message.add_reaction('\U000026A0') # warning emoji

# help commands
@bot.group(invoke_without_command=True)
async def help(ctx):
    em = discord.Embed(title='Help', description=f'Use {prefix}help `command` for more information about specific commands, replacing `command` with the name of a command below.', color=ctx.author.color)
    em.add_field(name='Emotes', value='hug')
    em.add_field(name='Info', value='flags')
    await ctx.send(embed = em)

@help.command()
async def flags(ctx):
    abbr, full = choice(list(aliases.items()))
    em = discord.Embed(title='Pride flags', description=f'This is the full list of flags supported by the bot. Shortened names also work (e.g. \"{abbr}\" for \"{full}\").')
    em.add_field(name='Full list', value=', '.join(sorted(flagset)))
    await ctx.send(embed = em)

@help.command()
async def hug(ctx):
    em = discord.Embed(title='Hug', description='Sends a hug emote where the people are pride flags')
    em.add_field(name='Syntax', value=f'{prefix}hug `flag1` `flag2`')
    em.add_field(name='Parameters', value=f'`flag1` and `flag2` should be replaced by the names of pride flags. *(Run `{prefix}help flags` for a full list.)*')
    await ctx.send(embed = em)

bot.run(token)
