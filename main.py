import cv2 as cv
from discord.ext import commands
import discord
import numpy as np
from pathlib import Path
import os
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
bot = commands.Bot(command_prefix=config['PREFIX'])
bot.remove_command('help')

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
async def hug(ctx, p1, p2):
    # map aliases to flags
    p1 = aliases[p1] if p1 in aliases else p1
    p2 = aliases[p2] if p2 in aliases else p2

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
    if p1 in flagset:
        flag1 = cv.imread(f'flags/{p1}.png', cv.IMREAD_UNCHANGED)
        flag1 = cv.resize(flag1, (img.shape[0], img.shape[1]))

        if p1 == p2:
            flag1 = flag1 * 0.8
            flag1 = flag1.astype('uint8')

    if p2 in flagset:
        flag2 = cv.imread(f'flags/{p2}.png', cv.IMREAD_UNCHANGED)
        flag2 = cv.resize(flag2, (img.shape[0], img.shape[1]))
    flag2 = rotate(flag2, 5, 1.1)

    # use the people as masks for the flags
    person1 = cv.bitwise_and(flag1, flag1, mask=mask1)
    person2 = cv.bitwise_and(flag2, flag2, mask=mask2)

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
    em = discord.Embed(title='Help', description='Use $help `command` for more information about specific commands, replacing `command` with the name of a command below.', color=ctx.author.color)
    em.add_field(name='Emotes', value='hug')
    em.add_field(name='Info', value='flags')
    await ctx.send(embed = em)

@help.command()
async def flags(ctx):
    abbr, full = choice(list(aliases.items()))
    em = discord.Embed(title='Pride flags', description='This is the full list of flags supported by the bot. Shortened names also work (e.g. \"%s\" for \"%s\").' % (abbr, full))
    em.add_field(name='Full list', value=', '.join(flagdict))
    await ctx.send(embed = em)

@help.command()
async def hug(ctx):
    em = discord.Embed(title='Hug', description='Sends a hug emote where the people are pride flags')
    em.add_field(name='Syntax', value='$hug `flag1` `flag2`')
    em.add_field(name='Parameters', value='`flag1` and `flag2` should be replaced by the names of pride flags. *(Run `$help flags` for a full list.)*')
    await ctx.send(embed = em)

bot.run(token)
