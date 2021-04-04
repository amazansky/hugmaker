import cv2 as cv
from discord.ext import commands
import discord
import numpy as np
from random import choice
import yaml

# access config file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

token = config['BOT_TOKEN']

# flag colors use BGR (not RGB!) for consistency with OpenCV
flagdict = {
    'abrosexual': [(149, 202, 119), (202, 228, 181), (255, 255, 255), (181, 150, 230), (110, 70, 216)],
    'asexual': [(0, 0, 0), (163, 163, 163), (255, 255, 255), (128, 0, 128)],
    'agender': [(0, 0, 0), (185, 185, 185), (255, 255, 255), (131, 244, 184), (255, 255, 255), (185, 185, 185), (0, 0, 0)],
    'aromantic': [(66, 165, 61), (121, 211, 167), (255, 255, 255), (169, 169, 169), (0, 0, 0)],
    'bigender': [(157, 122, 193), (200, 165, 234), (229, 198, 212), (255, 255, 255), (229, 198, 212), (230, 198, 155), (203, 130, 107)],
    'bisexual': [(112, 2, 214), (112, 2, 214), (150, 79, 155), (168, 56, 0), (168, 56, 0)],
    'demiboy': [(127, 127, 127), (195, 195, 195), (233, 217, 154), (255, 255, 255), (233, 217, 154), (195, 195, 195), (127, 127, 127)],
    'demigirl': [(127, 127, 127), (195, 195, 195), (200, 175, 254), (255, 255, 255), (200, 175, 254), (195, 195, 195), (127, 127, 127)],
    'gay': [(113, 141, 19), (171, 206, 45), (195, 232, 153), (255, 255, 255), (223, 173, 124), (199, 73, 79), (119, 28, 61)],
    'genderfluid': [(162, 117, 255), (255, 255, 255), (214, 24, 190), (0, 0, 0), (189, 62, 51)],
    'genderqueer': [(220, 126, 181), (255, 255, 255), (35, 129, 74)],
    'graysexual': [(143, 12, 115), (169, 176, 173), (255, 255, 255), (169, 176, 173), (143, 12, 115)],
    'grayromantic': [(37, 125, 17), (175, 178, 176), (255, 255, 255), (175, 178, 176), (37, 125, 17)],
    'lesbian': [(0, 45, 213), (86, 154, 255), (255, 255, 255), (164, 98, 211), (83, 2, 138)],
    'nonbinary': [(48, 244, 255), (255, 255, 255), (209, 89, 156), (0, 0, 0)],
    'omnisexual': [(201, 153, 253), (185, 85, 254), (69, 3, 40), (248, 94, 100), (250, 163, 138)],
    'pansexual': [(140, 33, 255), (0, 216, 255), (255, 177, 33)],
    'pangender': [(159, 247, 255), (206, 221, 255), (250, 235, 255), (255, 255, 255), (250, 235, 255), (206, 221, 255), (159, 247, 255)],
    'polysexual': [(185, 28, 246), (105, 213, 7), (246, 146, 28)],
    'pride': [(3, 3, 228), (0, 140, 255), (0, 237, 255), (38, 128, 0), (255, 77, 0), (135, 7, 117)],
    'transgender': [(250, 206, 91), (184, 169, 245), (255, 255, 255), (184, 169, 245), (250, 206, 91)]
}

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
    flag1 = np.zeros((img.shape[0], img.shape[1], 4), dtype='uint8')
    colors1 = flagdict[p1]
    for i, color in enumerate(colors1):
        # darken left person if the flags are the same
        if p1 == p2:
            color = [int(c*0.8) for c in color]
        else:
            color = list(color)
        color.append(255)
        color = tuple(color)
        cv.rectangle(flag1, (0, i*img.shape[1] // len(colors1)), (img.shape[1], (i+1) * img.shape[1] // len(colors1)), color, -1)

    flag2 = np.zeros((img.shape[0], img.shape[1], 4), dtype='uint8')
    colors2 = flagdict[p2]
    for i, color in enumerate(colors2):
        color = list(color)
        color.append(255)
        color = tuple(color)
        cv.rectangle(flag2, (0, i*img.shape[1] // len(colors2)), (img.shape[1], (i+1) * img.shape[1] // len(colors2)), color, -1)
    flag2 = rotate(flag2, 5, 1.1)

    # use the people as masks for the flags
    person1 = cv.bitwise_and(flag1, flag1, mask=mask1)
    person2 = cv.bitwise_and(flag2, flag2, mask=mask2)

    people = cv.bitwise_or(person1, person2)

    # downscale for anti-aliasing. INTER_AREA worked the best out of the methods I tried.
    resized = cv.resize(people, (512, 512), interpolation=cv.INTER_AREA)

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
