import cv2 as cv
import discord
from discord.ext import commands
import numpy as np
import os
import pathlib
from random import choice
import urllib
import yaml

# access config file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

prefix = config['PREFIX']
token = config['BOT_TOKEN']
userhug = config['REQUIRE_USER_HUG'] # checks if user hug requirement is enabled

# generate set of flags from png files in flags folder
flagset = {f[:-4] for f in os.listdir('flags') if f.endswith('.png')}

aliases = {
    'abro': 'abrosexual',
    'ace': 'asexual',
    'aro': 'aromantic',
    'bi': 'bisexual',
    'demi': 'demisexual',
    'demiro': 'demiromantic',
    'enby': 'nonbinary',
    'fluid': 'genderfluid',
    'flux': 'genderflux',
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
bot = commands.Bot(command_prefix=prefix)
bot.remove_command('help')

if config['SENTRY_DSN']: # activate Sentry reporting if enabled
    from discord_sentry_reporting import use_sentry

    use_sentry(
        bot,
        dsn = config['SENTRY_DSN'],
        traces_sample_rate = 1.0,
        environment = config['ENV_NAME']
    )

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
    # check if author is in the hug OR if user hug requirement is disabled in config
    # TODO: simplify the if statement
    if 'self' in (left[1], right[1]) or (left[1] == 'flag' and right[1] == 'flag') or userhug == False: # user is authorized to make this hug
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
        for p in left[0], right[0]:
            if p[6:-4] in flagset:
                pflag = cv.imread(p, cv.IMREAD_UNCHANGED)
                pflag = cv.resize(pflag, (img.shape[0], img.shape[1]))
            else:
                # TODO: Change the user agent to something other than magic browser
                req = urllib.request.Request(p, headers={'User-Agent' : 'Magic Browser'})
                con = urllib.request.urlopen(req)
                arr = np.asarray(bytearray(con.read()), dtype='uint8')
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

        # create output directory if it doesn't exist already
        pathlib.Path('output').mkdir(exist_ok=True)

        cv.imwrite('output/hug.png', resized)
        await ctx.send(file=discord.File('output/hug.png'))

    else: # user is not authorized to make this hug
        msg = 'To prevent abuse of this feature, you must include yourself in any hug containing at least one profile picture.'
        if ctx.author.id in config['BOT_OPS']:
            msg += ' As a bot operator, you may change this by setting `REQUIRE_USER_HUG` to `false` in `config.yml`.'
        await ctx.send(msg)


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

@bot.command()
async def echo(ctx, text):
    if ctx.author.id in config['BOT_OPS']: # check if user is authorized
        await ctx.send(text)
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
