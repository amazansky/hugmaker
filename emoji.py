import yaml
from discord.ext import commands
import cv2 as cv
import numpy as np
import cairosvg
import discord
import os
import pathlib
import urllib

# access config file
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

prefix = config['PREFIX']
token = config['BOT_TOKEN']

async def findmost(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    unr = np.unravel_index(await count(a1D), col_range)
    return np.array(unr)

# TODO: optimize this further?
async def count(a):
    results = {}
    for x in a:
        if x not in results:
            results[x] = 1
        else:
            results[x] += 1
    
    del results[0]
    return max(results, key=lambda x: results[x])

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

# start the bot
bot = commands.Bot(command_prefix=prefix)

@bot.command()
async def add(ctx, e):
    if ctx.author.id in config['BOT_OPS']: # check if user is authorized
        try:
            unicode = f'{ord(e):x}'
            # unicode = e
        except TypeError: # paramater was not a single character
            await ctx.send(f'Error: You must use `{prefix}add` with an emoji character.')
            return

        req = urllib.request.Request(f'https://twemoji.maxcdn.com/v/latest/svg/{unicode}.svg', headers={'User-Agent' : 'Magic Browser'})
        con = urllib.request.urlopen(req)

        cairosvg.svg2svg(bytestring=bytes(con.read()), write_to=f'emoji/{unicode}.svg')

        await ctx.message.add_reaction('\U00002705') # check mark emoji
    else:
        await ctx.message.add_reaction('\U000026A0') # warning emoji

@bot.command()
async def rm(ctx, e):
    if ctx.author.id in config['BOT_OPS']: # check if user is authorized
        try:
            unicode = f'{ord(e):x}'
            # unicode = e
        except TypeError: # paramater was not a single character
            await ctx.send(f'Error: You must use `{prefix}make` with an emoji character.')
            return

        for filename in f'emoji/{unicode}.svg', f'emoji/{unicode}.png':
            try:
                os.remove(filename)
            except OSError:
                pass

        await ctx.message.add_reaction('\U00002705') # check mark emoji
    else:
        await ctx.message.add_reaction('\U000026A0') # warning emoji

@bot.command()
async def make(ctx, e, flag, *, options=''):
    # send notice about any unrecognized options. currently only blur is recognized.
    options = options.split()
    unrecog = ['`' + o + '`' for o in options if o != 'blur']
    if unrecog:
        await ctx.send(f'Warning: Unrecognized option(s): {", ".join(unrecog)}.')

    try:
        unicode = f'{ord(e):x}'
        # unicode = e
    except TypeError: # paramater was not a single character
        await ctx.send(f'Error: You must use `{prefix}make` with an emoji character.')
        return

    epath = f'emoji/{unicode}.png'

    if pathlib.Path(epath).exists():
        emoji = cv.imread(epath, cv.IMREAD_UNCHANGED)
    else: # file is not converted to png yet. convert and try again.
        try:
            cairosvg.svg2png(url=f'emoji/{unicode}.svg', output_width=2048, output_height=2048, write_to=f'emoji/{unicode}.png')
            emoji = cv.imread(epath, cv.IMREAD_UNCHANGED)
        except URLError:
            await ctx.send(f'Error: The emoji you used is either unrecognized or has not been added to hugmaker at this time.')
            return

    most = await findmost(emoji)
    emoji_mask = cv.inRange(emoji, most, most)

    if flag in aliases:
        p = f'flags/{aliases[flag]}.png'
    elif flag in flagset:
        p = f'flags/{flag}.png'
    else:
        await ctx.send(f'Error: One or more of the flags you entered is not currently supported.')
        return

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

# Test command: converts and outputs every svg in the folder as a test
@bot.command()
async def t(ctx):
    emojis = {f[:-4] for f in os.listdir('emoji') if f.endswith('.svg')}

    for e in emojis:
        # print(e)
        await make(ctx, e, 'pride')

bot.run(token)
