import discord
from discord.ext import commands
import os
from random import choice

from util import config

if 'SENTRY_DSN' in config: # activate Sentry reporting if enabled
    from discord_sentry_reporting import use_sentry

    use_sentry(
        bot,
        dsn = config['SENTRY_DSN'],
        traces_sample_rate = 1.0,
        environment = config['ENV_NAME']
    )

bot = commands.Bot(command_prefix=config['PREFIX'])
bot.remove_command('help')

# load all cogs
for filename in os.listdir('cogs'):
    if filename.endswith('.py'):
        bot.load_extension(f'cogs.{filename[:-3]}')

@bot.command()
@commands.is_owner()
async def load(ctx, ext):
    bot.load_extension(f'cogs.{ext}')
    await ctx.message.add_reaction('\U00002705') # check mark emoji

@bot.command()
@commands.is_owner()
async def reload(ctx, ext):
    bot.reload_extension(f'cogs.{ext}')
    await ctx.message.add_reaction('\U00002705')

@bot.command()
@commands.is_owner()
async def unload(ctx, ext):
    bot.unload_extension(f'cogs.{ext}')
    await ctx.message.add_reaction('\U00002705')

bot.run(config['BOT_TOKEN'])
