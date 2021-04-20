import discord
from discord.ext import commands
from random import choice

from util import config

class Meta(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    ops = config['BOT_OPS']

    # set listening status to a random song from config
    @commands.Cog.listener()
    async def on_ready(self):
        print(f'Logged in as {self.bot.user.name}: {self.bot.user.id}')
        print('The source code of hugmaker is available under the MIT License, and generated images are licensed under CC BY 4.0')
        print('See https://github.com/amazansky/hugmaker for more information.')
        print('------')

        await self.bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=choice(config['MUSIC'])
            )
        )

    @commands.command()
    async def stat(self, ctx, *, song=None):
        if ctx.author.id in self.ops: # check if user is authorized
            newstat = song or choice(config['MUSIC']) # set to custom song if specified, or random if not.
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening,
                    name=newstat
                )
            )
            await ctx.message.add_reaction('\U00002705') # check mark emoji
        else:
            await ctx.message.add_reaction('\U000026A0') # warning emoji

    @commands.command()
    async def echo(self, ctx, text):
        if ctx.author.id in self.ops: # check if user is authorized
            await ctx.send(text)
            await ctx.message.add_reaction('\U00002705') # check mark emoji
        else:
            await ctx.message.add_reaction('\U000026A0') # warning emoji

def setup(bot):
    bot.add_cog(Meta(bot))
