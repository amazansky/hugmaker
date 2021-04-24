import discord
from discord.ext import commands
from random import choice

from util import config

class Help(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    prefix = config['PREFIX']
    ops = config['BOT_OPS']

    # help commands
    @commands.group(invoke_without_command=True)
    async def help(self, ctx):
        em = discord.Embed(title='Help', description=f'Use {self.prefix}help `command` for more information about specific commands, replacing `command` with the name of a command below.', color=ctx.author.color)
        em.add_field(name='Emote commands', value='hug, make')
        em.add_field(name='Bot info', value='about, flag, flags')
        await ctx.send(embed=em)

    @help.command()
    async def flag(self, ctx):
        em = discord.Embed(title='Flag', description='Sends more information about a specific pride flag')
        em.add_field(name='Syntax', value=f'{self.prefix}flag `name`')
        em.add_field(name='Parameters', value=f'`name` should be replaced by the name of a pride flag.')
        await ctx.send(embed=em)

    @help.command()
    async def flags(self, ctx):
        em = discord.Embed(title='Flags', description='Sends a list of pride flags supported by the bot')
        em.add_field(name='Syntax', value=f'{self.prefix}flags')
        await ctx.send(embed=em)

    @help.command()
    async def about(self, ctx):
        em = discord.Embed(title='About', description='Sends information about the bot')
        em.add_field(name='Syntax', value=f'{self.prefix}about')
        await ctx.send(embed=em)

    @help.command()
    async def hug(self, ctx):
        em = discord.Embed(title='Hug', description='Sends a hug emote where the people are pride flags')
        em.add_field(name='Syntax', value=f'{self.prefix}hug `flag1` `flag2`')
        em.add_field(name='Parameters', value=f'`flag1` and `flag2` should be replaced by the names of pride flags. *(Run `{self.prefix}flags` for a full list.)*')
        await ctx.send(embed=em)

    @help.command()
    async def make(self, ctx):
        em = discord.Embed(title='Make', description='Creates a pride flag emote from a specified emoji')
        em.add_field(name='Syntax', value=f'{self.prefix}make `emoji` `flag`')
        em.add_field(name='Parameters', value=f'`emoji` should be replaced by an emoji character, and `flag` should be replaced by the name of a pride flag. *(Run `{self.prefix}flags` for a full list.)*')
        await ctx.send(embed=em)

def setup(bot):
    bot.add_cog(Help(bot))
