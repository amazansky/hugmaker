import discord
from discord.ext import commands
from random import choice

from util import aliases, config, flagset

class Help(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    prefix = config['PREFIX']
    ops = config['BOT_OPS']

    # help commands
    @commands.group(invoke_without_command=True)
    async def help(self, ctx):
        em = discord.Embed(title='Help', description=f'Use {self.prefix}help `command` for more information about specific commands, replacing `command` with the name of a command below.', color=ctx.author.color)
        em.add_field(name='Emote commands', value='hug')
        em.add_field(name='Bot info', value='about, flags')
        await ctx.send(embed = em)

    @help.command()
    async def flags(self, ctx):
        abbr, full = choice(list(aliases.items()))
        em = discord.Embed(title='Pride flags', description=f'This is the full list of flags supported by the bot. Shortened names also work (e.g. \"{abbr}\" for \"{full}\").')
        em.add_field(name='Full list', value=', '.join(sorted(flagset)))
        await ctx.send(embed = em)

    @help.command()
    async def about(self, ctx):
        em = discord.Embed(title=f'About hugmaker ({config["BOT_VERSION"]})', description='Hi, I\'m hugmaker! I am a \
            Discord bot which creates custom emotes.')
        em.add_field(name='Source code', value='Hugmaker is open source! You can find the code at <https://github.com/amazansky/hugmaker>. \
            The code is available under the MIT license. In addition, any images generated by hugmaker are available under \
            CC BY 4.0. Have any issues or run into problems using the bot? Report them at the issue tracker: \
            <https://github.com/amazansky/hugmaker/issues>', inline=False)
        opstring = ", ".join(["**" + str(await self.bot.fetch_user(uid)) + "**" for uid in self.ops])
        em.add_field(name='Operators', value=f'On this server, the operators of hugmaker are: {opstring}. You can ask them \
            if you have any questions about the bot.', inline=False)
        await ctx.send(embed = em)

    @help.command()
    async def hug(self, ctx):
        em = discord.Embed(title='Hug', description='Sends a hug emote where the people are pride flags')
        em.add_field(name='Syntax', value=f'{self.prefix}hug `flag1` `flag2`')
        em.add_field(name='Parameters', value=f'`flag1` and `flag2` should be replaced by the names of pride flags. *(Run `{self.prefix}help flags` for a full list.)*')
        await ctx.send(embed = em)

def setup(bot):
    bot.add_cog(Help(bot))