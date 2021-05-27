import discord
from discord.ext import commands

from util import config

class Help(commands.Cog):
    prefix = config['PREFIX']
    ops = config['BOT_OPS']

    # help commands
    @commands.group(invoke_without_command=True)
    async def help(self, ctx):
        em = discord.Embed(title='Hugmaker Help', description=f'Use {self.prefix}help `command` for more information about specific commands, replacing `command` with the name of a command below.', color=ctx.author.color)
        em.add_field(name='Emote commands', value='gaysper, hug, make')
        em.add_field(name='Bot info', value='about, flag, flags')
        em.add_field(name='Operator commands', value='echo, stat')
        await ctx.send(embed=em)

    # BOT INFO COMMANDS
    @help.command()
    async def about(self, ctx):
        em = discord.Embed(title='Command: About', description='Sends information about the bot')
        em.add_field(name='Syntax', value=f'{self.prefix}about')
        await ctx.send(embed=em)

    @help.command()
    async def flag(self, ctx):
        em = discord.Embed(title='Command: Flag', description='Sends more information about a specific pride flag')
        em.add_field(name='Syntax', value=f'{self.prefix}flag `name`')
        em.add_field(name='Parameters', value=f'`name` should be replaced by the name of a pride flag.')
        await ctx.send(embed=em)

    @help.command()
    async def flags(self, ctx):
        em = discord.Embed(title='Command: Flags', description='Sends a list of pride flags supported by the bot')
        em.add_field(name='Syntax', value=f'{self.prefix}flags')
        await ctx.send(embed=em)

    # EMOTE COMMANDS
    @help.command()
    async def gaysper(self, ctx):
        em = discord.Embed(title='Command: Flag', description='Sends a pride flag Gaysper (ghost) emote')
        em.add_field(name='Syntax', value=f'{self.prefix}gaysper `body` `[outline]`')
        em.add_field(name='Parameters', value=f'`body` should be replaced by the name of a pride flag. Optionally, `[outline]` can also be replaced with the name of a pride flag. *(Run `{self.prefix}flags` for a full list.)*')
        await ctx.send(embed=em)

    @help.command()
    async def hug(self, ctx):
        em = discord.Embed(title='Command: Hug', description='Sends a hug emote where the people are pride flags')
        em.add_field(name='Syntax', value=f'{self.prefix}hug `left` `right`')
        em.add_field(name='Parameters', value=f'`left` and `right` should be replaced by the names of pride flags. *(Run `{self.prefix}flags` for a full list.)*')
        await ctx.send(embed=em)

    @help.command()
    async def make(self, ctx):
        em = discord.Embed(title='Command: Make', description='Creates a pride flag emote from a specified emoji')
        em.add_field(name='Syntax', value=f'{self.prefix}make `emoji` `flag` `[options]`')
        em.add_field(name='Parameters', value=f'`emoji` should be replaced by an emoji character.\n`flag` should be replaced by the name of a pride flag. *(Run `{self.prefix}flags` for a full list.)*', inline=False)
        em.add_field(
            name='Options',
            value='Replace `[options]` with any number of the following to modify the behavior of hugmaker\'s emoji generation:\n- `blur` to blur the flag\n- `inv` to generate an inverse version of the emoji',
            inline=False
        )
        await ctx.send(embed=em)

    # OPERATOR COMMANDS
    # TODO: Add load, reload, unload
    @help.command()
    async def echo(self, ctx):
        em = discord.Embed(title='Command: Echo', description='**[Operator only]** Echoes the string you pass in.')
        em.add_field(name='Syntax', value=f'{self.prefix}echo `message`')
        em.add_field(name='Parameters', value=f'`message` should be replaced by the message to echo. If it lasts multiple lines, use double quotes at the beginning and end of the message.', inline=False)
        await ctx.send(embed=em)

    @help.command()
    async def stat(self, ctx):
        em = discord.Embed(title='Command: Stat', description='**[Operator only]** Sets the bot\'s status')
        em.add_field(name='Syntax', value=f'{self.prefix}stat `status`')
        em.add_field(name='Parameters', value='`status` should be replaced by the new status. It will be set in the form "Listening to **`status`**".', inline=False)
        await ctx.send(embed=em)

def setup(bot):
    bot.add_cog(Help(bot))
