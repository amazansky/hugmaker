# hugmaker
A Discord bot to make custom hug and other emotes, either using colors from pride flags or profile pictures of server members.

## Setup
1. Clone this repository to your computer and open a terminal to that folder
1. Create and activate a virtual environment (highly recommended!): `python3 -m venv .venv`, then `source .venv/bin/activate`
1. Install dependencies: `pip install -r requirements.txt`
1. Copy `example.config.yml` to a new file `config.yml`. Within it,
    - Change the value of `BOT_TOKEN` to your bot token from the Discord Developer Portal
    - Also, customize the list of music to your heart's content
    - Add the Discord user IDs of your trusted bot operators
    - Set your preferred user hug requirement setting. `true` requires the user to appear in any hug in which there is at least one profile picture, whereas `false` does not enforce this requirement. The purpose of the requirement, if enabled, is to prevent users from making two users other than themselves hug.
1. Run the script! `python main.py`

## Usage
The bot's default prefix is `$`. You can change this in the config file if you so desire.

To use the bot, type `$hug <left> <right>`, replacing `<left>` and `<right>` with a **user mention** (or other string that resolves to a user class when passed into a [UserConverter](https://discordpy.readthedocs.io/en/latest/ext/commands/api.html#discord.ext.commands.UserConverter)) OR any of the following **currently supported flags**:
```
- abrosexual (alias: abro)
- achillean
- agender
- aromantic (alias: aro)
- asexual (alias: ace)
- bigender
- bisexual (alias: bi)
- demiboy
- demigirl
- demiromantic (alias: demiro)
- demisexual (alias: demi)
- gay
- genderfluid (alias: fluid)
- genderflux (alias: flux)
- genderqueer
- graysexual (alias: gray, grayce)
- grayromantic (alias: grayro)
- intersex
- lesbian (alias: les)
- nonbinary (aliases: enby, nb)
- omnisexual (alias: omni)
- pansexual (alias: pan)
- pangender
- polysexual (alias: poly)
- pride
- sapphic
- toric
- transgender (alias: trans)
- trixic
```
If there is at least one user mention/other user string in the hug, the bot will allow or deny user hugs based on your `REQUIRE_USER_HUG` config setting described above.

It's also pretty easy to add more flags to the bot by dropping a square png image of the flag into the `flags/` directory. If you want to define an alias, do that within `util.py`. The bot will take care of the rest!

## Licenses
Hug emotes created by hugmaker are derivatives of ["People Hugging" (1fac2)](https://abs.twimg.com/emoji/v2/svg/1fac2.svg) by [Twitter, Inc](https://twemoji.twitter.com/), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Emoji-generated emotes created by hugmaker are derivatives of [Twemoji](https://twemoji.twitter.com/) emotes by [Twitter, Inc](https://twemoji.twitter.com/), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The source code of hugmaker is licensed under MIT. Images created by hugmaker are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Alex Mazansky.
