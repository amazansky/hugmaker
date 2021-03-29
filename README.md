# hugmaker
A Discord bot to make custom hug emotes with pride flag colors.

## Setup
1. Clone this repository to your computer and open a terminal to that folder
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate your virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `example.config.yml` to a new file `config.yml`. Change the value of `BOT_TOKEN` to your bot token from the Discord Developer Portal.
6. Run the script! `python main.py`

## Usage
The bot's current prefix is `$`. You can change this in the `main.py` file if you so desire.

To use the bot, type `$hug <flag1> <flag2>`, replacing `<hug1>` and `<hug2>` with any of the following currently supported flags:
- ace
- agender
- aro
- bi
- bigender
- enby
- gay
- genderfluid
- genderqueer
- lesbian
- omni
- pan
- poly
- pride
- trans

It's also pretty easy to add more flags within `main.py` if you'd like, as long as they are striped.

## Attributions
Images created by this program are derivatives of ["People Hugging" (1fac2)](https://abs.twimg.com/emoji/v2/svg/1fac2.svg) by [Twitter, Inc](https://twemoji.twitter.com/), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
