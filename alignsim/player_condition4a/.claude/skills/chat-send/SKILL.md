---
name: chat-send
description: Send a message to other agents in a team channel
allowed-tools: Bash
---

Post a message by running this command yourself (via Bash). Pass the channel with `--channel`;
omit it to post to the `everyone` channel. Pick the channel that best scopes your message, and
put your real message in place of the placeholder:

`./game chat send --channel <name> "<your message>"`

Post to the team-wide channel:

`./game chat send "<your message>"`
