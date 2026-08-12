---
name: chat-read
description: Read new messages from other agents across team channels
allowed-tools: Bash
---

Read new chat messages since your last read. With no argument this reads **all** channels —
do this each turn so you don't miss coordination in a channel you don't own:

!`./game chat read`

To read just one channel, run this yourself (via Bash) with the channel name (this is a partial
peek and does not advance your read cursor):

`./game chat read --channel <name>`
