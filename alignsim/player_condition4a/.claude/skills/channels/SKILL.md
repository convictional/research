---
name: channels
description: List team channels or create a new public channel
allowed-tools: Bash
---

Channels organise chat by topic, the way a Slack workspace does. All channels are public —
everyone can read and post to any channel. Keep each channel focused on one topic so its
discussion stays findable, post where a message belongs (a customer's channel, a decision
thread, a function's channel), and fall back to `everyone` for broad or cross-cutting updates.

List the current channels:

!`./game chat channels`

Create a new public channel whenever a topic deserves its own thread — a specific customer, a
decision the team needs to settle, a workstream, or a function that stands up later. A focused
channel keeps that conversation together and easy to find later. Run this command yourself (via
Bash) with your channel name in place of the placeholder. Names are lowercase slugs (letters,
digits, `-`, `_`):

`./game channel create <name>`
