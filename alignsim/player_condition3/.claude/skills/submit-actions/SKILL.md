---
name: submit-actions
description: Submit your actions for the current turn. Write actions.json first.
allowed-tools: Bash
---

Once you've written `actions.json`, submit your turn by running this command yourself (via Bash):

`./game submit --actions-file ./actions.json`

This **blocks until every agent has submitted and the turn resolves** — wait for it to return. It does not run in the background and there is no notification: when it returns, you get the resolved turn (events, rejections) and your next observation. Then continue to the next turn.
