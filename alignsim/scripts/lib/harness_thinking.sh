#!/usr/bin/env bash
# harness_thinking.sh — shared reasoning-effort / thinking resolution for the AlignSim run scripts.
#
# Sourced by sandbox_run_condition{2,3,4}.sh so the three conditions resolve effort IDENTICALLY.
# Drift here is a research-integrity bug: the collaboration substrate is the only treatment, so
# reasoning effort must be configured the same way across every condition.
#
# ONE operator scale, mapped onto each harness's (different) mechanism:
#     off | minimal | low | medium | high | xhigh          (default: high)
#
#   - Pi           : the `--thinking <level>` CLI flag. models.json `reasoning:false` HARD-GATES
#                    non-reasoning models, so a level handed to Gemma is a harmless no-op.
#   - Claude Code  : `.claude/settings.json` — `effortLevel` (enum low|medium|high|xhigh) for models
#                    with the `effort` capability, else the `alwaysThinkingEnabled` on/off gate.
#
# "thinking" (Pi) and "effort" (Claude) are NOT the same mechanism; we map the shared ordinal scale
# to each as faithfully as the harness allows. Reasoning is ON by default for every model that can
# reason; only genuinely non-reasoning models (Gemma) stay off.

[ -n "${_HT_SOURCED:-}" ] && return 0
_HT_SOURCED=1

HT_VALID_LEVELS="off minimal low medium high xhigh"
HT_DEFAULT_LEVEL="high"

# ht_resolve_level [requested] -> effective level (default high when empty). Validates; fails (rc 1)
# on an unknown level so the run aborts under `set -e` instead of launching with a bad config.
ht_resolve_level() {
  local requested="${1:-}"
  local level="${requested:-$HT_DEFAULT_LEVEL}"
  case " $HT_VALID_LEVELS " in
    *" $level "*) printf '%s' "$level" ;;
    *) echo "harness_thinking: invalid effort/thinking level '$level' (valid: $HT_VALID_LEVELS)" >&2
       return 1 ;;
  esac
}

# ht_pi_thinking_flag <level> -> the Pi `--thinking <level>` flag (always emitted; Gemma is
# hard-gated by reasoning:false so the level is a no-op there).
ht_pi_thinking_flag() {
  printf -- '--thinking %s' "$1"
}

# ht_claude_settings_object <level> <model> -> a one-line JSON object to set in .claude/settings.json.
# Capability-aware (Claude Code 2.1.207 semantics):
#   - level "off"                          -> {"alwaysThinkingEnabled": false}   (reasoning disabled)
#   - Haiku (no `effort` capability)       -> {"alwaysThinkingEnabled": true}    (effortLevel is a no-op)
#   - effort-capable (Sonnet 4.6, Opus     -> {"effortLevel": "<mapped>"}        (omit alwaysThinkingEnabled;
#     4.7/4.8, and modern models default)                                          absent => adaptive reasoning ON)
# Sonnet 4.6 has no xhigh effort -> clamped to high.
ht_claude_settings_object() {
  local level="$1" model="$2"

  if [ "$level" = "off" ]; then
    printf '{"alwaysThinkingEnabled": false}'
    return
  fi

  case "$model" in
    *haiku*)
      printf '{"alwaysThinkingEnabled": true}'
      ;;
    *)
      local effort
      case "$level" in
        minimal|low) effort="low" ;;
        medium)      effort="medium" ;;
        high)        effort="high" ;;
        xhigh)       effort="xhigh" ;;
        *)           effort="high" ;;
      esac
      if [[ "$model" == *sonnet-4-6* && "$effort" == "xhigh" ]]; then
        echo "harness_thinking: sonnet-4-6 has no xhigh effort; clamping to high" >&2
        effort="high"
      fi
      printf '{"effortLevel": "%s"}' "$effort"
      ;;
  esac
}

# ── Hooks format: flat (template) -> grouped (both harnesses require it) ─────
# Player templates ship hooks in the FLAT shape {type, command, matcher}; BOTH Claude Code AND the
# pi claude-hooks-bridge require the GROUPED shape {matcher, hooks: [{type, command}]}. This is the
# single definition of that transform, applied to a hooks OBJECT ({PreToolUse:[...], PostToolUse:[...]}).
# Idempotent: entries already grouped (they carry a "hooks" array) pass through untouched, so applying
# it twice is safe. One definition, used by both paths — divergence here would be a research-integrity bug.
_HT_HOOKS_REGROUP='with_entries(.value |= map(if (type=="object" and has("hooks")) then . else {matcher: .matcher, hooks: [{type: .type, command: .command}]} end))'

# Claude: regroup .hooks (if present) THEN shallow-merge the effort object ($add). Preserves every
# other top-level key (env, etc.). Built by concatenation so $add stays literal (no escaping).
_HT_CLAUDE_SETTINGS_JQ='(if (.hooks|type)=="object" then .hooks |= ('"$_HT_HOOKS_REGROUP"') else . end) | . + $add'
# Pi bridge: emit ONLY the regrouped hooks object (matches the long-standing pi reshape — it drops
# non-hook keys, which the bridge ignores anyway; kept byte-identical to avoid disturbing that path).
_HT_PI_HOOKS_JQ='{hooks: (.hooks | '"$_HT_HOOKS_REGROUP"')}'

# ht_apply_claude_settings <ssh_cmd> <settings_path> <level> <model>
# Regroup the template's flat hooks to the grouped shape Claude Code requires AND merge the
# capability-aware effort object into an EXISTING .claude/settings.json, preserving everything else it
# holds (hooks, env). Used identically by every condition so effort + hook shape are applied the same
# way everywhere. <ssh_cmd> is the (word-split) sandbox SSH prefix.
ht_apply_claude_settings() {
  local ssh_cmd="$1" settings_path="$2" level="$3" model="$4"
  local obj; obj="$(ht_claude_settings_object "$level" "$model")"
  # The effort object is piped in on stdin ($(cat) reads it into --argjson). The settings path is
  # EMBEDDED in the remote command (not passed as an argv value) so the sandbox shell expands its
  # leading $HOME — $GAME_DIR/$GAME_BASE are literal "$HOME/..." strings resolved sandbox-side. The jq
  # program is single-quoted for the remote so its $add reaches jq's --argjson var unexpanded.
  printf '%s' "$obj" | $ssh_cmd bash -lc \
    "jq --argjson add \"\$(cat)\" '$_HT_CLAUDE_SETTINGS_JQ' $settings_path > $settings_path.tmp && mv $settings_path.tmp $settings_path"
}

# ht_apply_pi_hooks <ssh_cmd> <settings_path>
# Regroup the template's flat hooks in a pi agent's .claude/settings.json to the grouped shape the
# pi claude-hooks-bridge requires. Same transform as the Claude path (shared _HT_HOOKS_REGROUP).
ht_apply_pi_hooks() {
  local ssh_cmd="$1" settings_path="$2"
  $ssh_cmd bash -lc \
    "jq '$_HT_PI_HOOKS_JQ' $settings_path > $settings_path.tmp && mv $settings_path.tmp $settings_path"
}

# ht_pi_models_json <provider> <model_id> <api_key> -> the full ~/.pi/agent/models.json (via jq, host-side).
# Reasoning is ON for every reasoning-capable model:
#   - anthropic     : override-only config. We DON'T ship a custom model entry (that fully replaces Pi's
#                     built-in and drops its compat.forceAdaptiveThinking -> legacy thinking field ->
#                     opus-4-8 HTTP 400). Instead a `modelOverrides` entry DEEP-MERGES our contextWindow/
#                     maxTokens onto the built-in, preserving its reasoning + adaptive-thinking compat.
#   - openrouter    : deepseek-v4-flash / glm-5.2 are reasoning-capable -> reasoning:true unlocks Pi's
#                     reasoning request. thinkingFormat + supportsReasoningEffort auto-detect from the
#                     openrouter.ai baseUrl. thinkingLevelMap maps our scale to OpenRouter effort strings
#                     (off => disabled, xhigh clamped to high to dodge Pi issue #4055). deepseek needs
#                     reasoning_content echoed back on assistant messages across turns.
#   - llama-server  : Gemma is a genuine non-reasoner -> reasoning:false is correct and intentional.
ht_pi_models_json() {
  local provider="$1" model="$2" key="$3"
  case "$provider" in
    llama-server)
      jq -n --arg model "$model" '
        {providers: {"llama-server": {
          baseUrl: "http://host.lima.internal:8080/v1",
          api: "openai-completions",
          apiKey: "none",
          models: [{
            id: $model, name: "Gemma 4 26B A4B",
            reasoning: false, input: ["text"],
            contextWindow: 262144, maxTokens: 32768,
            cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
          }],
          compat: {supportsDeveloperRole: false, supportsReasoningEffort: false}
        }}}'
      ;;
    openrouter)
      # Per-model OpenRouter metadata (contextWindow MUST match the real limit — Pi uses it for context
      # management; maxTokens/cost per-Mtok from OpenRouter's model list). Add a row per new model.
      local ctx maxtok cin cout
      case "$model" in
        z-ai/glm-5.2)               ctx=1048576; maxtok=131072; cin=0.9086; cout=2.856 ;;
        deepseek/deepseek-v4-flash) ctx=1048576; maxtok=131072; cin=0.084;  cout=0.168 ;;
        *) ctx=131072; maxtok=32768; cin=0; cout=0
           echo "harness_thinking: '$model' not in the OpenRouter per-model table — conservative defaults (ctx=$ctx, maxTok=$maxtok, cost=0). Add a row if you'll use it repeatedly." >&2 ;;
      esac
      local model_compat='null'
      case "$model" in
        *deepseek*) model_compat='{"requiresReasoningContentOnAssistantMessages": true}' ;;
      esac
      jq -n --arg key "$key" --arg model "$model" \
            --argjson ctx "$ctx" --argjson maxtok "$maxtok" --argjson cin "$cin" --argjson cout "$cout" \
            --argjson compat "$model_compat" '
        {providers: {openrouter: {
          baseUrl: "https://openrouter.ai/api/v1",
          api: "openai-completions",
          apiKey: $key,
          models: [(
            {
              id: $model, name: $model,
              reasoning: true, input: ["text"],
              contextWindow: $ctx, maxTokens: $maxtok,
              thinkingLevelMap: {off: null, minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "high"},
              cost: {input: $cin, output: $cout, cacheRead: 0, cacheWrite: 0}
            } + (if $compat == null then {} else {compat: $compat} end)
          )]
        }}}'
      ;;
    *)  # anthropic
      jq -n --arg key "$key" --arg model "$model" '
        {providers: {anthropic: {
          api: "anthropic-messages",
          apiKey: $key,
          modelOverrides: {($model): {contextWindow: 200000, maxTokens: 65536}}
        }}}'
      ;;
  esac
}

# Remote merge program: fold a single-provider block ($g) into ~/.pi/agent/models.json, PRESERVING every
# other provider. `$old * $pblock` merges provider-level fields (RHS wins on apiKey/baseUrl/compat); models
# arrays are unioned by id (incoming wins) and modelOverrides merged, so even two same-provider pi runs coexist.
# ~/.pi/agent/models.json is a single global file shared by all pi runs — overwriting it clobbers a concurrent
# run's provider, so every run merges instead. The incoming block arrives on stdin.
_HT_PI_MERGE_SCRIPT='
mkdir -p ~/.pi/agent
incoming=$(cat)
[ -s ~/.pi/agent/models.json ] || printf "%s" "{\"providers\":{}}" > ~/.pi/agent/models.json
jq --argjson g "$incoming" "(\$g.providers|to_entries[0]) as \$e | \$e.key as \$pname | \$e.value as \$pblock | (.providers[\$pname]//{}) as \$old | .providers[\$pname]=((\$old*\$pblock)|(if \$pblock.models then .models=((\$pblock.models+(\$old.models//[]))|unique_by(.id)) else . end)|(if \$pblock.modelOverrides then .modelOverrides=((\$old.modelOverrides//{})+\$pblock.modelOverrides) else . end))" ~/.pi/agent/models.json > ~/.pi/agent/models.json.new && mv ~/.pi/agent/models.json.new ~/.pi/agent/models.json
'

# ht_apply_pi_models <ssh_cmd> <provider> <model_id> <api_key>
# Generate the provider block and MERGE it into the sandbox ~/.pi/agent/models.json (never overwrite).
ht_apply_pi_models() {
  local ssh_cmd="$1" provider="$2" model="$3" key="$4"
  ht_pi_models_json "$provider" "$model" "$key" | $ssh_cmd bash -lc "$_HT_PI_MERGE_SCRIPT"
}

# ht_run_config_json <agent_cli> <level> <model> [pi_provider] [pi_model_id] [pi_thinking_flag]
# Emit (on stdout) the resolved, NON-SECRET model-config blob to store in run_metadata.json.config so
# every run carries full reasoning provenance in the DB. Claude → the capability-aware settings object;
# Pi → the resolved models.json entry (baseUrl/models/reasoning/thinkingLevelMap/contextWindow/…) with
# the apiKey stripped. The Pi block is built with a REDACTED placeholder key that is then deleted, so no
# credential can ever reach metadata/DB (the model config is independent of the key value). Used by all
# three condition scripts so the stored shape stays identical across conditions.
ht_run_config_json() {
  local agent_cli="$1" level="$2" model="$3" provider="${4:-}" pi_model="${5:-}" pi_flag="${6:-}"
  if [ "$agent_cli" = "claude" ]; then
    jq -nc --arg model "$model" --arg thinking "$level" \
      --argjson settings "$(ht_claude_settings_object "$level" "$model")" \
      '{harness: "claude-code", model: $model, thinking: $thinking, claude_settings: $settings}'
  else
    jq -nc --arg model "$pi_model" --arg thinking "$level" --arg provider "$provider" --arg flag "$pi_flag" \
      --argjson models "$(ht_pi_models_json "$provider" "$pi_model" REDACTED | jq 'del(.providers[].apiKey)')" \
      '{harness: "pi", model: $model, provider: $provider, thinking: $thinking,
        pi_thinking_flag: $flag, models_json: $models}'
  fi
}
