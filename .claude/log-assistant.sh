#!/usr/bin/env bash
# Stop hook: append the assistant's response (text since the last real user prompt)
# to prompt-log.md. Deduped by uuid so /clear, /resume, /compact don't re-log.
set -u
LOG=/Users/matt/Desktop/projects/titan/prompt-log.md
STATE=/Users/matt/Desktop/projects/titan/.claude/.last-logged-uuid
DEBUG=/Users/matt/Desktop/projects/titan/.claude/.hook-debug.log
exec 2>>"$DEBUG"
echo "--- $(date '+%Y-%m-%d %H:%M:%S') stop hook fired ---" >>"$DEBUG"

input=$(cat)
tp=$(printf '%s' "$input" | jq -r '.transcript_path // empty')
[ -n "$tp" ] && [ -f "$tp" ] || exit 0

read -r uuid resp < <(jq -sr '
  [.[] | {type, uuid, isMeta: (.isMeta // false), content: .message.content}]
  | reverse
  | (map(.type == "user" and .isMeta == false and (.content | type == "array") and (.content | any(.type == "text"))) | index(true)) as $i
  | (if $i == null then . else .[0:$i] end) | reverse
  | [.[] | select(.type == "assistant")] as $a
  | ($a | last | .uuid // "") + " " + ($a | map(.content // [] | map(select(.type == "text") | .text) | .[]) | join("\n\n") | @json)
' "$tp" 2>/dev/null)

[ -z "${uuid:-}" ] && exit 0
last=$(cat "$STATE" 2>/dev/null || true)
[ "$uuid" = "$last" ] && exit 0

# resp is JSON-encoded; decode it
text=$(printf '%s' "$resp" | jq -r '.')
[ -z "$text" ] || [ "$text" = "null" ] && exit 0

printf '\n## %s (assistant)\n\n%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$text" >> "$LOG"
printf '%s' "$uuid" > "$STATE"
