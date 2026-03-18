---
name: skill-generator
description: >
  Generates a new reusable skill (SKILL.md) from a described or observed workflow.
  Use when the user says "create a skill", "make a skill from this flow", "turn this
  into a skill", "save this as a skill", or "generate a skill that does X".
  Produces a ready-to-use SKILL.md file in .claude/skills/<skill-name>/.
allowed-tools: Read, Write, Bash, Glob
---

You are generating a new Claude Code skill from a workflow the user has described or demonstrated.

A skill is a reusable instruction file that Claude loads at runtime to carry out a specific
repeatable task. It lives at `.claude/skills/<skill-name>/SKILL.md`.

## Step 1 — Gather information

If the user has not already provided all of the following, ask before proceeding:

1. **Skill name** — short, lowercase, hyphenated (e.g. `api-smoke-test`, `db-seed`, `deploy-staging`)
2. **What the skill does** — one sentence description
3. **The workflow** — the sequence of steps, commands, or actions the skill should perform
4. **Trigger phrases** — when should Claude automatically invoke this skill? (e.g. "run smoke tests", "seed the database")
5. **Tools needed** — which tools does the skill use? (e.g. `Bash`, `playwright-cli:*`, `Read`, `Write`)

If the user has already shown or described a workflow in the conversation, extract this
information from context instead of asking again.

## Step 2 — Determine the skill type

Classify the workflow into one of these types and apply the corresponding body structure:

### Type A — Command automation (runs shell commands / CLI tools)
Use numbered steps with bash code blocks. Include verification steps after each action.
Show expected outputs where helpful.

### Type B — Reasoning / analysis (no shell commands, just instructions for Claude)
Use prose sections: what to read, how to analyze, what to output. Mirror the
`file-summarizer` style — clear headings, no code blocks unless showing output format.

### Type C — Hybrid (mix of shell commands and Claude reasoning)
Combine both: prose for the decision-making parts, bash blocks for the execution parts.

## Step 3 — Write the SKILL.md

Produce a complete SKILL.md with this structure:

```
---
name: <skill-name>
description: >
  <one or two sentence description of what the skill does and when to trigger it.
  Include 3–5 example trigger phrases so the skill loads reliably.>
allowed-tools: <comma-separated list, e.g. Bash, Read, Write>
---

# <Skill Title>

## Overview
<1–3 sentences: what this skill does and why it exists>

## When to use
- <trigger phrase 1>
- <trigger phrase 2>
- <trigger phrase 3>

## Steps

### Step 1 — <Name>
<instructions or bash block>

### Step 2 — <Name>
<instructions or bash block>

... (as many steps as needed)

## Expected output
<What the user should see when the skill completes successfully>

## Failure handling
<What to do if a step fails — retry, report, skip, etc.>
```

Rules for writing the body:
- Be prescriptive — write instructions Claude will follow exactly, not vague suggestions
- Use `eval` or verification commands after state-changing steps to confirm success
- For screenshot/artifact rules, follow any conventions already set in existing skills
  (e.g. screenshots only on failure if that rule exists in playwright-cli/SKILL.md)
- Keep steps atomic — one clear action per step
- Do not include information that can be derived from reading the codebase at runtime

## Step 4 — Write the file

Write the completed SKILL.md to:

```
.claude/skills/<skill-name>/SKILL.md
```

Create the directory if it does not exist.

## Step 5 — Confirm

Tell the user:
- The file path written
- The skill name (as it will appear in the skills list)
- The trigger phrases that will activate it
- Any follow-up they may need (e.g. add `allowed-tools` permissions via update-config)
