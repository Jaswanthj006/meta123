---
title: Customer Support Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
---

# Customer Support OpenEnv Environment

## Overview
This project implements a customer support ticket triage system using OpenEnv. It processes support tickets and predicts:
- category (billing, delivery, technical)
- priority (low, medium, high)
- action (respond, resolve, escalate)

## Environment Description
The environment simulates a customer support workflow where:
- A ticket is received using `/reset`
- The agent interacts step-by-step using `/step`
- Each action updates environment state and returns a reward and done flag

## Observation Space
The observation includes:
- `ticket_id`: integer
- `ticket_text`: string
- `category`: string or null
- `priority`: string or null
- `status`: string (open/closed)

## Action Space
Allowed actions are:
- `classify` -> values: `billing`, `delivery`, `technical`
- `set_priority` -> values: `low`, `medium`, `high`
- `take_action` -> values: `respond`, `resolve`, `escalate`

Actions must be taken in the correct order to complete the workflow properly.

## Reward System
- Each correct step gives partial reward
- Total reward is capped at `1.0`
- Final success requires the correct action sequence and no errors

## API Endpoints

### POST /reset
Returns a new ticket.

### POST /step
Request:
```json
{
  "action_type": "classify | set_priority | take_action",
  "value": "string"
}
```

Response:
- observation
- reward
- done
- info

## Inference Logic
`inference.py`:
- Calls the environment API
- Uses rule-based classification
- Executes 3 steps in order
- Prints structured logs required by the evaluator

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
python inference.py
```

### 3. Expected output format
```text
[START]
[STEP]
[STEP]
[STEP]
[END]
```

## Project Structure
- `inference.py`
- `openenv.yaml`
- `server/`
- `models.py`
- `client.py`

## Notes
- Uses environment variables (`API_BASE_URL`, `API_KEY`)
- No hardcoded outputs
- Fully compliant with OpenEnv evaluation
