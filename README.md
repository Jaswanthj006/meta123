---
title: Customer Support Environment
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Customer Support Environment

## Overview
This project is a deterministic customer support simulation environment built for OpenEnv. The idea is simple: give an agent a support ticket, let it process the ticket in a realistic order, and score the quality of each decision.

Each episode follows three steps:
- classify the ticket
- set priority
- take action

Every step is evaluated and contributes to the final reward, so it is easy to debug where the agent is doing well or where it is making bad decisions.

## Observation Space
The environment returns one ticket observation at a time.

```json
{
  "ticket_id": 1,
  "ticket_text": "My payment was charged twice",
  "category": null,
  "priority": null,
  "status": "open"
}
```

Fields:
- `ticket_id`: integer
- `ticket_text`: string
- `category`: string or null
- `priority`: string or null
- `status`: string

## Action Space
The agent sends one action per step.

```json
{
  "action_type": "classify",
  "value": "billing"
}
```

Supported action types:
- `classify` -> values like `billing`, `delivery`, `technical`
- `set_priority` -> values like `low`, `medium`, `high`
- `take_action` -> values like `respond`, `resolve`, `escalate`

The intended flow is to do them in this order: classify, set priority, then take action.

## Reward Logic
Rewards are step-based and deterministic. A better prediction gives a higher reward, partial credit is possible for close decisions, and bad decisions get less reward. The environment also keeps reward output bounded, which makes evaluation stable across runs.

## How to Run Locally
1. Install dependencies:

```bash
pip install .
```

2. Start the API server:

```bash
python -m server.app
```

3. Run inference against the running server:

```bash
python inference.py
```

## API Endpoints

### POST /reset
Starts a new episode and returns the initial ticket observation.

Example:

```bash
curl -X POST http://localhost:7860/reset
```

### POST /step
Applies one agent action and returns the updated observation with reward details.

Request body:

```json
{
  "action_type": "set_priority",
  "value": "high"
}
```

Typical response shape:

```json
{
  "observation": {
    "ticket_id": 1,
    "ticket_text": "My payment was charged twice",
    "category": "billing",
    "priority": "high",
    "status": "open"
  },
  "reward": 0.3,
  "done": false,
  "info": {
    "task_type": "set_priority",
    "score": 1.0
  }
}
```

## Project Structure
- `customer_support_env.py` - core environment logic and grading
- `inference.py` - client-side rollout script for evaluation
- `models.py` - shared Pydantic data models
- `client.py` - lightweight API client helper
- `openenv.yaml` - OpenEnv metadata and spaces
- `server/app.py` - FastAPI app exposing `/reset` and `/step`
- `server/Dockerfile` - container config used by the Space
