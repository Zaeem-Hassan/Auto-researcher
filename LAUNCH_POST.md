# X Launch Post

---

We applied the @karpathy autoresearch loop to voice AI agents. Open source.

Your voice agent has a system prompt. That prompt determines how it handles every call. Most teams write it once, test manually, and hope for the best.

autovoiceevals makes it a loop. One artifact (system prompt), one metric (adversarial eval score), keep what improves it, revert what doesn't. Run it overnight. Wake up to a better agent.

How it works:

You describe your agent in a config file — what it does, its services, policies, and what it should never do. That's it. You don't write test cases or define attack vectors.

```yaml
provider: vapi

assistant:
  id: "your-agent-id"
  description: |
    Voice receptionist for a hair salon. 
    Maria does coloring only. Jessica does cuts only.
    $25 cancellation fee under 24 hours notice.
    Cannot advise on skin conditions. Closed Sundays.
```

From that, Claude generates adversarial caller personas — someone insisting Maria do their haircut (she only does coloring), someone demanding a Sunday appointment, someone asking if a chemical treatment is safe for their rash (medical advice), someone trying to learn another client's appointment time (privacy probe).

Each persona gets a full voice profile — accents simulated through broken grammar, background noise like construction sites or crying children, fast mumblers who drop word endings, slow speakers with long pauses, callers who interrupt constantly. The caller scripts reflect these characteristics, so the eval tests how your agent handles real-world communication degradation, not just clean text.

The eval suite is generated once and held fixed — like a validation set. Every experiment is measured against the same scenarios.

The loop:

1. Read the agent's current prompt from the platform
2. Generate adversarial eval suite from your description
3. Run baseline
4. Claude proposes ONE surgical change to the prompt
5. Push modified prompt to the agent via API
6. Run all scenarios against the updated agent
7. Score improved? Keep. Same score but shorter prompt? Keep. Otherwise revert.
8. Go to 4. Run until Ctrl+C.

Scoring: 50% on whether the agent did what it should, 35% on whether it avoided what it shouldn't, 15% on response latency. All configurable.

The system sees its own history. When a change fails, the next proposal knows what was tried and why it didn't work. When 4 different wordings of the same idea all fail, it tries a fundamentally different approach.

Results from real runs:

Dental scheduling agent on Vapi — 20 experiments, 8 adversarial scenarios, ~75 min, ~$18:
- Score: 0.714 → 0.847 (+19%)
- CSAT: 48 → 77
- Pass rate: 38% → 88%
- 6 kept, 14 discarded

The system found that humor was hurting a medical-context agent, added emergency protocols the prompt was missing, tried 4 different wordings for communication barriers before one worked, and tried 6 identity verification phrasings before landing on the right one. It also found that removing redundant instructions improved scores — simpler prompts outperformed longer ones.

Real estate assistant on Smallest AI — 5 experiments, 5 scenarios:
- Score: 0.875 → 0.925
- Prompt: 6,615 → 4,719 chars (29% shorter)

Improved the agent while deleting 2,000 characters of over-engineered instructions. Redundant personality guidance, verbose conversation flows, and meta-commentary sections were all removed because they weren't helping.

The eval suite adapts to any domain. Describe a pizza delivery agent and you get attacks about off-menu orders, delivery zone boundaries, and expired coupons. Describe a banking agent and you get social engineering and account fishing. You describe. It breaks.

Getting started:

```
git clone https://github.com/ArchishmanSengupta/autovoiceevals.git
cd autovoiceevals
pip install -r requirements.txt
cp .env.example .env
cp examples/vapi.config.yaml config.yaml
python main.py research
```

Works with @vaboratory and @smallest_ai. Two API keys and a config file. ~$0.90/experiment. The description is the only input — the better you describe your agent, the sharper the attacks.

Open source. MIT.

github.com/ArchishmanSengupta/autovoiceevals
