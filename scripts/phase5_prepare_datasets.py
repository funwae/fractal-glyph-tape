#!/usr/bin/env python3
"""
Phase 5 Dataset Preparation

Generates synthetic multi-turn dialog episodes for testing FGMS context efficiency.
Creates episodes with information mentioned early that must be recalled later.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Templates for generating diverse episodes
EPISODE_TEMPLATES = [
    {
        "category": "tech_decisions",
        "setup_patterns": [
            ("I want to use {tech} for {purpose}.", "{tech} is a great choice for {purpose}."),
            ("Let's go with {tech} for our {component}.", "Good decision on {tech}."),
            ("We should use {tech}.", "Agreed, {tech} is solid."),
        ],
        "follow_up_turns": [
            ("How should we configure it?", "I'll set up the optimal configuration."),
            ("What about performance?", "We can optimize that later."),
            ("Should we add monitoring?", "Yes, monitoring is important."),
            ("What about security?", "I'll add security best practices."),
            ("Any other recommendations?", "Let me think about that."),
        ],
        "question_templates": [
            "What {tech_type} did I choose for {purpose}?",
            "Which {tech_type} did we decide on earlier?",
            "What was my {tech_type} choice?",
        ],
        "examples": [
            {"tech": "PostgreSQL", "purpose": "our database", "component": "persistence layer", "tech_type": "database"},
            {"tech": "Redis", "purpose": "caching", "component": "cache layer", "tech_type": "cache"},
            {"tech": "React", "purpose": "the frontend", "component": "UI framework", "tech_type": "frontend framework"},
            {"tech": "FastAPI", "purpose": "the API", "component": "backend", "tech_type": "API framework"},
            {"tech": "Docker", "purpose": "deployment", "component": "containerization", "tech_type": "deployment tool"},
            {"tech": "AWS Lambda", "purpose": "serverless functions", "component": "compute", "tech_type": "platform"},
            {"tech": "TypeScript", "purpose": "type safety", "component": "type system", "tech_type": "language"},
            {"tech": "GraphQL", "purpose": "our API", "component": "API layer", "tech_type": "API technology"},
        ]
    },
    {
        "category": "configuration",
        "setup_patterns": [
            ("Set the {param} to {value}.", "Setting {param}={value}."),
            ("I want {param} at {value}.", "Configuring {param} to {value}."),
            ("Configure {param} as {value}.", "Done, {param} is now {value}."),
        ],
        "follow_up_turns": [
            ("What about error handling?", "I'll add robust error handling."),
            ("Should we log this?", "Yes, logging is essential."),
            ("Any performance impact?", "Minimal performance impact."),
            ("Is this production-ready?", "Yes, this is production-grade."),
            ("What else do we need?", "Let me review the checklist."),
        ],
        "question_templates": [
            "What did I set the {param} to?",
            "What value did we configure for {param}?",
            "What was the {param} setting?",
        ],
        "examples": [
            {"param": "timeout", "value": "5000ms"},
            {"param": "max_retries", "value": "3"},
            {"param": "batch_size", "value": "100"},
            {"param": "cache_ttl", "value": "300 seconds"},
            {"param": "rate_limit", "value": "1000 requests/min"},
            {"param": "connection_pool_size", "value": "20"},
            {"param": "max_file_size", "value": "10MB"},
            {"param": "session_timeout", "value": "24 hours"},
        ]
    },
    {
        "category": "requirements",
        "setup_patterns": [
            ("We need {feature} for {reason}.", "Great, I'll add {feature}."),
            ("Don't forget {feature}.", "Noted, {feature} is on the list."),
            ("Make sure to include {feature}.", "Will do, {feature} is important."),
        ],
        "follow_up_turns": [
            ("How long will this take?", "I estimate a few hours."),
            ("Any dependencies?", "I'll check for dependencies."),
            ("What's the priority?", "Let me know the priority level."),
            ("Should we test this first?", "Yes, testing is critical."),
            ("Document this please.", "I'll add documentation."),
        ],
        "question_templates": [
            "What feature did I request for {reason}?",
            "What did I say we need for {reason}?",
            "Which {feature_type} was required?",
        ],
        "examples": [
            {"feature": "2FA authentication", "reason": "security", "feature_type": "security feature"},
            {"feature": "real-time notifications", "reason": "user engagement", "feature_type": "feature"},
            {"feature": "data export functionality", "reason": "compliance", "feature_type": "feature"},
            {"feature": "dark mode support", "reason": "accessibility", "feature_type": "UI feature"},
            {"feature": "API rate limiting", "reason": "stability", "feature_type": "protection"},
            {"feature": "automated backups", "reason": "data safety", "feature_type": "backup system"},
            {"feature": "audit logging", "reason": "compliance", "feature_type": "logging"},
            {"feature": "multi-language support", "reason": "internationalization", "feature_type": "i18n feature"},
        ]
    },
]


def generate_episode(episode_id: str, template_idx: int, example_idx: int, num_filler_turns: int = 3) -> Dict[str, Any]:
    """Generate a single episode from a template."""
    template = EPISODE_TEMPLATES[template_idx]
    example = template["examples"][example_idx]

    # Select random setup pattern
    setup_user, setup_assistant = random.choice(template["setup_patterns"])
    setup_user = setup_user.format(**example)
    setup_assistant = setup_assistant.format(**example)

    # Build turns
    turns = [
        {"role": "user", "text": setup_user},
        {"role": "assistant", "text": setup_assistant},
    ]

    # Add filler turns to push the initial info back
    # Use ALL follow-up turns multiple times to make episodes longer
    for _ in range(num_filler_turns):
        for user_text, assistant_text in template["follow_up_turns"]:
            # Add some variation to prevent exact duplicates
            variation = random.choice(["", " Thanks!", " Got it.", " Sounds good."])
            turns.append({"role": "user", "text": user_text + variation})
            turns.append({"role": "assistant", "text": assistant_text})

    # Generate question
    question_template = random.choice(template["question_templates"])
    question = question_template.format(**example)

    # Determine answer
    if template["category"] == "tech_decisions":
        answer = example["tech"]
    elif template["category"] == "configuration":
        answer = example["value"]
    elif template["category"] == "requirements":
        answer = example["feature"]
    else:
        answer = str(list(example.values())[0])

    return {
        "episode_id": episode_id,
        "category": template["category"],
        "turns": turns,
        "question": question,
        "answer": answer,
        "turn_mentioned": 0,  # Always in the first turn for these templates
        "metadata": example
    }


def generate_dataset(
    num_episodes: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate train/val/test splits of episodes."""
    random.seed(seed)

    episodes = []
    episode_num = 0

    # Generate episodes from all templates
    while len(episodes) < num_episodes:
        for template_idx in range(len(EPISODE_TEMPLATES)):
            template = EPISODE_TEMPLATES[template_idx]
            for example_idx in range(len(template["examples"])):
                if len(episodes) >= num_episodes:
                    break

                # Vary number of filler turns (3-7 repetitions of all follow-up turns)
                num_filler = random.randint(3, 7)

                episode = generate_episode(
                    episode_id=f"ep_{episode_num:04d}",
                    template_idx=template_idx,
                    example_idx=example_idx,
                    num_filler_turns=num_filler
                )
                episodes.append(episode)
                episode_num += 1

            if len(episodes) >= num_episodes:
                break

    # Shuffle and split
    random.shuffle(episodes)

    train_end = int(len(episodes) * train_ratio)
    val_end = train_end + int(len(episodes) * val_ratio)

    return {
        "train": episodes[:train_end],
        "val": episodes[train_end:val_end],
        "test": episodes[val_end:]
    }


def save_dataset(dataset: Dict[str, List[Dict[str, Any]]], base_path: Path):
    """Save dataset splits to JSONL files."""
    base_path.mkdir(parents=True, exist_ok=True)

    for split_name, episodes in dataset.items():
        output_path = base_path / f"{split_name}.jsonl"

        with open(output_path, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')

        print(f"✓ Saved {len(episodes)} episodes to {output_path}")


def generate_agent_logs_style(num_episodes: int = 30) -> List[Dict[str, Any]]:
    """
    Generate dataset in the agent logs style.

    This creates more realistic agent interaction logs with mixed topics.
    """
    episodes = []

    for i in range(num_episodes):
        # Randomly select 2-3 templates and mix them
        num_topics = random.randint(2, 3)
        selected_templates = random.sample(range(len(EPISODE_TEMPLATES)), num_topics)

        all_turns = []
        questions = []

        for template_idx in selected_templates:
            template = EPISODE_TEMPLATES[template_idx]
            example = random.choice(template["examples"])

            # Add setup
            setup_user, setup_assistant = random.choice(template["setup_patterns"])
            all_turns.extend([
                {"role": "user", "text": setup_user.format(**example)},
                {"role": "assistant", "text": setup_assistant.format(**example)},
            ])

            # Add 1-2 follow-ups
            for _ in range(random.randint(1, 2)):
                if template["follow_up_turns"]:
                    user_text, assistant_text = random.choice(template["follow_up_turns"])
                    all_turns.extend([
                        {"role": "user", "text": user_text},
                        {"role": "assistant", "text": assistant_text},
                    ])

            # Generate question for this topic
            question_template = random.choice(template["question_templates"])
            question = question_template.format(**example)

            if template["category"] == "tech_decisions":
                answer = example["tech"]
            elif template["category"] == "configuration":
                answer = example["value"]
            else:
                answer = example["feature"]

            questions.append({
                "question": question,
                "answer": answer,
                "topic": template["category"]
            })

        episodes.append({
            "episode_id": f"agent_log_{i:03d}",
            "turns": all_turns,
            "questions": questions,
            "num_turns": len(all_turns),
            "num_topics": len(questions)
        })

    return episodes


def main():
    """Main entry point for dataset generation."""
    print("=" * 70)
    print("Phase 5 Dataset Preparation")
    print("=" * 70)

    base_path = Path("data/phase5")

    # Generate synthetic dialog dataset
    print("\n1. Generating synthetic dialog dataset...")
    dataset = generate_dataset(num_episodes=100, seed=42)

    print(f"\nDataset splits:")
    print(f"  Train: {len(dataset['train'])} episodes")
    print(f"  Val:   {len(dataset['val'])} episodes")
    print(f"  Test:  {len(dataset['test'])} episodes")

    # Save to individual split files
    for split_name, episodes in dataset.items():
        output_path = base_path / f"dialog_{split_name}.jsonl"
        with open(output_path, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')
        print(f"✓ Saved {len(episodes)} episodes to {output_path}")

    # Generate agent logs style dataset
    print("\n2. Generating agent logs dataset...")
    agent_logs = generate_agent_logs_style(num_episodes=50)

    agent_logs_path = base_path / "agent_logs.jsonl"
    with open(agent_logs_path, 'w') as f:
        for episode in agent_logs:
            f.write(json.dumps(episode) + '\n')

    print(f"✓ Saved {len(agent_logs)} agent log episodes to {agent_logs_path}")

    # Create metadata file
    metadata = {
        "created_at": datetime.now().isoformat(),
        "datasets": {
            "dialog": {
                "train": len(dataset['train']),
                "val": len(dataset['val']),
                "test": len(dataset['test']),
                "description": "Synthetic multi-turn dialogs with early information recall"
            },
            "agent_logs": {
                "total": len(agent_logs),
                "description": "Mixed-topic agent interaction logs"
            }
        },
        "templates": len(EPISODE_TEMPLATES),
        "categories": [t["category"] for t in EPISODE_TEMPLATES]
    }

    metadata_path = base_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata to {metadata_path}")

    # Show example episode
    print("\n" + "=" * 70)
    print("Example Episode:")
    print("=" * 70)
    example = dataset['test'][0]
    print(f"\nID: {example['episode_id']}")
    print(f"Category: {example['category']}")
    print(f"\nConversation:")
    for i, turn in enumerate(example['turns']):
        prefix = "User: " if turn['role'] == 'user' else "Assistant: "
        print(f"  {prefix}{turn['text']}")
    print(f"\nTest Question: {example['question']}")
    print(f"Expected Answer: {example['answer']}")

    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
