import yaml
import random
from pathlib import Path


def generate_base_text_snippets():
    """Generate various base text snippets to be repeated/combined for larger samples."""

    snippets = [
        # Tech content
        "Machine learning algorithms process vast amounts of data to identify patterns and make predictions. Neural networks, inspired by biological neural systems, use layers of interconnected nodes to learn complex representations. Deep learning has revolutionized computer vision, natural language processing, and speech recognition through architectures like convolutional networks and transformers.",

        # Science content
        "Climate change affects global weather patterns, ocean currents, and ecosystem dynamics. Rising atmospheric CO2 concentrations drive greenhouse effects, leading to temperature increases, sea level rise, and biodiversity loss. Mitigation strategies include renewable energy adoption, carbon capture technologies, and sustainable land use practices.",

        # Business content
        "Digital transformation involves integrating technology into business operations to improve efficiency, customer experience, and competitive advantage. Cloud computing, artificial intelligence, and data analytics enable organizations to scale operations, automate processes, and derive insights from information. Success requires organizational change management and employee training.",

        # Healthcare content
        "Medical research advances through clinical trials, epidemiological studies, and laboratory investigations. Precision medicine uses genetic information to customize treatments for individual patients. Telemedicine expands healthcare access through remote consultations, monitoring devices, and digital health platforms.",

        # Education content
        "Online learning platforms provide flexible educational opportunities through video lectures, interactive exercises, and virtual classrooms. Adaptive learning systems personalize content delivery based on student performance and preferences. Educational technology enhances engagement through gamification, simulations, and collaborative tools.",

        # Finance content
        "Financial markets facilitate capital allocation through stocks, bonds, derivatives, and alternative investments. Risk management involves portfolio diversification, hedging strategies, and regulatory compliance. Fintech innovations include digital payments, robo-advisors, cryptocurrency, and blockchain applications.",

        # Programming content
        "Software development methodologies like Agile and DevOps emphasize iterative development, continuous integration, and cross-functional collaboration. Modern applications use microservices architectures, containerization, and cloud-native technologies for scalability and reliability. Code quality practices include testing, code reviews, and documentation.",

        # Research content
        "Scientific methodology requires hypothesis formation, experimental design, data collection, statistical analysis, and peer review. Reproducibility ensures research validity through transparent methods, open data sharing, and independent verification. Interdisciplinary collaboration addresses complex problems requiring diverse expertise."
    ]

    return snippets


def create_sample_of_target_tokens(target_tokens, base_snippets):
    """Create a text sample targeting approximately the specified token count."""
    # Rough estimate: ~4 characters per token in English
    target_chars = target_tokens * 4

    current_text = "Respond with exactly \"OK\"\n"

    while len(current_text) < target_chars:
        # Add a random snippet
        snippet = random.choice(base_snippets)
        current_text += snippet + " "

        # Add some variety - sometimes add technical terms, numbers, code
        if random.random() < 0.3:  # 30% chance
            if random.random() < 0.5:
                current_text += f"The implementation requires {random.randint(100, 999)} iterations with parameters alpha={random.uniform(0.01, 0.99):.3f} and beta={random.uniform(0.01, 0.99):.3f}. "
            else:
                current_text += f"""
def process_data(input_data, threshold={random.uniform(0.1, 0.9):.2f}):
    result = []
    for item in input_data:
        if item.confidence > threshold:
            result.append(item.transform())
    return result
"""

        # Sometimes add structured data
        if random.random() < 0.2:  # 20% chance
            current_text += f"""
{{
    "id": {random.randint(1000, 9999)},
    "name": "sample_data_{random.randint(100, 999)}",
    "values": [{', '.join([str(random.randint(1, 100)) for _ in range(5)])}],
    "metadata": {{"version": "{random.randint(1, 5)}.{random.randint(0, 9)}", "created": "2025-01-{random.randint(10, 30)}"}}
}}
"""

    return current_text.strip()


def generate_all_samples():
    """Generate 1000 samples with uniform token count distribution up to 8000 tokens."""
    base_snippets = generate_base_text_snippets()
    samples = []

    # Uniform distribution across 10 bins of 100 samples each:
    # Bin 1: 1-800 tokens (100 samples)
    # Bin 2: 800-1600 tokens (100 samples)
    # Bin 3: 1600-2400 tokens (100 samples)
    # Bin 4: 2400-3200 tokens (100 samples)
    # Bin 5: 3200-4000 tokens (100 samples)
    # Bin 6: 4000-4800 tokens (100 samples)
    # Bin 7: 4800-5600 tokens (100 samples)
    # Bin 8: 5600-6400 tokens (100 samples)
    # Bin 9: 6400-7200 tokens (100 samples)
    # Bin 10: 7200-8000 tokens (100 samples)

    print("Generating 1000 samples in uniform bins...")

    bins = [
        (1, 800),
        (800, 1600),
        (1600, 2400),
        (2400, 3200),
        (3200, 4000),
        (4000, 4800),
        (4800, 5600),
        (5600, 6400),
        (6400, 7200),
        (7200, 8000)
    ]

    for bin_idx, (min_tokens, max_tokens) in enumerate(bins):
        print(f"Generating bin {bin_idx + 1}/10: {min_tokens}-{max_tokens} tokens")

        for i in range(100):
            target_tokens = random.randint(min_tokens, max_tokens)
            sample = create_sample_of_target_tokens(target_tokens, base_snippets)
            samples.append(sample)

            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/100 samples for bin {bin_idx + 1}")

    print(f"Generated {len(samples)} total samples")
    return samples


def generate_synthetic_text_data_if_needed():
    """Generate synthetic test data only if the file doesn't exist."""
    output_path = Path(__file__).parent.parent / "input_data" / "test_data_token_count_comparison.yaml"

    if output_path.exists():
        print(f"Synthetic data already exists at: {output_path}")
        print("Skipping data generation.")
        return

    print("Synthetic data file not found. Generating 1000 samples...")

    samples = generate_all_samples()
    data = {"test_samples": samples}

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, width=1000)

    print(f"Successfully generated {len(samples)} samples and saved to {output_path}")
    print("Distribution: 10 bins of 100 samples each, from 1-8000 tokens")
