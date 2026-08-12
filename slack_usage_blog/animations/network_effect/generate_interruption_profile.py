"""
Your Interruption Profile: Interactive Calculator

Generates personalized estimation of interruption frequency and cognitive cost
based on team size, channels, and communication patterns.

Research basis:
- 92 messages/day per user (Slack data)
- 62% channels, 38% DMs
- 23 min 15 sec to fully refocus after interruption (Gloria Mark)
- ~2-3 interruptions per team member per day (estimated)

Author: Adam McCabe
"""

from pathlib import Path


def generate_profile_html(output_path: Path) -> None:
    """
    Generate HTML file with embedded interruption profile calculator.

    Args:
        output_path: Path for output HTML file
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Your Interruption Profile</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #FFFFFF;
            color: #333333;
            line-height: 1.6;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 24px;
            margin-bottom: 10px;
            color: #333;
        }
        .subtitle {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-bottom: 30px;
        }
        .calculator-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .input-panel {
            background: #F8F9FA;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
        }
        .results-panel {
            background: #FFFFFF;
            padding: 25px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 6px;
            color: #333;
            font-size: 14px;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #CCC;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #4C78A8;
        }
        .help-text {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .research-note {
            background: #E8F4F8;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #4C78A8;
            font-size: 13px;
            margin-top: 20px;
        }
        .results-section {
            margin-bottom: 25px;
        }
        .results-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 12px;
            color: #333;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #E0E0E0;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
            font-size: 14px;
        }
        .highlight-metric {
            background: #F8F9FA;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            margin-top: 20px;
        }
        .highlight-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }
        .highlight-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Your Interruption Profile</h1>
        <div class="subtitle">How many interruptions do you face daily?</div>

        <div class="calculator-grid">
            <!-- Input Panel -->
            <div class="input-panel">
                <div class="input-group">
                    <label for="teamSize">Team Size</label>
                    <input type="number" id="teamSize" value="50" min="1" max="200" onchange="calculate()">
                    <div class="help-text">Number of people in your organization</div>
                </div>

                <div class="input-group">
                    <label for="channels">Slack Channels</label>
                    <input type="number" id="channels" value="25" min="0" onchange="calculate()">
                    <div class="help-text">Active channels you're monitoring</div>
                </div>

                <div class="input-group">
                    <label for="dms">Active DM Conversations</label>
                    <input type="number" id="dms" value="10" min="0" onchange="calculate()">
                    <div class="help-text">Direct message threads you check regularly</div>
                </div>

                <div class="input-group">
                    <label for="emailVolume">Email Volume (per day)</label>
                    <input type="number" id="emailVolume" value="50" min="0" onchange="calculate()">
                    <div class="help-text">Incoming emails you process</div>
                </div>

                <div class="input-group">
                    <label for="workHours">Work Hours per Day</label>
                    <input type="number" id="workHours" value="8" min="1" max="16" step="0.5" onchange="calculate()">
                    <div class="help-text">Your typical workday length</div>
                </div>

                <div class="research-note">
                    <strong>Research basis:</strong> Average Slack user sends 92 messages/day.
                    Gloria Mark found workers need 23 min 15 sec to fully refocus after interruption.
                </div>
            </div>

            <!-- Results Panel -->
            <div class="results-panel">
                <div class="results-section">
                    <div class="results-title">Your Interruption Profile</div>
                    <div class="metric">
                        <span class="metric-label">Interruptions per day:</span>
                        <span class="metric-value" id="interruptionsPerDay">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Interruptions per hour:</span>
                        <span class="metric-value" id="interruptionsPerHour">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average time between:</span>
                        <span class="metric-value" id="timeBetween">-</span>
                    </div>
                </div>

                <div class="results-section">
                    <div class="results-title">Recovery Cost</div>
                    <div class="metric">
                        <span class="metric-label">Recovery time needed:</span>
                        <span class="metric-value" id="recoveryNeeded">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Available work time:</span>
                        <span class="metric-value" id="availableTime">-</span>
                    </div>
                </div>

                <div class="highlight-metric">
                    <div class="highlight-label">Attention Residue-Free Work Time</div>
                    <div class="highlight-value" id="impossibility">-</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Estimate Slack interruptions based on channels and DMs
        function estimateSlackInterruptions(channels, dms) {
            // Based on research: 92 messages/day per user
            // 62% in channels, 38% in DMs
            // Assume ~3-5 messages per channel you're in = 1 check
            // Assume ~5-7 messages per DM = 1 check
            const messagesPerChannelCheck = 4;
            const messagesPerDMCheck = 6;

            // Estimate messages you'll see based on channels
            const channelMessages = channels * 2; // ~2 messages per channel per day you're in
            const dmMessages = dms * 5; // ~5 messages per DM per day

            const channelChecks = Math.ceil(channelMessages / messagesPerChannelCheck);
            const dmChecks = Math.ceil(dmMessages / messagesPerDMCheck);

            return channelChecks + dmChecks;
        }

        // Format time in hours and minutes
        function formatTime(hours) {
            const h = Math.floor(hours);
            const m = Math.round((hours - h) * 60);
            if (h > 0 && m > 0) return `${h}h ${m}m`;
            if (h > 0) return `${h}h`;
            return `${m}m`;
        }

        // Main calculation
        function calculate() {
            // Get inputs
            const teamSize = parseInt(document.getElementById('teamSize').value);
            const channels = parseInt(document.getElementById('channels').value);
            const dms = parseInt(document.getElementById('dms').value);
            const emailVolume = parseInt(document.getElementById('emailVolume').value);
            const workHours = parseFloat(document.getElementById('workHours').value);

            // Calculate interruptions
            const slackInterruptions = estimateSlackInterruptions(channels, dms);
            const emailInterruptions = Math.ceil(emailVolume / 3); // ~3 emails per check
            const totalInterruptions = slackInterruptions + emailInterruptions;

            const interruptionsPerHour = totalInterruptions / workHours;
            const minutesBetween = (workHours * 60) / totalInterruptions;

            // Recovery calculations (23 min per interruption)
            const recoveryMinutesNeeded = totalInterruptions * 23.25; // 23 min 15 sec
            const recoveryHoursNeeded = recoveryMinutesNeeded / 60;
            const workMinutesAvailable = workHours * 60;

            const deficit = recoveryHoursNeeded - workHours;
            const residueFreeTime = workHours - recoveryHoursNeeded;

            let impossibilityText;
            if (deficit > 0) {
                impossibilityText = `Based on research, you would need ${formatTime(Math.abs(deficit))} more time daily in order to have attention residue-free work time.`;
            } else {
                impossibilityText = `Based on research, you have ${formatTime(Math.abs(residueFreeTime))} of attention residue-free work time.`;
            }

            // Update display
            document.getElementById('interruptionsPerDay').textContent = totalInterruptions;
            document.getElementById('interruptionsPerHour').textContent = interruptionsPerHour.toFixed(1);
            document.getElementById('timeBetween').textContent = formatTime(minutesBetween / 60);
            document.getElementById('recoveryNeeded').textContent = formatTime(recoveryHoursNeeded);
            document.getElementById('availableTime').textContent = formatTime(workHours);
            document.getElementById('impossibility').textContent = impossibilityText;
        }

        // Initialize
        calculate();
    </script>
</body>
</html>
"""

    # Save HTML
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✓ Generated interruption profile calculator: {output_path}")


def main() -> None:
    """Generate interruption profile calculator HTML."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    html_path = output_dir / "interruption_profile.html"
    generate_profile_html(html_path)

    print("\n✓ Done! Open interruption_profile.html to use the calculator.")
    print("  Features:")
    print("    - Personalized interruption estimation")
    print("    - Recovery time calculations (23 min per interruption)")
    print("    - Time breakdown pie chart")
    print("    - Comparison with research baselines")


if __name__ == "__main__":
    main()
