import argparse

# Existing code...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-timeout-seconds', type=float, default=30,
                        help='Timeout for the agent in seconds')
    args = parser.parse_args()
    orchestrator = AgentOrchestrator(..., agent_timeout_seconds=args.agent_timeout_seconds)
    # Continue with existing logic...