import time
from concurrent.futures import TimeoutError

class AgentOrchestrator:
    def __init__(self, agent_timeout_seconds=30, use_mcp=False):
        self.agent_timeout_seconds = agent_timeout_seconds
        self.use_mcp = use_mcp
        self.mcp_client = None

    def _build_agents(self):
        # Logic to build agents
        if self.use_mcp:
            # Pass fresh MCPClient instance for each agent
            mcp_client = MCPClient()
            # build agents with mcp_client

    def _query_agents(self):
        future = ...  # Presumed future that runs the agent
        try:
            result = future.result(timeout=self.agent_timeout_seconds)
            return result
        except TimeoutError:
            return AgentResponse(confidence=0.1, message="Agent execution timed out")
        except Exception as e:
            return AgentResponse(confidence=0.1, message=str(e))

    def close(self):
        if self.mcp_client:
            self.mcp_client.close()

# Update entry point
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-timeout-seconds', type=int, default=30)
    parser.add_argument('--mcp-timeout-seconds', type=int, default=30)

    args = parser.parse_args()
    orchestrator = AgentOrchestrator(agent_timeout_seconds=args.agent_timeout_seconds)
    # further logic
