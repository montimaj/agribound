# Agent

The optional agent layer (`pip install "agribound[agent]"`). See
[Agent layer](../user-guide/agent.md).

::: agribound.agent.agent
    options:
      members:
        - agent
        - AgentResult
        - STATUSES

## Tools

::: agribound.agent.tools
    options:
      members:
        - ToolContext
        - ToolRegistry
        - preflight_execution
        - plan_network_services
      heading_level: 3

## Plans and the confirmation gate

::: agribound.agent.plans
    options:
      members:
        - Plan
      heading_level: 3

::: agribound.agent.gate
    options:
      members:
        - ConfirmationGate
        - prompt_confirm
        - deny_all
        - check_plan_current
      heading_level: 3

## Errors

::: agribound.agent.errors
    options:
      heading_level: 3

## Session transcript

::: agribound.agent.session
    options:
      members:
        - AgentSession
      heading_level: 3

## MCP server

::: agribound.agent.mcp_server
    options:
      members:
        - build_server
        - default_workdir
      heading_level: 3

## Anthropic backend

::: agribound.agent.backends.anthropic_backend
    options:
      members:
        - AnthropicBackend
      heading_level: 3
