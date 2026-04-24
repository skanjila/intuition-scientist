"""MCP client and tool-backend package.

Exports
-------
MCPClient
    DuckDuckGo web-search client (original MCP implementation).
ToolBackend
    Protocol that every tool back-end must satisfy.
VectorStoreBackend
    In-memory TF-IDF / remote vector-store retrieval.
TicketToolBackend
    Helpdesk / CRM ticket search (Zendesk, Intercom, Jira).
CRMToolBackend
    CRM account/opportunity search (Salesforce, HubSpot).
ObservabilityToolBackend
    Live metric + alert retrieval (Datadog, Prometheus, Grafana).
StructuredDataToolBackend
    Deterministic ledger-to-invoice three-way match engine.
ERPToolBackend
    ERP inventory / PO / supplier data (SAP, Oracle, NetSuite).
DataWarehouseToolBackend
    Cloud data-warehouse analytics queries (BigQuery, Snowflake, Redshift).
"""

from src.mcp.mcp_client import MCPClient
from src.mcp.tool_backend import ToolBackend
from src.mcp.vector_store_backend import VectorStoreBackend
from src.mcp.ticket_tool import TicketToolBackend
from src.mcp.crm_tool import CRMToolBackend
from src.mcp.observability_tool import ObservabilityToolBackend
from src.mcp.structured_data_tool import StructuredDataToolBackend
from src.mcp.erp_tool import ERPToolBackend
from src.mcp.datawarehouse_tool import DataWarehouseToolBackend

__all__ = [
    "MCPClient",
    "ToolBackend",
    "VectorStoreBackend",
    "TicketToolBackend",
    "CRMToolBackend",
    "ObservabilityToolBackend",
    "StructuredDataToolBackend",
    "ERPToolBackend",
    "DataWarehouseToolBackend",
]
