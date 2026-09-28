"""Read-only MCP handshake and server-info check; never submits a workflow."""
import asyncio
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    root = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(command=str(root / 'maintenance/run-comfy-mcp.sh'))
    async with asyncio.timeout(90):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool('server_info', {})
                if result.is_error:
                    raise RuntimeError(result.model_dump_json())
                data = {'server': init.server_info.model_dump(), 'tools': [t.name for t in tools.tools],
                        'server_info': result.model_dump(mode='json')}
                out = root / '.toolkit/comfyui/mcp-check.json'
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(data, indent=2) + '\n')
                print(f'MCP handshake and server_info passed; {len(tools.tools)} tools. Report: {out}')

if __name__ == '__main__':
    asyncio.run(main())
