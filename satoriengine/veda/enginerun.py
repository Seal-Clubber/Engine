import asyncio
# from .engine import Engine 
from satoriengine.veda.engine import Engine

async def main():
    await Engine.create()
    await asyncio.Event().wait()
    #await asyncio.Future()
    #await asyncio.sleep(10)
    #await asyncio.create_task(client._keepAlive())


    
asyncio.run(main())