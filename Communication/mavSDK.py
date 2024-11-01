from mavsdk import System
import asyncio

async def connect():
    print('start of run')
    # drone = System(mavsdk_server_address='127.0.0.1')
    # drone = System(mavsdk_server_address='udp://:14550', port=50055)
    # drone = System(mavsdk_server_address='localhost', port=50055)
    drone = System()
    print('drone system object initialized')
    # await drone.connect()
    await drone.connect(system_address="udp://:14540")

    print('waiting to connect')
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered with UUID: {state.uuid}")
            break

async def takeoffLand():
    try:
        drone = System(mavsdk_server_address='localhost', port=50051)
        print('drone initialized')
        await drone.connect()
        print('connected')
        await drone.action.arm()
        print('armed')
        await drone.action.takeoff()
        print('takeoff')
        asyncio.sleep(5)
        await drone.action.land()
        print('land')
    except Exception as e:
        print('error: ', e)

if __name__ == '__main__':
    asyncio.run(connect())
    # asyncio.run(takeoffLand())
